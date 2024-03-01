import math
import numpy as np
import pulp
import os
import sys
import dataclasses
sys.path.append('/home/server/DistributedOffload/')
from flexgen.compression import CompressionConfig
from flexgen.opt_config import get_opt_config
from flexgen.flex_opt import Policy
from flexgen.utils import GB,MB, T
from collections import defaultdict


@dataclasses.dataclass
class CostModelConfig:
    s: int = 32#512 # 输入句子句子长度
    n: int = 512#128 # 生成长度

    l: int = 96 # 96层
    h1: int = 12288 # attention 的隐藏层
    h2: int = 12288 * 4 # MLP 的隐藏层
    nh: int = 96

    gmem: int = (8 -2) * GB
    cmem: int = 204 * GB
    nmem: int = 1500 * GB
        
    btype: int = 2 # float16
    batch_size: int=2


def init_cost_config(num_layer,opt_config):
    config = CostModelConfig()
    config.l = opt_config.num_hidden_layers
    config.h1 = opt_config.hidden_size
    config.h2 = opt_config.ffn_embed_dim
    config.nh = opt_config.n_head
    return config
def get_cache_percentage(wg,wc,wn,num_layer,tp,opt_config,s,n,batch_size,compress_w=False, compress_cache=False): # TODO:cache 的tp
    config = init_cost_config(num_layer,opt_config)
#     s = config.s
#     n = config.n
    l = num_layer #config.l
    h1 = config.h1
    h2 = config.h2
    nh = config.nh
    gbs = batch_size
    bls = gbs * 1
    hg = 1
    hc = 0
    hn = 0
    gmem = config.gmem 
    cmem = config.cmem 
    nmem = config.nmem 
    # layer weight size
    wi = (8 * h1 ** 2 + 4 * h1 * h2) / tp 
    if compress_w:
        wi = wi / 4
    ## GPU peak memory uconstaints
    # gpu_home_p == wi * l * wg + 2 * s * h1 * bls * hg + 4 * (s + n) * h1 * bls * l * cg
    interp = 8 * gbs * s * h1 \
                    + gbs * (2 * s * h1 + 2 * nh * s ** 2) \
                    + gbs * (2 * s * h1) \
                    + 4 * gbs * s * h1 \
                    + 2 * gbs * s * h2 \
                    + 2 * gbs * s * h1
    gpu_w_p = 2 * wi * (1 - wg) + 2 * s * h1 * gbs * (1 - hg) \
                     + interp
    cg1 = (gmem - gpu_w_p - (wi * l * wg + 2 * s * h1 * bls * hg)) /  (4 * (s + n) * h1 * bls * l) 

    de_cg = wi * l * wg + 2 * h1 * bls * hg + 2 * wi * (1 - wg) + 2 * h1 * gbs * (1 - hg) \
                     + 8 * gbs * h1 \
                    + 4 * gbs * h1 \
                    + 2 * gbs * h2 \
                    + 2 * gbs * h1
    denominator = (4 * (s + n) * h1 * bls * l + 2 * 2 * gbs * (s + n) * h1 + gbs * (2 * h1 + 2 * (s + n) * h1 + 2 * nh * (s + n) + 2 * (s + n) * h1 + 2 * h1))
    if tp > 1:
        cg2 = (gmem-de_cg) / (denominator/tp)
    elif compress_cache:
        cg2 = (gmem-de_cg) / (denominator)
    else:
        cg2 = (gmem-de_cg-1.5*GB) / (1.5*denominator)
    enough_gmem = False
    enough_cmem = False
    cg = min(cg1,cg2)
    if cg < 0:
        #print("cg < 0 can't put",cg1,cg2,cg)
        return None
    if cg > 1:
        #print("GPU memory is enough for cache", cg1,cg2,cg,gmem)
        cg = 1
        enough_gmem = True
        enough_cmem = True
        
    cpu_w_p = wi * (1 - wg) + 2 * s * h1 * gbs * (1 - hg)
    cc1 = (cmem - cpu_w_p - (wi * l * wc + 2 * s * h1 * bls * hc)) / 4 * s * h1 * bls * l
    de_cc = wi * l * wc + 2 * h1 * bls * hc + wi * wn + 4 * h1 * gbs * hn + 8 * (s + n) * h1 * gbs * (1-cg) + 2 * nh * (s + n) * gbs + 2 * h1 * gbs 
    denominator = (4 * (s + n) * h1 * bls * l - 8 * (s + n) * h1 * gbs) 
    if denominator <= 0 and (cmem - de_cc) >= 0:
        cc = 0
    else:
        cc2 = (cmem - de_cc)/ denominator
        cc = min(cc1, cc2)
    if cc > 1:
        #print("CPU memory is enough large, can stop iterating", cc)
        enough_cmem = True
        cc = 1 - cg
        if (1-cg-cc) * 4 * (s + n) * h1 * bls  * l  > (nmem  - (wi * l * wn + 2 * s * h1 * bls * hn)):
            #print("error")
            return None
        if cg < 0 or cc < 0:
            #print("can't put",cg,cc)
            return None
        return (cg,cc,1-cg-cc,enough_cmem,enough_gmem)
    if (1-cg-cc) * 4 * (s + n) * h1 * bls  * l  > (nmem  - (wi * l * wn + 2 * s * h1 * bls * hn)):
        #print("error")
        return None
    if cg < 0 or cc < 0:
        #print("can't put",cg,cc)
        return None
    return (cg,cc,1-cg-cc,enough_cmem,enough_gmem)


# +
def get_setting(model_name,s,n,batch_size,compress_w=False,compress_cache=False): # s 输入句子句子长度 n 输出长度
    opt_config = get_opt_config(model_name)
    layers_num = opt_config.num_hidden_layers
    device_profile_latency = []
    percentages = [0.4804097702687206,0.9804097702687206,0.12497384754864356,0.24996513006485807,0.3749564125810726,0.4999476950972871,0.6249389776135016,0.7499302601297162,0.8749215426459307,0.9999128251621452,0.9999476950972871,0.999982565032429,0.2499694861467106,0.4999738452686091,0.7499782043905076,0.9999564087810151,0.9999738452686091,0.999991281756203,9.945498667303179e-06,2.9836496001909535e-05,0.5000198909973346]
    #percentages = [0.4804097702687206,0.9804097702687206,0.12475633528265107,0.24967511371020143,0.3745938921377518,0.49951267056530213,0.6244314489928525,0.7493502274204028,0.8742690058479532,0.9991877842755036,0.9995126705653021,0.9998375568551007,0.24971549341570476,0.49975613721346124,0.7497967810112177,0.9995935620224353,0.9997561372134612,0.999918712404487,9.945498667303179e-06,2.9836496001909535e-05,0.5000198909973346]
    deduplicated_percentages = [0]
    # 去重
    for p in percentages:
        is_add = True
        for d in deduplicated_percentages:
            if abs(p - d) < 0.01:
                is_add = False
                break
        if is_add:
            deduplicated_percentages.append(p)
    deduplicated_percentages.sort()
    cnt = 0
    profile_setting = [[[] for hold_layer_num in range(1,layers_num+2)] for tp in range(5)]
    for tp in [1,2,4]:
        for hold_layer_num in range(1,layers_num+1): # 这个假设成线性
            enough_cpu_memory = False
            for i in  range(0,len(deduplicated_percentages)):
                disk_percent = deduplicated_percentages[i]
                for j in range(i,len(deduplicated_percentages)):
                    cpu_percent = deduplicated_percentages[j] - disk_percent
                    gpu_percent = 1 - disk_percent - cpu_percent
                    cache_percentage = get_cache_percentage(gpu_percent, cpu_percent,disk_percent,hold_layer_num,tp,opt_config,s,n,batch_size,compress_w,compress_cache)
                    if cache_percentage is not None:
                        enough_cpu_memory = cache_percentage[-2]
                        if cache_percentage[-1] == True: # enpugh GPU mem
                            cnt += 1
#                             gpu_percent = max(0,gpu_percent-0.01)
#                             cpu_percent = 1 - gpu_percent
                            profile_setting[tp][hold_layer_num].append((gpu_percent, cpu_percent, cache_percentage[0],cache_percentage[1]))
                            break
                        cnt += 1
                        profile_setting[tp][hold_layer_num].append((gpu_percent, cpu_percent, cache_percentage[0],cache_percentage[1]))
                if enough_cpu_memory:
                    break
            if len(profile_setting[tp][hold_layer_num])>1:
                profile_setting[tp][hold_layer_num]=profile_setting[tp][hold_layer_num][0:]
    return profile_setting
model_name='/home/server/models/OPT-30B'
name = model_name.split("/")[-1]
opt_config = get_opt_config(model_name)
# for k in [32,64,128,246,512]:
#     # get_cache_percentage(wg,wc,wn,num_layer,tp,opt_config,s,n,batch_size)
#     print(get_cache_percentage(0.5,0.5,0,24,1,opt_config,32,k,4,True,True))
p = get_setting(model_name,128,512,4,True,False)
p[1][12]
# tps = [1,2,4]
# kk = defaultdict(lambda: defaultdict(list))
# for tp in tps:
#     aa = np.load(f'/home/server/DistributedOffload/profile/profile_{name}_{tp}_new.npy',allow_pickle=True)
#     print(len(aa))
#     for a in aa:
#         kk[a[0]][a[1]].append(a[2:])
# opt_config = get_opt_config(model_name)
# layers_num = opt_config.num_hidden_layers
# for tp in [1,2,4]:
#     for hold_layer_num in range(1,layers_num+1): # 这个假设成线性tp
#         print(tp,hold_layer_num,p[tp][hold_layer_num])
# 基于剪枝 或者sgd的动态规划


# 我是否考虑数据并行？
# 已知张量并行比较快，有网络限制，显存不足可以卸载

# +
@dataclasses.dataclass
class VM:
    index: int
    name : str
    num_dev : int
    gmem : float
    cmem : float
    nmem : float
    dev_bandwidth : float # GB/s
    cost : float # yuan
    
# devs = [VM(0,'T4',4,16 * GB,204 * GB,1500 * GB,484.639,68.75), # GB/s
#         VM(1,'T4',2,16 * GB,204 * GB,1500 * GB,484.639,34.38),
#         VM(2,'T4',1,16 * GB,204 * GB,1500 * GB,484.639,11.63),
#         VM(3,'V100',4,16 * GB,204 * GB,500 * GB,484.639,105.84),
#         VM(4,'V100',1,16 * GB,204 * GB,500 * GB,484.639,26.46)]
#devs = [
#         VM(0,'T4',4,16 * GB,204 * GB,1500 * GB,484.639 * GB,68.75), # GB/s
#         VM(1,'T4',2,16 * GB,204 * GB,1500 * GB,484.639 * GB,34.38),
#         VM(2,'T4',1,16 * GB,204 * GB,1500 * GB,484.639 * GB,11.63),
#         VM(0,'V100',2,16 * GB,204 * GB,500 * GB,484.639 * GB,105.84),
#         VM(1,'V100',2,16 * GB,204 * GB,500 * GB,484.639 * GB,105.84),
#         VM(2,'V100',2,16 * GB,204 * GB,500 * GB,484.639 * GB,105.84),
#         VM(3,'V100',1,16 * GB,204 * GB,500 * GB,484.639 * GB,26.46)]

devs = [VM(0,'V100',1,16 * GB,204 * GB,500 * GB,484.639 * GB,105.84),
       VM(0,'V100',1,16 * GB,204 * GB,500 * GB,484.639 * GB,105.84),
       VM(0,'V100',1,16 * GB,204 * GB,500 * GB,484.639 * GB,105.84),
       VM(0,'V100',1,16 * GB,204 * GB,500 * GB,484.639 * GB,105.84),
       VM(1,'V100',2,16 * GB,204 * GB,500 * GB,484.639 * GB,105.84),
       VM(1,'V100',2,16 * GB,204 * GB,500 * GB,484.639 * GB,105.84)]
#devs = sorted(devs, lambda x: x.cost)
latency_matrix = [[200/1000 for j in range(0,len(devs))] for i in range(len(devs))] # 200ms
bandwidth_matrix = [[0.1 * GB/8 for j in range(0,len(devs))] for i in range(len(devs))] # 0.1Gbit
for i in range(len(devs)):
    latency_matrix[i][i]=0
    bandwidth_matrix[i][i]=484.639 * GB

def get_tp_prefill_latency(tp_devs,cost_config):
    batch_size = cost_config.batch_size
    num_dev = tp_devs.num_dev
    dev_bandwidth = tp_devs.dev_bandwidth
    # hidden_size=768, input_dim=768
    data_size = batch_size * cost_config.s *  cost_config.h1 * cost_config.btype
    return (num_dev -1) * data_size / (num_dev * dev_bandwidth) * 2 # 2 表示有两次
    
def get_tp_gen_latency(tp_devs,cost_config):
    # tensor_model_parallel_all_gather 可以忽略不计，太短了
    batch_size = cost_config.batch_size
    num_dev = tp_devs.num_dev
    dev_bandwidth = tp_devs.dev_bandwidth
    # hidden_size=768, input_dim=768
    data_size = batch_size * 1 *  cost_config.h1 * cost_config.btype
    return (num_dev -1) * data_size / (num_dev * dev_bandwidth) * 2 * cost_config.n # 要不要乘以n 如果延迟是以per token来比较的话？

def get_pp_prefill_latency(pre,tp_devs,cost_config):
    batch_size = cost_config.batch_size
    data_size = batch_size * cost_config.s *  cost_config.h1 * cost_config.btype
    return latency_matrix[pre.index][tp_devs.index] + data_size / bandwidth_matrix[pre.index][tp_devs.index]
def get_pp_gen_latency(pre,tp_devs,cost_config):
    batch_size = cost_config.batch_size
    data_size = batch_size * cost_config.n *  cost_config.h1 * cost_config.btype
    return latency_matrix[pre.index][tp_devs.index] + data_size / bandwidth_matrix[pre.index][tp_devs.index]

# model_name = '/home/server/models/OPT-30B'
# opt_config = get_opt_config(model_name)
# cost_budget = 10000
# name = model_name.split("/")[-1]
# tps = [1,2,4]

# kk = defaultdict(lambda: defaultdict(list))
# for tp in tps:
#     for layer in range(1,opt_config.num_hidden_layers+1):
#         aa = np.load(f'/home/server/DistributedOffload/profile/profile_{name}_{tp}_{layer}_new_0.npy',allow_pickle=True)
#         for a in aa:
#             kk[a[0]][a[1]].append(a[2:])
def get_computing_latency(tp_dev,pp_layers): # per token TODO
    min_latency = 3/(10*tp_dev.num_dev)  #9999999999
    min_profile = [1/(10*tp_dev.num_dev) ,2/(10*tp_dev.num_dev) ]
#     for value in kk[tp_dev.num_dev][pp_layers]:
#         sum_ = value[-1] + value[-2]
#         if sum_ < min_latency:
#             min_latency = sum_
#             min_profile = value
    return min_latency,min_profile
cost_budget = 10000
beam_search_k = 1
def get_best_devs(pres, devss,layer_num=1,pre=None): # 层数和前面的 TODO
    results = []
    if pres is None:
        for devs in devss:
            for tp_dev in devs:
                cost_config = init_cost_config(layer_num,opt_config)
                tp_latency = get_tp_prefill_latency(tp_dev,cost_config) + get_tp_gen_latency(tp_dev,cost_config)
                min_computing_latency,min_profile = get_computing_latency(tp_dev,layer_num)
                all_latency = tp_latency * layer_num + min_profile[-2] * cost_config.s +min_profile[-1] * cost_config.n
                best_dev = ([tp_dev],layer_num,min_profile,tp_dev.cost,all_latency,devs,tp_dev)
                results.append(best_dev)
        results.sort(key=lambda s:(s[4],s[3]))
        return results
    for i in range(len(pres)):
        pre = pres[i]
        devs = devss[i]
        for tp_dev in devs:
            if pre is None or pre[3] + tp_dev.cost > cost_budget:
                continue
            latency,_ =get_computing_latency(tp_dev,layer_num)
            cost_config = init_cost_config(layer_num,opt_config)
            tp_latency = get_tp_prefill_latency(tp_dev,cost_config) + get_tp_gen_latency(tp_dev,cost_config)
            pp_latency = get_pp_prefill_latency(pre[0][-1],tp_dev,cost_config) + get_pp_gen_latency(pre[0][-1],tp_dev,cost_config)# pre latency
            min_computing_latency,min_profile = get_computing_latency(tp_dev,layer_num)
            # 按照距离排序，再按照显存排序
            all_latency = max(pre[4], layer_num * tp_latency + pp_latency + min_profile[-2] * cost_config.s +min_profile[-1] * cost_config.n)
            pp = pre[0].copy()
            pp.append(tp_dev)
            best_dev = (pp,i,min_profile,pre[3] + tp_dev.cost,all_latency,devs,tp_dev)
            results.append(best_dev)
    results.sort(key=lambda s:(s[4],s[3]))
    return results

map_ = defaultdict()
def DP(k,j,devs,opt_config): # 分成j个stage，前k层给前j个stage
    if f"{k}_{j}" in map_:
        return map_[f"{k}_{j}"]
    min_latency = 999999999999
    best_dev = None
    if j == 1: # 只剩一个stage
        results = get_best_devs(None,[devs],k) # 得到已经排列的集合
        results = results[:beam_search_k] 
        left_devs = []
        for result in results:
            new_devs = result[5].copy()  # 原始devs （还没有remove的）
            new_devs.remove(result[6]) # 移除对应的dev
            left_devs.append(new_devs) # 集合的集合
        map_[f"{k}_{j}"] = (results,left_devs,devs)
        return results,left_devs,devs # 最好的集合，remove的设备集合的集合，原始集合
    results = []
    for i in range(j,k+1):
        pres,left_devs,_ = DP(i-1,j-1,devs,opt_config)
        tp_devs = get_best_devs(pres,left_devs,k-i)
        results += tp_devs
    results.sort(key=lambda s:(s[4],s[3]))
    results = results[:beam_search_k]
    left_devs = []
    for result in results:
        new_devs = result[5].copy()
        new_devs.remove(result[6])
        left_devs.append(new_devs)
    map_[f"{k}_{j}"] = (results,left_devs,devs)
    return results,left_devs,devs
best_dev = None
min_latency = 999999999999
stage_num = 0
import time
begin = time.time()
final_results = []
for j in range(1,min(opt_config.num_hidden_layers,len(devs)+1)): # 遍历分成多少个stage比较好               
    results,left_devss,_ = DP(opt_config.num_hidden_layers,j,devs,opt_config)
    final_results += results
final_results.sort(key=lambda s:(s[4],s[3]))
end=time.time()
print(end-begin,final_results[0])
# kk


# +
# profile = np.load('/home/server/DistributedOffload/profile_OPT-125M.npy')
# min_latency = 9999999999
# min_profile = None
# for tp_dev in devs:
#     for pp_layers in range(opt_config.num_hidden_layers):
#         for p in profile:
#             if p[0] == tp_dev.num_dev and p[1] == pp_layers:
#                 if p[-1] < min_latency:
#                     min_latency = p[-1]
#                     min_profile = p

# +
# tp = 2
# model_name = '/home/server/models/OPT-30B'
# opt_config = get_opt_config(model_name)
# name = model_name.split("/")[-1]
# profile = [[None for hold_layer_num in range(1,layers_num+1)] for tp in [1,2,4]]
# file = []
# for layer_num in range(1,opt_config.num_hidden_layers+1):
#     profile_data = np.load(f'/home/server/DistributedOffload/profile/profile_{name}_{tp}_{layer_num}.npy')
#     file.append(profile_data)
# np.save(f'/home/server/DistributedOffload/profile/profile_{name}_{tp}',file)


# +
# model_name = '/home/server/models/OPT-30B'
# opt_config = get_opt_config(model_name)
# name = model_name.split("/")[-1]
# tps = [2]

# kk = defaultdict(lambda: defaultdict(list))

# for tp in tps:
#     aa = np.load(f'/home/server/DistributedOffload/profile/profile_{name}_{tp}.npy',allow_pickle=True)
#     for a in aa:
#         for k in a:
#             is_duplicated = False
#             for g in kk[k[0]][k[1]]: 
#                 if list(g) == list(k[2:]):
#                     is_duplicated = True
#                     break
#             if not is_duplicated:
#                 kk[k[0]][k[1]].append(k[2:])
# profile = []
# for key,value in kk.items():
#     for key1,value2s in value.items():
#         for value2 in value2s:
#             arr = [key,key1]
#             arr.extend(value2)
#             profile.append(arr)
# -

# backup
def DP(k,j,devs,opt_config): # 分成j个stage，前k层给前j个stage
    min_latency = 999999999999
    best_dev = None
    if j == 1:
        for tp_dev in devs:
            if tp_dev.cost > cost_budget:
                continue
            cost_config = init_cost_config(k,opt_config)
            tp_latency = get_tp_prefill_latency(tp_dev,cost_config) + get_tp_gen_latency(tp_dev,cost_config)
            min_computing_latency,min_profile = get_computing_latency(tp_dev,k)
            all_latency = tp_latency * k + min_profile[-2] * cost_config.s +min_profile[-1] * cost_config.n
            #memory[f'{k}_{j}_{tp_dev.index}'] = all_latency
            if all_latency < min_latency:
                min_latency = all_latency
                best_dev = ([tp_dev],k,min_profile,tp_dev.cost,min_latency)
        return best_dev
    for i in range(1,k):
        if j-1 > i:
            continue
        for tp_dev in devs:
            new_devs = devs.copy()
            new_devs.remove(tp_dev)   
            pre = DP(i,j-1,new_devs,opt_config)
            if pre is None or pre[-2] + tp_dev.cost > cost_budget:
                continue
            cost_config = init_cost_config(k-i,opt_config)
            tp_latency = get_tp_prefill_latency(tp_dev,cost_config) + get_tp_gen_latency(tp_dev,cost_config)
            pp_latency = get_pp_prefill_latency(pre[0][-1],tp_dev,cost_config) + get_pp_gen_latency(pre[0][-1],tp_dev,cost_config)# pre latency
            min_computing_latency,min_profile = get_computing_latency(tp_dev,k-i)
            all_latency = max(pre[-1], (k-i) * tp_latency + pp_latency + min_profile[-2] * cost_config.s +min_profile[-1] * cost_config.n)
            if all_latency < min_latency:
                min_latency = all_latency
                pre[0].append(tp_dev)
                best_dev = (pre[0],i,min_profile,pre[-2] + tp_dev.cost,min_latency)
    #memory[f'{i}_{j}_{tp_dev.index}'] = all_latency
    return best_dev
