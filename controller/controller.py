
"""
# 心跳，发现有down掉的，另一方面完好的发送信号说自己卡住了，之后查找最新的资源，
# 有新资源：重新分配,尽量减少已有的
# 没有新资源：卸载
# 总之根据现有的资源，重新分配
# 发现同伴挂掉了，就应该暂停等待controller分配任务
"""
import sys
sys.path.append(r"/home/server/DistributedOffload")
from util.TCPutil import TcpServer,ServerThread,Package
import time
from flexgen.opt_config import get_opt_config
from experimental.my_cost_model import get_setting
model = '/home/server/OPT-125M'
opt_config = get_opt_config(model)
profile_setting = get_setting(model)
total_layer_num = opt_config.num_hidden_layers
address = '0.0.0.0'
port = 9090
max_world_size = 6
tcp_server = TcpServer(address,port)
tcp_thread = ServerThread(tcp_server, {})
tcp_thread.setDaemon(True)
tcp_thread.start()
def update(tcp_thread):
    num_pipeline_stages = len(tcp_thread.agent_dict)
    pipeline_stage_sizes = [total_layer_num // num_pipeline_stages
                                + int(i < total_layer_num % num_pipeline_stages)
                                for i in range(num_pipeline_stages)]
    layer_start_ids = [0]
    for stage_size in pipeline_stage_sizes:
        layer_start_ids.append(layer_start_ids[-1] + stage_size)
    head_address = tcp_thread.agent_dict[list(tcp_thread.agent_dict.keys())[0]].address
    ip = head_address[0]
    port = head_address[1]
    pp_size = len(tcp_thread.agent_dict)
    tp_size = 1
    keys = list(tcp_thread.agent_dict.keys())
    if len(keys) != num_pipeline_stages:
        return 
    for key_id in range(len(keys)):
        pstage_size = pipeline_stage_sizes[key_id]
        setting = profile_setting[tp_size][pstage_size][0]
        package = Package(ip,port,len(tcp_thread.agent_dict),key_id,tp_size,pp_size,layer_start_ids[key_id],pstage_size,
                setting[0],setting[1],setting[2],setting[3])
        tcp_server.send_package[key_id]=package

while len(tcp_thread.agent_dict) < max_world_size:
    # FIXME: naive way to wait for all agents to connect  
    time.sleep(0.5)
update(tcp_thread)
print("全收到！")
tcp_server.receive_all = True
while True:
    time.sleep(3)
    for key in list(tcp_thread.agent_dict.keys()):
        if time.time() - tcp_thread.received_heartbeat_time[key] > 30:
            print(f"{key} 超时")
            del tcp_thread.agent_dict[key]
            del tcp_thread.received_heartbeat_time[key]
    if len(tcp_thread.agent_dict) == 0:
        print("there is no agent")
        break
    update(tcp_thread)
