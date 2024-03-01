import argparse
from itertools import count
import os
import pickle
import traceback
from typing import Union, List, Optional
import ast
import numpy as np
import torch
import time
from random import sample
from math import ceil
import torch.distributed as dist
from transformers import AutoTokenizer

from flexgen.compression import CompressionConfig
from flexgen.dist_utils import initialize_distributed
from flexgen.flex_opt import (Policy, InputEmbed, OutputEmbed, SelfAttention,
                              MLP, TransformerLayer, OptLM, get_filename,
                              add_parser_arguments, get_test_inputs,move_weight,
                              DUMMY_WEIGHT)
from flexgen.opt_config import get_opt_config
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, TorchTensor)
from flexgen.timer import timers
from flexgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, array_4d, str2bool, project_decode_latency,set_value, get_value)
from parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank,
    get_pipeline_model_parallel_world_size, get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_prev_ranks,
    get_pipeline_model_parallel_next_ranks,get_tensor_model_parallel_group,get_pipeline_model_parallel_groups)
from util.TCPutil import TcpServer
import struct
#from alpa_serve.trace import Trace, TraceReplay, report_group_stats
from scipy.stats import entropy
import numpy as np
import time
import queue
import threading
# from alpa_serve.simulator.workload import Workload, GammaProcess, UniformMMPP
import func_timeout
from multiprocessing import Process,Event
from experimental.cost_model import solve_lp, CostModelConfig
#os.environ["NCCL_DEBUG"] = "TRACE"


global_model = None
global_args = None
global_tokenizer = None
global_clientThread = None
global_timeout = 1000000

class DistOptLM(OptLM):
    def __init__(self, config, env, path, policy, pipeline_rank,
                 num_pipeline_stages, comm_device, num_inner_iterations=None,
                 async_comm=False, local: bool = False,
                 local_path: str = None):
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = self.policy.num_gpu_batches
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_stages = num_pipeline_stages
        self.num_inner_iterations = num_inner_iterations if num_inner_iterations is not None else num_pipeline_stages
        self.async_comm = async_comm
        self.local = local
        self.local_path = local_path
        self.error = False
        if comm_device == "cpu":
            self.comm_device = self.env.cpu
        elif comm_device == "gpu":
            self.comm_device = self.env.gpu
        else:
            raise ValueError(f"Invalid comm_device: {comm_device}")

        layers = []
        if pipeline_rank == 0:
            layers.append(InputEmbed(self.config, self.env, self.policy))
        pipeline_stage_sizes = [config.num_hidden_layers // num_pipeline_stages
                                + int(i < config.num_hidden_layers % num_pipeline_stages)
                                for i in range(num_pipeline_stages)]
        layer_start_ids = [0]
        for stage_size in pipeline_stage_sizes:
            layer_start_ids.append(layer_start_ids[-1] + stage_size)
        for i in range(layer_start_ids[pipeline_rank], layer_start_ids[pipeline_rank + 1]):
            if self.policy.sep_layer:
                layers.append(SelfAttention(self.config, self.env, self.policy, i))
                layers.append(MLP(self.config, self.env, self.policy, i))
            else:
                layers.append(TransformerLayer(self.config, self.env, self.policy, i))
        if pipeline_rank == num_pipeline_stages - 1:
            layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        self.task = None
        self.init_all_weights()       

    def load_weight(self, b, t, i, j, k):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            t += 1
        if t == self.num_inner_iterations:
            t = 0
            i += 1
        if i == self.execute_gen_len:
            i = 0
            b += 1
        if b == self.num_pipeline_batches // self.num_inner_iterations:
            return

        # Load from weight_home to weight_read_buf
        with torch.cuda.stream(self.load_weight_stream):
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def init_cache(self, t, j, k):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[t][j][k])

    def load_cache(self, t, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            t += 1
        if t == self.num_inner_iterations:
            t = 0
            i += 1
        if i == self.execute_gen_len:
            return

        # Load from cache_home to cache_read_buf
        with torch.cuda.stream(self.load_cache_stream):
            self.layers[j].load_cache(self.cache_home[t][j][k], self.cache_read_buf[t][j][k], i)

    def store_cache(self, t, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            t -= 1
        if t == -1:
            t = self.num_inner_iterations - 1
            i -= 1
        if i == -1:
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        with torch.cuda.stream(self.store_cache_stream):
            self.layers[j].store_cache(self.cache_home[t][j][k], self.cache_write_buf[t][j][k], i)

    def delete_cache(self, t, j, k):
        v = self.cache_home[t][j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, b, t, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            t += 1
        if t == self.num_inner_iterations:
            t = 0
            i += 1
        if i == self.execute_gen_len:
            i = 0
            b += 1
        if b == self.num_pipeline_batches // self.num_inner_iterations:
            return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        if j > 0: # load from the last layer
            val = self.hidden[t][i][j-1][k].pop().move(dst)
            self.hidden[t][i][j][k].store(val)
            return
        if self.num_pipeline_stages > 1 and not (i == 0 and self.pipeline_rank == 0):
            # Already received the input from previous hidden states
            self.hidden[t][i][j][k].val = self.hidden[t][i][j][k].val.move(dst)
            return
        gpu_batch_size = self.policy.gpu_batch_size
        num_gpu_batches = self.num_gpu_batches
        num_inner_iterations = self.num_inner_iterations
        left = ((b * num_inner_iterations + t) * num_gpu_batches + k) * gpu_batch_size
        right = left + gpu_batch_size
        if i == 0:  # load from the input ids # ([24, 514])
            val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int64)
            val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
        else:  # load from the last generated token
            pos = self.task.prompt_len + i
            val = dst.allocate((gpu_batch_size, 1), np.int64)
            val.load_from_np(self.output_ids[left:right, pos-1:pos])
        self.hidden[t][i][j][k].store(val)

    def store_hidden(self, b, t, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            t -= 1
        if t == -1:
            t = self.num_inner_iterations - 1
            i -= 1
        if i == -1:
            i = self.execute_gen_len - 1
            b -= 1
        if b == -1:
            return

        # Store to hidden states buffers
        if j != self.num_layers - 1 or self.pipeline_rank != self.num_pipeline_stages - 1 or i != self.execute_gen_len - 1:
            # Move to home
            x = self.hidden[t][i][j][k]
            if x.val:
                x.val = x.val.move(self.act_home)

        if j == self.num_layers - 1 and self.pipeline_rank == self.num_pipeline_stages - 1:
            # store to output
            if i == self.execute_gen_len - 1:  # last token
                hidden_val = self.hidden[t][i][j][k].pop()
            else:
                hidden_val = self.hidden[t][i][j][k].val

            ids = hidden_val.data.detach().cpu().numpy()
            # print("ids", hidden_val.data.shape)# [20,1] args.gpu_batch_size=20, args.num_gpu_batches=1
            gpu_batch_size = self.policy.gpu_batch_size
            num_gpu_batches = self.num_gpu_batches
            num_inner_iterations = self.num_inner_iterations
            left = ((b * num_inner_iterations + t) * num_gpu_batches + k) * gpu_batch_size
            right = left + gpu_batch_size
            pos = self.task.prompt_len + i
            self.output_ids[left:right, pos:pos+1] = ids

    def send_hidden(self, t, i, j, k, tag=0, async_=False):
        # Suppose we need to send tensors on GPUs
        x = self.hidden[t][i][j][k]
        val = x.pop().move(self.comm_device)
        ranks,groups = get_pipeline_model_parallel_next_ranks()
        if async_:
            for idx in range(len(ranks)):
                future = dist.isend(val.data, ranks[idx], group=groups[idx], tag=tag)
            return future
        else:
            for idx in range(len(ranks)):
                dist.send(val.data, ranks[idx], group=groups[idx],tag=tag)

    def recv_hidden(self, t, i, j, k, tag=0, async_=False):
        # sender_rank = (self.pipeline_rank - 1) % self.num_pipeline_stages
        val_holder = self.hidden[t][i][j][k]
        seq_len = self.task.prompt_len if i == 0 else 1
        shape, dtype = self.layers[j].input_act_shape_and_dtype(
            self.policy.gpu_batch_size, seq_len)
        if val_holder.val is None:
            val_holder.val = self.comm_device.allocate(shape, dtype)
        else:
            val_holder.val = val_holder.val.move(self.comm_device)
        def move_value_callback():
            val_holder.val = val_holder.val.move(self.act_home)
        ranks,groups = get_pipeline_model_parallel_prev_ranks()
        if async_:
            for idx in range(len(ranks)):
                future = dist.irecv(val_holder.val.data, ranks[idx], group=groups[idx],tag=tag)
            return future, move_value_callback
        else:
            for idx in range(len(ranks)):
                dist.recv(val_holder.val.data, ranks[idx], group=groups[idx],tag=tag)
            move_value_callback()

    def compute_layer(self, t, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.layers[j].forward(self.hidden[t][i][j][k], self.cache_read_buf[t][j][k],
            self.weight_read_buf[j], self.attention_mask[t][k],
            self.cache_write_buf[t][j][k], i, k)

    def update_attention_mask(self, b, t, i, k):
        if i > 0:
            mask = self.attention_mask[t][k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        num_gpu_batches = self.num_gpu_batches
        num_inner_iterations = self.num_inner_iterations
        left = ((b * num_inner_iterations + t) * num_gpu_batches + k) * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[t][k].val = val

    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
        )
        assert stop is None, "Not implemented."
        num_pipeline_stages = self.num_pipeline_stages
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        num_prompts = len(task.inputs)
        num_inner_iterations = self.num_inner_iterations

        assert num_prompts % (gpu_batch_size * num_gpu_batches) == 0
        num_pipeline_batches = num_prompts // (gpu_batch_size * num_gpu_batches)
        self.num_pipeline_batches = num_pipeline_batches
        assert num_pipeline_batches % num_inner_iterations == 0
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len
        # Output token ids
        self.output_ids = np.ones((num_prompts, prompt_len + gen_len), dtype=np.int64)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch, t-th stage.
        # cache[t][j][k]
        self.cache_home = array_3d(num_inner_iterations, num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_3d(num_inner_iterations, num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_3d(num_inner_iterations, num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # hidden[t][i][j][k]
        self.hidden = array_4d(num_inner_iterations, gen_len, num_layers, num_gpu_batches, ValueHolder)
        # attention_mask[t][k]
        self.attention_mask = array_2d(num_inner_iterations, num_gpu_batches, ValueHolder)
        # Init cache
        self.set_task(task)
        for t in range(num_inner_iterations):
            for j in range(num_layers):
                for k in range(num_gpu_batches):
                    self.init_cache(t, j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        dist.barrier()
        # Generate
        if not overlap:
            # No overlap, easy to understand, suitable for debugging
            self.generation_loop_normal()
        else:
            # Overlap I/O and compute
            if self.policy.num_gpu_batches == 1:
                self.generation_loop_overlap_one_batch()
            else:
                self.generation_loop_overlap_multi_batch()
        # Delete cache
        for t in range(num_inner_iterations):
            for j in range(num_layers):
                for k in range(num_gpu_batches):
                    self.delete_cache(t, j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()

        return self.output_ids

    def send_recv_hidden(self, sending_job, receiving_job):
        st, si = sending_job if sending_job is not None else (None, None)
        rt, ri = receiving_job if receiving_job is not None else (None, None)
        sending = sending_job is not None and not (si == self.execute_gen_len - 1 and self.pipeline_rank == self.num_pipeline_stages - 1)
        receiving = receiving_job is not None and not (ri == 0 and self.pipeline_rank == 0)

        def _send():
            sending_futures = []
            if not sending:
                return sending_futures
            for k in range(self.num_gpu_batches):
                sending_future = self.send_hidden(st, si, self.num_layers - 1, k, self.sending_tag, async_=self.async_comm)
                sending_futures.append(sending_future)
                self.sending_tag += 1
            return sending_futures

        def _recv():
            receiving_futures = []
            if not receiving:
                return receiving_futures
            for k in range(self.num_gpu_batches):
                receiving_future = self.recv_hidden(rt, ri, 0, k, self.receiving_tag, async_=self.async_comm)
                receiving_futures.append(receiving_future)
                self.receiving_tag += 1
            return receiving_futures
        timers('send_recv_hidden').start()
        # Use special order below to avoid deadlock
        if self.pipeline_rank == 0:
            # Receive first and then send
            receiving_futures = _recv()
            sending_futures = _send()
        else:
            # Send first and then receive
            sending_futures = _send()
            receiving_futures = _recv()
        if self.async_comm:
            for sending_future in sending_futures:
                sending_future.wait()
            for receiving_future, callback in receiving_futures:
                receiving_future.wait()
                callback()
        timers('send_recv_hidden').stop()

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()
    
    def barrier(self):
        dist.barrier()

    def generation_loop_normal(self):
        self.sending_tag = 0
        self.receiving_tag = 0
        last_sending_job = None
        for b in range(self.num_pipeline_batches // self.num_inner_iterations):
            for i in range(self.execute_gen_len):
                for t in range(self.num_inner_iterations):# 这个就是转够这么多个设备填满流水线，再从最后一个流水线stage得到结果进行下一个generation。
                    timer_name = "generate-prompt" if i == 0 else "generate"
                    timers(timer_name).start()
                    for k in range(self.num_gpu_batches):
                        self.update_attention_mask(b, t, i, k)
                    if self.num_pipeline_stages > 1:
                        self.send_recv_hidden(last_sending_job, (t, i))
                    for k in range(self.num_gpu_batches):
                        for j in range(self.num_layers):
                            self.load_weight(b, t, i, j, k)
                            self.load_cache(t, i, j, k)
                            self.load_hidden(b, t, i, j, k)
                            self.sync() # 这个才是费时间的
                            self.compute_layer(t, i, j, k)
                            self.sync()
                            self.store_hidden(b, t, i, j, k)
                            self.store_cache(t, i, j, k)
                            self.sync()
                            
                    last_sending_job = (t, i)
                    timers(timer_name).stop()

        if self.num_pipeline_stages > 1:
            self.send_recv_hidden(last_sending_job, None)           
            self.barrier()

    def generation_loop_overlap_one_batch(self):
        assert self.num_gpu_batches == 1
        # Prologue
        self.load_weight(0, 0, 0, 0, 0)
        self.sync()
        self.sending_tag = 0
        self.receiving_tag = 0
        last_sending_job = None
        # Generate
        for b in range(self.num_pipeline_batches // self.num_inner_iterations):
            for i in range(self.execute_gen_len):
                for t in range(self.num_inner_iterations):
                    timer_name = "generate-prompt" if i == 0 else "generate"
                    timers(timer_name).start()
                    self.update_attention_mask(b, t, i, 0)
                    if self.num_pipeline_stages > 1:
                        self.send_recv_hidden(last_sending_job, (t, i))
                    for j in range(self.num_layers):
                        timers("load").start()
                        self.load_weight(b, t, i, j+1, 0)
                        self.load_cache(t, i, j+1, 0)
                        self.load_hidden(b, t, i, j, 0)
                        timers("load").stop()
                        timers("compute_layer").start()
                        self.compute_layer(t, i, j, 0)
                        timers("compute_layer").stop()
                        timers("store").start()
                        self.store_cache(t, i, j-1, 0)
                        self.store_hidden(b, t, i, j, 0)
                        self.sync()
                        timers("store").stop()
                    last_sending_job = (t, i)
                    timers(timer_name).stop()
        if self.num_pipeline_stages > 1:
            self.send_recv_hidden(last_sending_job, None)
            self.barrier()
        
    def generation_loop_overlap_multi_batch(self):
        self.sending_tag = 0
        self.receiving_tag = 0
        last_sending_job = None

        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, 0, 0, k)

        for b in range(self.num_pipeline_batches // self.num_inner_iterations):
            for i in range(self.execute_gen_len):
                for t in range(self.num_inner_iterations):
                    timer_name = "generate-prompt" if i == 0 else "generate"
                    timers(timer_name).start()
                    for k in range(self.num_gpu_batches):
                        self.update_attention_mask(b, t, i, k)
                    if self.num_pipeline_stages > 1:
                        self.send_recv_hidden(last_sending_job, (t, i))
                    for j in range(self.num_layers):
                        for k in range(self.num_gpu_batches):
                            self.load_weight(b, t, i, j + 1, k)
                            self.load_cache(t, i, j, k + 1)
                            self.load_hidden(b, t, i, j, k)
                            self.compute_layer(t, i, j, k)
                            self.store_cache(t, i, j, k - 1)
                            self.store_hidden(b, t, i, j, k)
                            self.sync()

                    last_sending_job = (t, i )

                    timers(timer_name).stop()

        if self.num_pipeline_stages > 1:
            self.send_recv_hidden(last_sending_job, None)
            self.barrier()

def comm_test(comm_device):
    # A small all_reduce for warmup.
    a = torch.ones(1).to(comm_device)
    dist.all_reduce(a,group=get_tensor_model_parallel_group())
    # assert a.item() == args.world_size

def run_flexgen_dist(args):
    t_name = args.model.replace("175b", "66b")
    tokenizer = AutoTokenizer.from_pretrained(t_name, padding_side="left")
    num_inner_iterations = args.num_inner_iterations if args.num_inner_iterations is not None else args.pp
    num_prompts = args.num_prompts # 最少都要大于等于流水线的stage数目
    args.num_gpu_batches = 1
    args.gpu_batch_size  = num_prompts // num_inner_iterations // args.num_gpu_batches
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len
    # Task and policy
    warmup_inputs = get_test_inputs(6, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
    gpu = TorchDevice(f"cuda:{args.local_rank}")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir, None, args.local_rank)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
    TorchTensor.name_count = count(start=args.rank, step=args.world_size)
    comm_test(gpu.dev if args.comm_device == "gpu" else cpu.dev)
    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False),
                    args.by_layer)
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"
    set_value("global_policy",policy)
    opt_config = get_opt_config(args.model)
    total_layer_num = opt_config.num_hidden_layers
    profile_sets = None
    if args.num_hidden_layers is not None:
        opt_config.num_hidden_layers = args.num_hidden_layers
    if args.pp > 1 or args.tp > 1:
        opt_config.is_distributed = True
    if get_pipeline_model_parallel_rank() == get_pipeline_model_parallel_world_size() - 1 \
        and get_tensor_model_parallel_rank() == get_tensor_model_parallel_world_size() - 1:                
        print(args)
        print(opt_config)
    model = DistOptLM(opt_config, env, args.path, policy, get_pipeline_model_parallel_rank(),
                      get_pipeline_model_parallel_world_size(), args.comm_device, num_inner_iterations=num_inner_iterations,
                      async_comm=args.async_comm, local = args.local, local_path = args.model)
    global global_model
    global_model = model
    global global_args
    global_args = args
    global global_tokenizer
    global_tokenizer = tokenizer
    try:
        print("warmup - generate")
        output_ids = model.generate(
            warmup_inputs, max_new_tokens=5, verbose=args.verbose)     
        if model.error:
            dist.destroy_process_group()
            #setting = global_clientThread.setting
            #print("error",setting)
            # initialize_distributed(setting.head_ip, setting.port, setting.world_size,
            #                   setting.rank, args.local_rank, args.comm_device,setting.tp, setting.pp)
            # model.move_layer(setting.start_id,setting.pipeline_stage_size,get_pipeline_model_parallel_rank(),setting.pp)
            # policy.update_percent(setting.wg*100,setting.wc*100,setting.cg*100,setting.cc*100,100,0)
            # model.move_all_weight()
        if args.profile is None or not args.profile:
            costs = []
            args.percent[0] = args.percent[0]/100
            args.percent[1] = args.percent[1]/100
            for k in [args.gen_len]: # [32,64,128,246,512]:
                for timer_name in ["all","generate-prompt", "generate","send_recv_hidden","all_reduce", "all_gather","compute_layer","store","load"]:
                    timers(timer_name).reset()
                timers("all").start()
                prompt_len = args.prompt_len
                args.gen_len = k 
                layer_num = int(model.num_layers/2)
                if not args.compress_weight:
                    config = CostModelConfig()
                    config.l = layer_num #opt_config.num_hidden_layers
                    config.h1 = opt_config.hidden_size
                    config.h2 = opt_config.ffn_embed_dim
                    config.nh = opt_config.n_head
                    config.s = prompt_len
                    config.n = args.gen_len
                    _,new_p,_,_ = solve_lp(config,args.gpu_batch_size*1,args.gpu_batch_size)
                    policy.update_percent(new_p.w_gpu_percent,new_p.w_cpu_percent,new_p.cache_gpu_percent,
                    new_p.cache_cpu_percent,new_p.act_gpu_percent,new_p.act_cpu_percent)
                    print(policy)
                inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
                output_ids = model.generate(
                    inputs, max_new_tokens=args.gen_len,
                    debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                # show_str = "Outputs:\n" + 70 * '-' + "\n"
                # for i in [0, len(outputs)-1]:
                #     show_str += f"{i}: {outputs[i]}\n"
                #     show_str += "-" * 70 + "\n"
                # print(show_str)
                timers("all").stop()
                prompt_cost = sum(timers("generate-prompt").costs)
                generate_cost = sum(timers("generate").costs)
                net_cost = sum(timers("send_recv_hidden").costs)
                compute_layer = sum(timers("compute_layer").costs)
                store = sum(timers("store").costs)
                load = sum(timers("load").costs)
                all_cost = sum(timers("all").costs)
                #print(k,prompt_cost,generate_cost,prompt_cost+generate_cost,all_cost,(args.gen_len*num_prompts)/all_cost)
                print(f"[{prompt_cost},{generate_cost},{net_cost},{compute_layer},{store},{load},{all_cost},{(args.gen_len*num_prompts)/all_cost}]")
                costs.append(all_cost)
            np.save(args.log_file,costs)
            # for k in [2,4,6,8,12,16,24]:
            #     for timer_name in ["all","generate-prompt", "generate","send_recv_hidden","all_reduce", "all_gather","compute_layer","store","load"]:
            #         timers(timer_name).reset()
            #     timers("all").start()
            #     prompt_len = 32
            #     num_prompts = k
            #     model.policy.gpu_batch_size  = num_prompts // num_inner_iterations // args.num_gpu_batches
            #     layer_num = int(model.num_layers/2)
            #     config = CostModelConfig()
            #     config.l = layer_num #opt_config.num_hidden_layers
            #     config.h1 = opt_config.hidden_size
            #     config.h2 = opt_config.ffn_embed_dim
            #     config.nh = opt_config.n_head
            #     config.s = prompt_len
            #     config.n = args.gen_len
            #     _,new_p,_,_ = solve_lp(config,args.gpu_batch_size*1,args.gpu_batch_size)
            #     policy.update_percent(new_p.w_gpu_percent,new_p.w_cpu_percent,new_p.cache_gpu_percent,
            #     new_p.cache_cpu_percent,new_p.act_gpu_percent,new_p.act_cpu_percent)
            #     inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
            #     output_ids = model.generate(
            #         inputs, max_new_tokens=args.gen_len,
            #         debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
            #     timers("all").stop()
            #     prompt_cost = sum(timers("generate-prompt").costs)
            #     generate_cost = sum(timers("generate").costs)
            #     all_cost = sum(timers("all").costs)
            #     print(k,prompt_cost,generate_cost,prompt_cost+generate_cost,all_cost,(args.gen_len*num_prompts)/all_cost)
            #     costs.append((args.gen_len*num_prompts)/all_cost)
            # name = args.model.split("/")[-1]
            # np.save(f'./experimental/result/fcosts_{name}_{args.tp}_{args.pp}_throughput',costs)

      
    finally:
        env.close_copy_threads()

    # if get_pipeline_model_parallel_rank() != get_pipeline_model_parallel_world_size() - 1:
    #     return

    # # Log output
    # prefill_latency = sum(prompt_costs)
    # prefill_throughput = num_prompts * prompt_len / prefill_latency
    # if cut_gen_len:  # project latency of cut_gen_len to gen_len
    #     costs = np.array(generate_costs).reshape(-1, cut_gen_len-1).sum(axis=0).tolist()
    #     decode_latency = project_decode_latency([None] + costs, prompt_len, gen_len)
    # else:
    #     decode_latency = sum(generate_costs)
    # decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    # num_generated_tokens = num_prompts * gen_len
    # total_latency = prefill_latency + decode_latency
    # total_throughput = num_generated_tokens / total_latency
    # _, gpu_peak_mem = gpu.mem_stats()
    # _, cpu_peak_mem = cpu.mem_stats()

    # if DUMMY_WEIGHT not in args.path:
    #     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #     show_str = "Outputs:\n" + 70 * '-' + "\n"
    #     for i in [0, len(outputs)-1]:
    #         show_str += f"{i}: {outputs[i]}\n"
    #         show_str += "-" * 70 + "\n"
    #     print(show_str)

    # gpu.print_stats()
    # cpu.print_stats()
    # projected = args.debug_mode or cut_gen_len

    # log_str = (f"model size: {opt_config.model_bytes()/GB:.3f} GB\t"
    #            f"cache size: {cache_size/GB:.3f} GB\t"
    #            f"hidden size (prefill): {hidden_size/GB:.3f} GB\n"
    #            f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\n"
    #            f"prefill latency: {prefill_latency:.2f} s\t"
    #            f"prefill throughput: {prefill_throughput:.2f} token/s\n"
    #            f"decode latency: {decode_latency:.2f} s\t"
    #            f"decode throughput: {decode_throughput:.2f} token/s\n"
    #            f"total latency: {total_latency:.2f} s\t"
    #            f"total throughput: {total_throughput:.2f} token/s\t"
    #            f"Avg latency: {total_latency / num_prompts:.2f} s\t")
    # print(log_str)

    # if not args.no_log:
    #     if args.log_file == "auto":
    #         basename = f"rank-{args.rank}-{get_filename(args)}"
    #         log_filename = basename + ".log"
    #     else:
    #         log_filename = args.log_file
    #     with open(log_filename, "a") as fout:
    #         fout.write(log_str + "\n")
        
def add_distributed_parser_arguments(parser):
    parser.add_argument('--head-ip', type=str, default=None, help='the IP address of the head node')
    parser.add_argument('--port', type=int, default=None, help='the port of the head node')
    parser.add_argument('--rank', metavar='I', type=int, default=None)
    parser.add_argument('--local-rank', metavar='I', type=int, default=None)
    parser.add_argument('--world-size', metavar='N', type=int, default=None)
    parser.add_argument('--use-mpi', action='store_true', default=False,
                        help="Get distributed info from MPI")
    parser.add_argument('--comm-device', type=str, default='gpu',
                        choices=['gpu', 'cpu'],
                        help='communication through gpu nvlink or cpu memory '
                             'and socket')
    parser.add_argument('--num-inner-iterations', metavar='I', type=int, default=None)
    parser.add_argument('--async-comm', action='store_true', default=False,
                        help="Use asynchronous communication")
    parser.add_argument("--pp", type=int, default=1,
        help="pipeline parallelism")
    parser.add_argument("--tp", type=int, default=1,
        help="tensor parallelism")
    parser.add_argument("--tensor-ranks", type=str, default=None,
        help="tensor parallelism ranks, eg.[[1,2],[3,4]]")
    parser.add_argument("--pipeline-ranks", type=str, default=None,
        help="pipeline parallelism ranks, eg.[[1,2],[3,4]]")
    parser.add_argument("--num-prompts", type=int, default=6,
        help="num of prompts")
    parser.add_argument("--memory-fraction", type=float, default=1.0,
        help="memory_fraction")


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    add_distributed_parser_arguments(parser)
    args = parser.parse_args()
    
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, args.local_rank)

    if args.head_ip is not None and args.port is not None:
        if args.use_mpi:
            args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
            args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
            args.local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
        tensor_ranks, pipeline_ranks = None, None
        if args.tensor_ranks is not None and args.pipeline_ranks is not None:
            tensor_ranks = ast.literal_eval(args.tensor_ranks)
            pipeline_ranks = ast.literal_eval(args.pipeline_ranks)
        timers('Init distributed').reset()
        timers('Init distributed').start()
        initialize_distributed(args.head_ip, args.port, args.world_size,
                               args.rank, args.local_rank, args.comm_device,args.tp, args.pp, tensor_ranks, pipeline_ranks)
        timers('Init distributed').stop() 
        # Init distributed latency: 1.07 s
        # Init weight latency: 0.11 s

        print(f"Init distributed latency: {sum(timers('Init distributed').costs):.2f} s\t")
    else:
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    assert len(args.percent) == 6

    # if get_tensor_model_parallel_rank() == 0
    #     address = '0.0.0.0'
    #     port = 9091
    #     tcp_server = TcpServer(address,port)
    #     req_queue = queue.Queue()
    #     request_threads = [
    #         threading.Thread(
    #             receive_request, args=(tcp_server, req_queue)
    #         ) for _ in range(num_request_threads)
    #     ]
    #     for t in request_threads:
    #         t.setDaemon(True)
    #         t.start()
    # heartbeat_threads = [
    #     threading.Thread(
    #         send_heartbeat, args=(tcp_server_address,tcp_server_port,rank)
    #     ) for _ in range(num_request_threads)
    # ]
    # for t in heartbeat_threads:
    #     t.setDaemon(True)
    #     t.start()

    try:
        run_flexgen_dist(args)
    except Exception as e:
        print(str(e))
        traceback.print_exc()
        raise e
    
