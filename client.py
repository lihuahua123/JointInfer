from alpa_serve.trace import Trace, TraceReplay, report_group_stats
from scipy.stats import entropy
import numpy as np
import time
import queue
import threading
from alpa_serve.simulator.workload import Workload, GammaProcess, UniformMMPP
from dist_flex_opt import handle_request
global submit_queue
submit_queue = queue.Queue()
def simulated_requests():
    def submit_worker_func(queue,id):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return
            handle_request(item[0][0])
            queue.task_done()
    def submit_request(*args):
        submit_queue.put_nowait(args)
    def synchronize():
        submit_queue.join()
    def close_submit_threads():
        global submit_queue
        for _ in range(len(submit_threads)):
            submit_queue.put_nowait(None)
        for t in submit_threads:
            t.join()
        submit_queue.join()
        submit_queue = None

    num_submit_threads = 4
    submit_threads = [
        threading.Thread(
            target=submit_worker_func, args=(submit_queue,0)
        ) for _ in range(num_submit_threads)
    ]
    for t in submit_threads:
        t.start()
    trace_name = "azure_v2"
    trace_dir = "/home/server/DistributedOffload/alpa_serve/azure_v2.pkl"
    trace = Trace(trace_name, trace_dir)
    n_model = 5
    models = [f"gpt{i}" for i in range(n_model)]
    train_start = "13.0.0"
    train_end = "13.23.60"
    replays = trace.replay(models,model_mapping_strategy="stripe",
                                arrival_distribution="gamma",
                                start_time=train_start,
                                end_time=train_end,
                                interval_seconds=5400)
    ws = []
    model_name = 'gpt0'
    slo_scale = 1
    single_latency = 1000
    num_models = 1
    slos = [slo_scale * single_latency] * num_models
    ws.append(replays[model_name].to_workload(slos[0]))
    workload = Workload.merge(*ws)

    prompts = ["Paris is the capital city of"]
    prompt_len = max([len(i) for i in prompts]) 
    input_ids = global_tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    input_ = (input_ids[0],) * 24 # 一批请求24
    args = [(input_, workload.requests[i].model_name,
                    workload.requests[i].slo, float(workload.arrivals[i]),
                    i) for i in range(len((workload)))]
    for i in range(3):
        start =  time.time() + float(workload.arrivals[i]) - 1123200
        while time.time() < start:
            pass
        submit_request(args[i]) # submit 
        synchronize()
        end = time.time()
        e2e_latency = end - start
        print(e2e_latency)
    synchronize()
    print("ok")
    close_submit_threads()

simulated_requests()