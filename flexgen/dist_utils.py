import torch
import torch.distributed as dist
from parallel_utils.parallel_state import initialize_model_parallel
from datetime import datetime, timedelta
import os
_COMM_DEVICE = None
_PIPELINE_PARALLEL_PRED_GROUP = None
_PIPELINE_PARALLEL_SUCC_GROUP = None


def initialize_distributed(head_ip, port, world_size, rank, local_rank, comm_device, tp, pp, tensor_ranks=None, pipeline_ranks=None):
    print(f'Initializing distributed environment at {head_ip}:{port}, '
          f'world_size={world_size}, rank={rank}, local_rank={local_rank}.')

    # Initialize distributed environment
    torch.cuda.set_device(local_rank)
    distributed_init_method = f'tcp://{head_ip}:{port}'
    global _COMM_DEVICE
    _COMM_DEVICE = comm_device
    if comm_device == 'cpu':
        backend = 'gloo'
    elif comm_device == 'gpu':
        backend = 'nccl'
    else:
        raise ValueError(f'Unknown comm_device: {comm_device}')
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    dist.init_process_group(backend=backend,
                            init_method=distributed_init_method,
                            world_size=world_size,
                            rank=rank,
                            timeout=timedelta(seconds=300))
    # dist.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(tp,pp,tensor_ranks, pipeline_ranks)
    # initialize_model_parallel(tp,pp)
    print("Finished initializing distributed environment")

def get_pipeline_parallel_pred_group():
    return _PIPELINE_PARALLEL_PRED_GROUP

def get_pipeline_parallel_succ_group():
    return _PIPELINE_PARALLEL_SUCC_GROUP

def get_comm_device():
    return _COMM_DEVICE

def suppress_output(rank):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', True)
        if force:
            builtin_print("rank #%d:" % rank, *args, **kwargs, flush=True)
        elif rank == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
