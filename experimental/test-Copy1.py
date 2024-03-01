import torch
import time
import torch.distributed as dist
head_ip = '192.168.249.124'
port = 7773
distributed_init_method = f'tcp://{head_ip}:{port}'
backend = 'nccl'
world_size = 2
rank = 0
dist.init_process_group(backend=backend,
                            init_method=distributed_init_method,
                            world_size=world_size,
                            rank=rank)


