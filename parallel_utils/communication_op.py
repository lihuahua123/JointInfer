import torch

from .parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
)
from flexgen.timer import timers

def tensor_model_parallel_all_reduce(input_):
    """All-reduce the input tensor across model parallel group.

    NOTE: This operation is applied in-place on the input tensor.
    """
    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    # All-reduce.
    timers('all_reduce').start()
    torch.distributed.all_reduce(input_,
                                 group=get_tensor_model_parallel_group())
    timers('all_reduce').stop()
    return input_


def tensor_model_parallel_all_gather(input_, dim=-1):
    """All-gather the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # Allocate output tensor.
    # output_tensor = torch.empty((world_size, ) + input_size,
    #                             dtype=input_.dtype,
    #                             device=input_.device) # 这一步直接在第一维度多了world_size个向量
    #print(input_.shape,output_tensor.shape) 
    # torch.Size([20, 32, 12568]) torch.Size([4, 20, 32, 12568])
    # All-gather. 用的是nccl，比较快，但是得在GPU上 用空间换时间
    group_ = get_tensor_model_parallel_group()
    backen_ = torch.distributed.get_backend(group_)
    input_device = input_.device
    timers('all_gather').start()
    if  backen_ == 'gloo': # 能减少显存1个G，但是慢了20倍，因此还是用nccl比较好
        tensor_list = [torch.empty(input_size, dtype=input_.dtype,device='cpu') for _ in range(world_size)]
        torch.distributed.all_gather(tensor_list, input_.to('cpu'), group=group_) # works for gloo
        output_tensor = torch.cat(tensor_list, dim=0).cuda() 
        input_.to(input_device)
    elif backen_ == 'nccl':
        output_tensor = torch.empty((world_size, ) + input_size,
                                dtype=input_.dtype,
                                device=input_device)
        torch.distributed.all_gather_into_tensor(output_tensor, input_, group=group_)
        output_tensor = output_tensor.movedim(0, dim)
    else:
        raise ValueError(f'Unknown backend: {backen_}')
    
    # Reshape
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                          (world_size * input_size[dim], ) +
                                          input_size[dim + 1:])
    timers('all_gather').stop()
    return output_tensor
