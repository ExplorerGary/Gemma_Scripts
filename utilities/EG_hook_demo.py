import torch
import torch.distributed as dist
from torch.distributed import GradBucket
from torch.futures import Future

# 可替换为你自己的压缩函数
def EG_compress(input_data: torch.Tensor) -> torch.Tensor:
    # 暂时不压缩，仅返回原数据（placeholder）
    return input_data

def EG_decompress(compressed_data: torch.Tensor) -> torch.Tensor:
    # 解压函数（目前原样返回）
    return compressed_data

def EG_Hook(
    state,  # DDP hook 要求的 state 参数，暂时不使用可设为 None
    bucket: GradBucket
) -> Future:
    """
    A custom DDP communication hook that simulates compression -> all_reduce -> decompression.
    """

    # 1. 取出梯度张量（flattened tensor）
    input_tensor = bucket.buffer()

    # 2. 压缩
    compressed_tensor = EG_compress(input_tensor)

    # 3. 通信（异步 all_reduce），返回一个 future
    fut = dist.all_reduce(compressed_tensor, op=dist.ReduceOp.SUM, async_op=True).get_future()

    # 4. 在通信完成后解压，并返回还原后的梯度
    def decompress_and_return(fut: Future):
        reduced_tensor = fut.value()  # 得到 all-reduce 后的结果
        decompressed = EG_decompress(reduced_tensor)
        decompressed /= dist.get_world_size()  # 取平均
        return decompressed

    # 5. 返回 future 链
    return fut.then(decompress_and_return)
