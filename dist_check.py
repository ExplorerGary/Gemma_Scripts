import torch.distributed as dist

def dist_check():
    if dist.is_available():
        print(f"Distributed available: ✅")
        if dist.is_initialized():
            print(f"Distributed initialized: ✅ (rank={dist.get_rank()})")
        else:
            print("Distributed available, but not initialized ❌")
    else:
        print("Distributed not available ❌")

