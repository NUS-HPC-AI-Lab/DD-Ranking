import os
import time
import torch
import numpy as np
import random
import pandas as pd


def set_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000) % 1000000
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_results(results, save_path):
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def setup_dist(args):
    # local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # torch.cuda.set_device(local_rank)  # This ensures each process uses the correct GPU
    # torch.distributed.init_process_group(backend="nccl" if device == "cuda" else "gloo")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    torch.distributed.init_process_group(
        backend=args.dist_backend, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def logging(message):
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(message)


def broadcast_string(string_data, device, src=0):
    rank = torch.distributed.get_rank()
    
    if rank == src:
        encoded_string = string_data.encode('utf-8')
        string_len = torch.tensor([len(encoded_string)], dtype=torch.long, device=device)
    else:
        string_len = torch.tensor([0], dtype=torch.long, device=device)
    
    torch.distributed.broadcast(string_len, src=src)
    
    if rank == src:
        string_tensor = torch.tensor(list(encoded_string), dtype=torch.uint8, device=device)
    else:
        string_tensor = torch.zeros(string_len.item(), dtype=torch.uint8, device=device)
    
    torch.distributed.broadcast(string_tensor, src=src)
    received_string = ''.join([chr(byte) for byte in string_tensor])
    return received_string

def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True

def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    torch.distributed.barrier()
    torch.distributed.all_reduce(t)
    return t