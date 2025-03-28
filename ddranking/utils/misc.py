import os
import time
import torch
import numpy as np
import random
import pandas as pd


def set_seed():
    seed = int(time.time() * 1000) % 1000000
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_results(results, save_path):
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)

def setup_dist(device):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)  # This ensures each process uses the correct GPU
    torch.distributed.init_process_group(backend="nccl" if device == "cuda" else "gloo")
    return local_rank

def logging(message):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
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