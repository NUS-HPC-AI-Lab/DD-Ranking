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