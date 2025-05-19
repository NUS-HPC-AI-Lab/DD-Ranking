import os
import argparse
import torch
from ddranking.metrics import SoftLabelEvaluator
from ddranking.config import Config

def main(args):
    root = args.root
    method_name = args.method_name
    dataset = args.dataset
    ipc = args.ipc

    print(f"Evaluating {method_name} on {dataset} with ipc{ipc}")
    syn_image_dir = os.path.join(root, f"DD-Ranking/baselines/{method_name}/{dataset}/IPC{ipc}/")
    config = Config.from_file(f"./{method_name}/{dataset}/IPC{ipc}.yaml")
    soft_obj = SoftLabelEvaluator(config)
    soft_obj.compute_metrics(image_path=syn_image_dir, syn_lr=args.syn_lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/wangkai/")
    parser.add_argument("--method_name", type=str, default="SRe2L")
    parser.add_argument("--dataset", type=str, default="ImageNet1K")
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--syn_lr", type=float, default=0.001)
    args = parser.parse_args()
    main(args)