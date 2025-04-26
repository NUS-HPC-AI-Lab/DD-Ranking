import os
import torch
import argparse
from ddranking.metrics import HardLabelEvaluator
from ddranking.config import Config

def main(args):
    root = args.root
    method_name = args.method_name
    dataset = args.dataset
    ipc = args.ipc

    print(f"Evaluating {method_name} on {dataset} with ipc{ipc}")
    syn_image = torch.load(os.path.join(root, f"DD-Ranking/baselines/{method_name}/{dataset}/IPC{ipc}/images.pt"))
    hard_label = torch.load(os.path.join(root, f"DD-Ranking/baselines/{method_name}/{dataset}/IPC{ipc}/labels.pt"))
    config = Config.from_file(f"./{method_name}/{dataset}/IPC{ipc}.yaml")
    hard_obj = HardLabelEvaluator(config)

    hard_obj.compute_metrics(image_tensor=syn_image, image_path=None, hard_labels=hard_label, syn_lr=args.syn_lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/wangkai/")
    parser.add_argument("--method_name", type=str, default="MTT")
    parser.add_argument("--dataset", type=str, default="TinyImageNet")
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--syn_lr", type=float, default=0.01)
    args = parser.parse_args()
    main(args)