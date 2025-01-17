'''
Generate prototype using the diffusers pipeline
Author: Su Duo & Houjunjie
Date: 2023.9.21
'''

from diffusers import StableDiffusionGenLatentsPipeline
import diffusers

import torch
import torchvision  
from torchvision import transforms
from torch.utils.data import DataLoader

import argparse
import json
import numpy as np
import math
import os
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from dataset_utils import *
import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=10, type=int, 
                        help='batch size')
    parser.add_argument('--data_dir', default='/home-ext/tbw/suduo/data/imagenet', type=str, 
                        help='root dir')
    parser.add_argument('--dataset', default='imagenet', type=str, 
                        help='data prepare to distillate:imagenet/tiny-imagenet')
    parser.add_argument('--diffusion_checkpoints_path', default="/home-ext/tbw/suduo/D3M/stablediffusion/checkpoints/stable-diffusion-v1-5", type=str, 
                        help='path to stable diffusion model from pretrained')
    parser.add_argument('--ipc', default=1, type=int, 
                        help='image per class')
    parser.add_argument('--km_expand', default=10, type=int, 
                        help='expand ration for minibatch k-means model')
    parser.add_argument('--label_file_path', default='/home-ext/tbw/suduo/data/imagenet_classes.txt', type=str, 
                        help='root dir')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='number of workers')
    parser.add_argument('--save_prototype_path', default='/home-ext/tbw/suduo/D3M/prototypes', type=str, 
                        help='where to save the generated prototype json files')
    parser.add_argument('--size', default=512, type=int, 
                        help='init resolution (resize)')
    args = parser.parse_args()
    return args


def initialize_km_models(label_list, args):
    km_models = {}
    for prompt in label_list:
        model_name = f"MiniBatchKMeans_{prompt}"
        model = MiniBatchKMeans(n_clusters=args.ipc, random_state=0, batch_size=(
            args.km_expand * args.ipc), n_init="auto")
        km_models[model_name] = model
    return km_models


def prototype_kmeans(pipe, data_loader, label_list, km_models, args):
    latents = {}
    for label in label_list:
        latents[label] = []
    
    for images, labels in tqdm(data_loader, total=len(data_loader), position=0):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        prompts = []
        for label in labels:
            prompt = label_list[label.item()]
            prompts.append(prompt)

        init_latents, _ = pipe(prompt=prompts, image=images, strength=0.7, guidance_scale=8)

        for latent, prompt in zip(init_latents, prompts):
            latent = latent.view(1, -1).cpu().numpy()
            latents[prompt].append(latent)
            if len(latents[prompt]) == args.km_expand * args.ipc:
                km_models[f"MiniBatchKMeans_{prompt}"].partial_fit(np.vstack(latents[prompt]))
                latents[prompt] = []  # save the memory, avoid repeated computation
    if len(latents[prompt]) >= args.ipc:
        km_models[f"MiniBatchKMeans_{prompt}"].partial_fit(np.vstack(latents[prompt]))
    return km_models


def gen_prototype(label_list, km_models):
    prototype = {}
    for prompt in label_list:
        model_name = f"MiniBatchKMeans_{prompt}"
        model = km_models[model_name]
        cluster_centers = model.cluster_centers_
        N = int(math.sqrt(cluster_centers.shape[1] / 4))
        num_clusters = cluster_centers.shape[0]
        reshaped_centers = []
        for i in range(num_clusters):
            reshaped_center = cluster_centers[i].reshape(4, N, N)
            reshaped_centers.append(reshaped_center.tolist())
        prototype[prompt] = reshaped_centers
    return prototype


def save_prototype(prototype, args):
    os.makedirs(args.save_prototype_path, exist_ok=True)
    json_file = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-kmexpand{args.km_expand}.json')
    with open(json_file, 'w') as f:
        json.dump(prototype, f)
    print(f"prototype json file saved at: {args.save_prototype_path}")


def main():
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1.obtain label-prompt list
    label_list = gen_label_list(args)

    # 2.obtain training data
    trainloader, _ = load_dataset(args)

    # 3.define the diffusers pipeline
    pipe = StableDiffusionGenLatentsPipeline.from_pretrained(args.diffusion_checkpoints_path, torch_dtype=torch.float16)
    pipe = pipe.to(args.device)

    # 4.initialize & run partial k-means model each class
    km_models = initialize_km_models(label_list, args)
    fitted_km = prototype_kmeans(pipe=pipe, data_loader=trainloader, label_list=label_list, km_models=km_models, args=args)

    # 5.generate prototypes and save them as json file
    prototype = gen_prototype(label_list, fitted_km)
    save_prototype(prototype, args)
    

if __name__ == "__main__" : 
    main()
