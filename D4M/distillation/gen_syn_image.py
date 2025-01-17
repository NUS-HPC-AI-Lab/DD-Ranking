from diffusers import StableDiffusionLatents2ImgPipeline

import torch
import torchvision  
from torchvision import transforms

import argparse
from dataset_utils import *
import json
import os
from tqdm import tqdm

from dataset_utils import *

import ipdb



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=10, type=int, 
                        help='batch size')
    parser.add_argument('--diffusion_checkpoints_path', default="/home-ext/tbw/suduo/D3M/stablediffusion/checkpoints/stable-diffusion-v1-5", type=str, 
                        help='path to stable diffusion model from pretrained')
    parser.add_argument('--dataset', default='cifar10', type=str, 
                        help='data prepare to distillate')
    parser.add_argument('--guidance_scale', '-g', default=8, type=float, 
                        help='diffusers guidance_scale')
    parser.add_argument('--ipc', default=1, type=int, 
                        help='image per class')
    parser.add_argument('--km_expand', default=10, type=int, 
                        help='expand ration for minibatch k-means model')
    parser.add_argument('--label_file_path', default='/home-ext/tbw/suduo/data/imagenet_classes.txt', type=str, 
                        help='root dir')
    parser.add_argument('--prototype_path', default='/home-ext/tbw/suduo/D3M/prototypes/imagenet-ipc1-kmexpand1.json', type=str, 
                        help='prototype path')
    parser.add_argument('--save_init_image_path', default='/home-ext/tbw/suduo/data/init_data/random', type=str, 
                        help='where to save the generated prototype json files')
    parser.add_argument('--strength', '-s', default=0.75, type=float, 
                        help='diffusers strength')
    args = parser.parse_args()
    return args


def load_prototype(args):
    prototype_file_path = args.prototype_path
    with open(prototype_file_path, 'r') as f:
        prototype = json.load(f)

    for prompt, data in prototype.items():
        prototype[prompt] = torch.tensor(data, dtype=torch.float16).to(args.device)
    print("prototype loaded.")
    return prototype


def gen_syn_images(pipe, prototypes, label_list, args):
    for prompt, pros in tqdm(prototypes.items(), total=len(prototypes), position=0):

        assert  args.ipc % pros.size(0) == 0
        
        for j in range(int(args.ipc/(pros.size(0)))):
            for i in range(pros.size(0)):
                sub_pro = pros[i:i+1]
                sub_pro_random = torch.randn((1, 4, 64, 64), device='cuda',dtype=torch.half)
                images = pipe(prompt=prompt, latents=sub_pro, negative_prompt='cartoon, anime, painting', is_init=True, strength=args.strength, guidance_scale=args.guidance_scale).images
                index = label_list.index(prompt)
                save_path = os.path.join(args.save_init_image_path, "{}_ipc{}_{}_s{}_g{}_kmexpand{}".format(args.dataset, int(pros.size(0)), args.ipc, args.strength, args.guidance_scale, args.km_expand))
                os.makedirs(os.path.join(save_path, "{}/".format(index)), exist_ok=True)
                images[0].resize((224, 224)).save(os.path.join(save_path, "{}/{}-image{}{}.png".format(index, index, i, j)))


def main():
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1.obtain label-prompt list
    label_dic = gen_label_list(args)

    # 2.define the diffusers pipeline
    pipe = StableDiffusionLatents2ImgPipeline.from_pretrained(args.diffusion_checkpoints_path, torch_dtype=torch.float16)
    pipe = pipe.to(args.device)

    # 3.load prototypes from json file
    prototypes = load_prototype(args)

    # 4.generate initialized synthetic images and save them for refine
    gen_syn_images(pipe=pipe, prototypes=prototypes, label_list=label_dic, args=args)


if __name__ == "__main__" : 
    main()
