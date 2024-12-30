import os
import sys
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid


def load_dataset(args):
    # Obtain dataloader
    transform_train = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
    ])
    if args.dataset == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=False,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=False,
                                   transform=transform_test)
    elif args.dataset == 'cifar100':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.486, 0.441), (0.267, 0.256, 0.276))
        ])
        trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=False,
                                    transform=transform_train)
        testset = datasets.CIFAR100(root=args.data_dir, train=False, download=False,
                                   transform=transform_test)  
    elif args.dataset == 'imagenet':
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = datasets.ImageFolder(root=args.data_dir + "/train", 
                                        transform=transform_train)
        testset = datasets.ImageFolder(root=args.data_dir + "/val", 
                                       transform=transform_train)
    elif args.dataset == 'tiny_imagenet':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = datasets.ImageFolder(root=args.data_dir + "/train", 
                                        transform=transform_train)
        testset = datasets.ImageFolder(root=args.data_dir + "/val", 
                                       transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    return trainloader, testloader


def load_syn_dataset(args):
    # Obtain dataloader
    if args.dataset == 'syn_cifar10':
        transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])
        trainset = datasets.ImageFolder(root=args.data_dir, 
                                        transform=transform_train)

    elif args.dataset == 'syn_imagenet':
        transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = datasets.ImageFolder(root=args.data_dir, 
                                        transform=transform_train)
       

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    return trainloader


def gen_label_list(args):
    # obtain label-prompt list
    with open(args.label_file_path, "r") as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        line = line.strip()
        label = line.split('\t')[0]
        labels.append(label)
    
    return labels
