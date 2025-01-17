import os
import random
import argparse
import collections
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from synthesize.utils import *
from validation.utils import ImageFolder
from validation.utils import (
    ImageFolder,
    ShufflePatches,
    mix_aug,
    AverageMeter,
    accuracy,
    get_parameters,
)


def init_images(args, model, image_syn):
    if args.subset != 'cifar10' and args.subset != 'cifar100':
        trainset = ImageFolder(
            classes=args.classes,
            ipc=args.mipc,
            shuffle=True,
            root=args.train_dir,
            transform=None,
        )

        trainset.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                MultiRandomCrop(
                    num_crop=args.num_crop, size=args.input_size, factor=args.factor
                ),
                normalize,
            ]
        )

    if args.subset == 'cifar10':
        # train_data = datasets.CIFAR10(root=args.train_dir, train=True, download=True)
        # test_data = datasets.CIFAR10(root=args.train_dir, train=False, download=True)
        def create_imagefolder_structure(dataset, split='train'):
            root_dir = f'./cifar10/{split}'
            os.makedirs(root_dir, exist_ok=True)
    
            class_names = dataset.classes  # ['airplane', 'automobile', ...]
    
            for idx, class_name in enumerate(class_names):
                class_dir = os.path.join(root_dir, 'n'+ str(idx))  
                os.makedirs(class_dir, exist_ok=True)
    
            for i in range(len(dataset)):
                image, label = dataset[i]
                class_name = str(label)  
                img_path = os.path.join(root_dir, 'n'+class_name, f'{i}.png')  
                image.save(img_path)

        # create_imagefolder_structure(train_data, split='train')
        # create_imagefolder_structure(test_data, split='test')
        
        trainset = ImageFolder(
            classes=args.classes,
            ipc=args.mipc,
            shuffle=True,
            root='./cifar10/train/',
            transform=None,
        )
        
        trainset.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            MultiRandomCrop(
                    num_crop=args.num_crop, size=args.input_size, factor=args.factor
                ),
            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            normalize,
        ]
    )

    if args.subset == 'cifar100':
        # train_data = datasets.CIFAR100(root=args.train_dir, train=True, download=True)
        # test_data = datasets.CIFAR100(root=args.train_dir, train=False, download=True)
        def create_imagefolder_structure(dataset, split='train'):
            root_dir = f'./cifar100/{split}'
            os.makedirs(root_dir, exist_ok=True)
    
            class_names = dataset.classes  # ['airplane', 'automobile', ...]
    
            for idx, class_name in enumerate(class_names):
                class_dir = os.path.join(root_dir, 'n'+ str(idx))  
                os.makedirs(class_dir, exist_ok=True)
    
            for i in range(len(dataset)):
                image, label = dataset[i]
                class_name = str(label)  
                img_path = os.path.join(root_dir, 'n'+class_name, f'{i}.png') 
                image.save(img_path)

        # create_imagefolder_structure(train_data, split='train')
        # create_imagefolder_structure(test_data, split='test')
        
        trainset = ImageFolder(
            classes=args.classes,
            ipc=args.mipc,
            shuffle=True,
            root='./cifar100/train/',
            transform=None,
        )
        trainset.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            MultiRandomCrop(
                    num_crop=args.num_crop, size=args.input_size, factor=args.factor
                ),
            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            normalize,
        ]
    )

    # 一个batch包含一类的所有图片
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.mipc,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    for c, (images, labels) in enumerate(tqdm(train_loader)):
        images = selector(
            args.ipc * args.factor**2,
            model,
            images,
            labels,
            args.input_size,
            m=args.num_crop,
        )
        images = mix_images(images, args.input_size, args.factor, args.ipc)
        save_images(args, denormalize(images), c)
        # image_syn.data[c*args.ipc:(c+1)*args.ipc] = images.detach().data


def save_images(args, images, class_id):
    for id in range(images.shape[0]):
        dir_path = "{}/{:05d}".format(args.syn_data_path, class_id)
        place_to_store = dir_path + "/class{:05d}_id{:05d}.jpg".format(class_id, id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)
        
        
def save_pt(args):
    root_dir = args.syn_data_path
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # mean = [0.4914, 0.4822, 0.4465]
    # std = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),  
        transforms.ToTensor(),        
        ShufflePatches(args.factor),
        transforms.RandomResizedCrop(
            size=args.input_size,
            scale=(1 / args.factor, args.max_scale_crops),
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=mean, std=std),
    ])

    image_data = []
    labels = []

    for label, subfolder in enumerate(os.listdir(root_dir)):
        subfolder_path = os.path.join(root_dir, subfolder)
    
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)
            
                if file_path.endswith('.jpg') or file_path.endswith('.png') or file_path.endswith('.png'):
                    image = Image.open(file_path)
                    image = transform(image)  
                
                    image_data.append(image)
                    labels.append(label)  

    image_data_tensor = torch.stack(image_data)  # [num_images, 3, 64, 64]
    labels_tensor = torch.tensor(labels)  # [num_images]

    print(image_data_tensor.shape)
    os.makedirs(args.ranking_data_path, exist_ok=True)
    torch.save(image_data_tensor.cpu(), os.path.join(args.ranking_data_path, "images_%s_%d_best.pt"%(args.subset, args.ipc)))


def main(args):
    print(args)
    with torch.no_grad():
        if not os.path.exists(args.syn_data_path):
            os.makedirs(args.syn_data_path)
        else:
            shutil.rmtree(args.syn_data_path)
            os.makedirs(args.syn_data_path)
        if not os.path.exists(args.ranking_data_path):
            os.makedirs(args.ranking_data_path)

        model_teacher = load_model(
            model_name=args.arch_name,
            dataset=args.subset,
            pretrained=True,
            classes=args.classes,
        )

        model_teacher = nn.DataParallel(model_teacher).cuda()
        model_teacher.eval()
        for p in model_teacher.parameters():
            p.requires_grad = False
            
        image_syn = torch.randn(size=(args.nclass*args.ipc, 3, args.input_size, args.input_size), dtype=torch.float, requires_grad=True).cuda()
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(args.nclass)], dtype=torch.long, requires_grad=False).cuda().view(-1)

        init_images(args, model_teacher, image_syn)
        # torch.save(image_syn.cpu(), os.path.join(args.ranking_data_path, "images_%s_%d_best.pt"%(args.subset, args.ipc)))
        
        # save_pt(args=args)
        
        
        


if __name__ == "__main__":
    pass
