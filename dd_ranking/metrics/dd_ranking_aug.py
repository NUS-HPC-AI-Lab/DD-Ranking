import os
import torch
import random
import kornia
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from dd_ranking.utils.utils import get_dataset, get_random_images, build_model
from dd_ranking.utils.utils import set_seed, train_one_epoch, validate


class Augmentation:
    def __init__(self, dataset: str, real_data_path: str, model_name: str, ipc: int, im_size: tuple=(32, 32), soft_label_mode: str='N', device: str="cuda"):
        
        channel, im_size, num_classes, dst_train, dst_test, class_map, class_map_inv = get_dataset(dataset, real_data_path, im_size)
        self.images_train, self.labels_train, self.class_indices_train = self.load_real_data(dst_train, class_map, num_classes)
        
        self.ipc = ipc
        self.model_name = model_name
        self.num_classes = num_classes
        self.im_size = im_size
        self.device = device
        self.soft_label_mode = soft_label_mode

        # default params for training a model
        self.num_eval = 5
        self.test_interval = 10
        self.batch_size = 256
        self.num_epochs = 300
        self.default_lr = 0.01

        self.test_loader = DataLoader(dst_test, batch_size=self.batch_size, shuffle=False)

    def load_real_data(self, dataset, class_map, num_classes):
        images_all = []
        labels_all = []
        class_indices = [[] for c in range(num_classes)]
        for i, (image, label) in enumerate(dataset):
            images_all.append(torch.unsqueeze(image, 0))
            labels_all.append(class_map[label])
        images_all = torch.cat(images_all, dim=0)
        labels_all = torch.tensor(labels_all)
        for i, label in enumerate(labels_all):
            class_indices[label].append(i)
        return images_all, labels_all, class_indices

    def apply_augmentation(self, images):
        pass

    def hyper_param_search_for_no_aug(self, images, labels):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]

        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            print(f"Searching lr:{lr} for no augmentation...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            acc = self.compute_no_aug_metrics(model, images, lr, labels=labels)
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr

    def hyper_param_search_for_custom_aug(self, images, labels):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]

        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            print(f"Searching lr:{lr} for custom augmentation...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            acc = self.compute_custom_aug_metrics(model, images, lr, labels=labels)
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr
    
    def compute_custom_aug_metrics(self, model, images, lr, labels):
        train_dataset = TensorDataset(images, labels)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)
        lr_scheduler = StepLR(optimizer, step_size=self.num_epochs // 2, gamma=0.1)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_one_epoch(
                epoch=epoch, 
                stu_model=model, 
                loader=train_loader, 
                loss_fn=loss_fn, 
                optimizer=optimizer, 
                lr_scheduler=lr_scheduler, 
                aug_func=self.apply_augmentation, 
                device=self.device,
                soft_label_mode=self.soft_label_mode
            )
            if epoch % self.test_interval == 0:
                metric = validate(
                    model=model, 
                    loader=self.test_loader,
                    device=self.device
                )
                if metric['top1'] > best_acc1:
                    best_acc1 = metric['top1']

        return best_acc1
    
    def compute_no_aug_metrics(self, model, images, lr, labels):
        train_dataset = TensorDataset(images, labels)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)
        lr_scheduler = StepLR(optimizer, step_size=self.num_epochs // 2, gamma=0.1)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_one_epoch(
                epoch=epoch, 
                stu_model=model, 
                loader=train_loader, 
                loss_fn=loss_fn, 
                optimizer=optimizer, 
                lr_scheduler=lr_scheduler, 
                device=self.device,
                soft_label_mode=self.soft_label_mode
            )
            if epoch % self.test_interval == 0:
                metric = validate(
                    model=model, 
                    loader=self.test_loader, 
                    device=self.device
                )
                if metric['top1'] > best_acc1:
                    best_acc1 = metric['top1']

        return best_acc1

    def compute_metrics(self, images, labels=None, syn_lr=None):
        if not labels:
            labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), dtype=torch.long, requires_grad=False).view(-1)
        
        aug_metrics = []
        for i in range(self.num_eval):
            set_seed()
            print(f"{i+1}th Evaluation")

            print("Caculating syn data no augmentation metrics...")
            syn_data_default_aug_acc, best_lr = self.hyper_param_search_for_no_aug(
                images=images, 
                labels=labels
            )
            print(f"Syn data no augmentation acc: {syn_data_default_aug_acc:.2f}%")

            print("Caculating syn data custom augmentation metrics...")
            if syn_lr:
                model = build_model(
                    model_name=self.model_name, 
                    num_classes=self.num_classes, 
                    im_size=self.im_size, 
                    pretrained=False, 
                    device=self.device
                )
                syn_data_custom_aug_acc = self.compute_custom_aug_metrics(
                    model=model, 
                    images=images, 
                    lr=syn_lr, 
                    labels=labels
                )
                del model
            else:
                syn_data_custom_aug_acc, best_lr = self.hyper_param_search_for_custom_aug(
                    images=images, 
                    labels=labels
                )
            print(f"Syn data custom augmentation acc: {syn_data_custom_aug_acc:.2f}%")

            print("Caculating random data custom augmentation metrics...")
            random_images, random_labels = get_random_images(
                images_all=self.images_train, 
                labels_all=self.labels_train, 
                class_indices=self.class_indices_train, 
                n_images_per_class=self.ipc
            )
            random_data_custom_aug_acc, best_lr = self.hyper_param_search_for_custom_aug(
                images=random_images, 
                labels=random_labels
            )
            print(f"Random data custom augmentation acc: {random_data_custom_aug_acc:.2f}%")

            print("Caculating full data no augmentation metrics...")
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False, 
                device=self.device
            )
            full_data_default_aug_acc, best_lr = self.compute_no_aug_metrics(
                model=model, 
                images=self.images_train, 
                lr=self.default_lr, 
                labels=self.labels_train
            )
            del model
            print(f"Full data no augmentation acc: {full_data_default_aug_acc:.2f}%")

            numerator = 1.00 * (syn_data_custom_aug_acc - random_data_custom_aug_acc)
            denominator = 1.00 * (full_data_default_aug_acc - syn_data_default_aug_acc)
            aug_metrics.append(numerator / denominator)
        aug_metrics_mean = np.mean(aug_metrics)
        aug_metrics_std = np.std(aug_metrics)

        return aug_metrics_mean, aug_metrics_std
        

class DSA_Augmentation(Augmentation):

    def __init__(self, func_names: list, params: dict, seed: int=-1, aug_mode: str='M', *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.params = params
        self.seed = seed
        self.aug_mode = aug_mode
        self.transform_funcs = self.create_transform_funcs(func_names)

        # dsa params for training a model
        self.num_epochs = 1000

    def create_transform_funcs(self, func_names):
        funcs = []
        for func_name in func_names:
            funcs.append(getattr(self, 'rand_'+func_name))
        return funcs
    
    def set_seed_DiffAug(self):
        if self.params["latestseed"] == -1:
            return
        else:
            torch.random.manual_seed(self.params["latestseed"])
            self.params["latestseed"] += 1
    
    # The following differentiable augmentation strategies are adapted from https://github.com/VICO-UoE/DatasetCondensation
    def rand_scale(self, x):
        # x>1, max scale
        # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
        ratio = self.params["ratio_scale"]
        self.set_seed_DiffAug()
        sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
        self.set_seed_DiffAug()
        sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
        theta = [[[sx[i], 0,  0],
                [0,  sy[i], 0],] for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        if self.params["siamese"]: # Siamese augmentation:
            theta[:] = theta[0]
        grid = F.affine_grid(theta, x.shape).to(x.device)
        x = F.grid_sample(x, grid)
        return x

    def rand_rotate(self, x): # [-180, 180], 90: anticlockwise 90 degree
        ratio = self.params["ratio_rotate"]
        self.set_seed_DiffAug()
        theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
        theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
            [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        if self.params["siamese"]: # Siamese augmentation:
            theta[:] = theta[0]
        grid = F.affine_grid(theta, x.shape).to(x.device)
        x = F.grid_sample(x, grid)
        return x

    def rand_flip(self, x):
        prob = self.params["prob_flip"]
        self.set_seed_DiffAug()
        randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
        if self.params["siamese"]: # Siamese augmentation:
            randf[:] = randf[0]
        return torch.where(randf < prob, x.flip(3), x)

    def rand_brightness(self, x):
        ratio = self.params["brightness"]
        self.set_seed_DiffAug()
        randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        if self.params["siamese"]: # Siamese augmentation:
            randb[:] = randb[0]
        x = x + (randb - 0.5)*ratio
        return x

    def rand_saturation(self, x):
        ratio = self.params["saturation"]
        x_mean = x.mean(dim=1, keepdim=True)
        self.set_seed_DiffAug()
        rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        if self.params["siamese"]: # Siamese augmentation:
            rands[:] = rands[0]
        x = (x - x_mean) * (rands * ratio) + x_mean
        return x

    def rand_contrast(self, x):
        ratio = self.params["contrast"]
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        self.set_seed_DiffAug()
        randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        if self.params["siamese"]: # Siamese augmentation:
            randc[:] = randc[0]
        x = (x - x_mean) * (randc + ratio) + x_mean
        return x

    def rand_crop(self, x):
        # The image is padded on its surrounding and then cropped.
        ratio = self.params["ratio_crop_pad"]
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        self.set_seed_DiffAug()
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        self.set_seed_DiffAug()
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        if self.params["siamese"]: # Siamese augmentation:
            translation_x[:] = translation_x[0]
            translation_y[:] = translation_y[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

    def rand_cutout(self, x):
        ratio = self.params["ratio_cutout"]
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        self.set_seed_DiffAug()
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        self.set_seed_DiffAug()
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        if self.params["siamese"]: # Siamese augmentation:
            offset_x[:] = offset_x[0]
            offset_y[:] = offset_y[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x
        
    def apply_augmentation(self, images):
        
        if not self.transform_funcs:
            return images

        if self.seed == -1:  
            self.params["siamese"] = False
        else: 
            self.params["siamese"] = True
            
        self.params["latestseed"] = self.seed
        
        transformed_images = images
        if self.aug_mode == 'M': # original
            for f in self.transform_funcs:
                transformed_images = f(transformed_images)
                
        elif self.aug_mode == 'S':
            self.set_seed_DiffAug()
            p = self.transform_funcs[torch.randint(0, len(self.transform_funcs), size=(1,)).item()]
            transformed_images = p(transformed_images)
                
        transformed_images = transformed_images.contiguous()
            
        return transformed_images
    
    def compute_metrics(self, images, labels=None, syn_lr=None):
        aug_metrics = super().compute_metrics(images, labels=labels, syn_lr=syn_lr)
        print(f"DSA Augmentation Metrics Mean: {aug_metrics[0] * 100:.2f}%  Std: {aug_metrics[1] * 100:.2f}%")
        return aug_metrics


class ZCA_Whitening_Augmentation(Augmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = kornia.enhance.ZCAWhitening()

    def apply_augmentation(self, images):
        return self.transform(images, include_fit=True)
    
    def compute_metrics(self, images, labels=None, syn_lr=None):
        aug_metrics = super().compute_metrics(images, labels=labels, syn_lr=syn_lr)
        print(f"ZCA Whitening Augmentation Metrics Mean: {aug_metrics[0] * 100:.2f}%  Std: {aug_metrics[1] * 100:.2f}%")
        return aug_metrics
        
        
class Mixup_Augmentation(Augmentation):
    def __init__(self, params: dict, batch_size: int=256, num_epochs: int=300, lr: float=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = kornia.augmentation.RandomMixUpV2(
            lambda_val = params["lambda_range"],
            same_on_batch = params["same_on_batch"],
            keepdim = params["keepdim"],
            p = params["prob"]
        )
        
    def apply_augmentation(self, images, seed):
        labels = torch.tensor([np.ones(self.ipc) * i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False).view(-1)
        return self.transform(images, labels)
    
    def compute_metrics(self, images, labels=None, syn_lr=None):
        aug_metrics = super().compute_metrics(images, labels=labels, syn_lr=syn_lr)
        print(f"Mixup Augmentation Metrics Mean: {aug_metrics[0] * 100:.2f}%  Std: {aug_metrics[1] * 100:.2f}%")
        return aug_metrics


class Cutmix_Augmentation(Augmentation):
    def __init__(self, params: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = kornia.augmentation.RandomCutMixV2(
            num_mix = params["times"],
            cut_size = params["size"],
            same_on_batch = params["same_on_batch"],
            beta = params["beta"],
            keepdim = params["keep_dim"],
            p = params["prob"]
        )

    def apply_augmentation(self, images):
        labels = torch.tensor([np.ones(self.ipc) * i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False).view(-1)
        return self.transform(images, labels)

    def compute_metrics(self, images, labels=None, syn_lr=None):
        aug_metrics = super().compute_metrics(images, labels=labels, syn_lr=syn_lr)
        print(f"Cutmix Augmentation Metrics Mean: {aug_metrics[0] * 100:.2f}%  Std: {aug_metrics[1] * 100:.2f}%")
        return aug_metrics


if __name__ == "__main__":
    images = torch.randn(10, 3, 32, 32)
    labels = torch.randn(10)