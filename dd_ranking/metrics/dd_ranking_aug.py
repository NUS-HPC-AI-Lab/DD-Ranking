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
from dd_ranking.utils import get_dataset, get_random_images, build_model, save_results
from dd_ranking.utils import set_seed, train_one_epoch, validate
from dd_ranking.aug import DSA_Augmentation, Mixup_Augmentation, Cutmix_Augmentation, ZCA_Whitening_Augmentation


class Augmentation_Metrics:
    def __init__(self, dataset: str, real_data_path: str, model_name: str, ipc: int, 
                 im_size: tuple=(32, 32), soft_label_mode: str='N', save_path: str=None, device: str="cuda"):
        
        channel, im_size, num_classes, dst_train, dst_test, class_map, class_map_inv = get_dataset(dataset, real_data_path, im_size)
        self.images_train, self.labels_train, self.class_indices_train = self.load_real_data(dst_train, class_map, num_classes)
        
        self.ipc = ipc
        self.model_name = model_name
        self.num_classes = num_classes
        self.im_size = im_size
        self.device = device
        self.soft_label_mode = soft_label_mode

        if not save_path:
            save_path = f"./results/{dataset}/{model_name}/ipc{ipc}/aug_scores.csv"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        self.save_path = save_path

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
                aug_func=self.transform, 
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

    def compute_metrics(self, images, labels=None):
        if not labels:
            labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), dtype=torch.long, requires_grad=False).view(-1)
        
        hard_recs = []
        aug_imps = []
        aug_metrics = []
        for i in range(self.num_eval):
            set_seed()
            print(f"####################### {i+1}th Evaluation #######################")

            print("Caculating syn data no augmentation metrics...")
            syn_data_default_aug_acc, best_lr = self.hyper_param_search_for_no_aug(
                images=images, 
                labels=labels
            )
            print(f"Syn data no augmentation acc: {syn_data_default_aug_acc:.2f}%")

            print("Caculating syn data custom augmentation metrics...")
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
            full_data_default_aug_acc = self.compute_no_aug_metrics(
                model=model, 
                images=self.images_train, 
                lr=self.default_lr, 
                labels=self.labels_train
            )
            del model
            print(f"Full data no augmentation acc: {full_data_default_aug_acc:.2f}%")

            hard_rec = 1.00 * (full_data_default_aug_acc - syn_data_default_aug_acc)
            aug_imp = 1.00 * (syn_data_custom_aug_acc - random_data_custom_aug_acc)
            aug_metrics.append(aug_imp / hard_rec)

            hard_recs.append(hard_rec)
            aug_imps.append(aug_imp)
        
        results_to_save = {
            "hard_recs": hard_recs,
            "aug_imps": aug_imps,
            "aug_metrics": aug_metrics
        }
        save_results(results_to_save, self.save_path)

        hard_recs_mean = np.mean(hard_recs)
        hard_recs_std = np.std(hard_recs)
        aug_imps_mean = np.mean(aug_imps)
        aug_imps_std = np.std(aug_imps)
        aug_metrics_mean = np.mean(aug_metrics)
        aug_metrics_std = np.std(aug_metrics)

        return {
            "hard_recs_mean": hard_recs_mean,
            "hard_recs_std": hard_recs_std,
            "aug_imps_mean": aug_imps_mean,
            "aug_imps_std": aug_imps_std,
            "aug_metrics_mean": aug_metrics_mean,
            "aug_metrics_std": aug_metrics_std
        }
        

class DSA_Augmentation_Metrics(Augmentation_Metrics):

    def __init__(self, func_names: list, params: dict, seed: int=-1, aug_mode: str='M', *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.params = params
        self.seed = seed
        self.aug_mode = aug_mode
        # self.transform_funcs = self.create_transform_funcs(func_names)
        self.aug_func = DSA_Augmentation(func_names, params, seed, aug_mode)
        # dsa params for training a model
        self.num_epochs = 1000
        
    def transform(self, images):
        return self.aug_func(images)
    
    def compute_metrics(self, images, labels=None):
        aug_metrics = super().compute_metrics(images, labels=labels)
        print(f"DSA Hard Recovery Mean: {aug_metrics['hard_recs_mean']:.2f}%  Std: {aug_metrics['hard_recs_std']:.2f}%")
        print(f"DSA Augmentation Improvement Mean: {aug_metrics['aug_imps_mean']:.2f}%  Std: {aug_metrics['aug_imps_std']:.2f}%")
        print(f"DSA Augmentation Metrics Mean: {aug_metrics['aug_metrics_mean']:.2f}%  Std: {aug_metrics['aug_metrics_std']:.2f}%")
        return aug_metrics


class ZCA_Whitening_Augmentation_Metrics(Augmentation_Metrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_func = ZCA_Whitening_Augmentation()

    def transform(self, images):
        return self.aug_func(images)
    
    def compute_metrics(self, images, labels=None):
        aug_metrics = super().compute_metrics(images, labels=labels)
        print(f"ZCA Whitening Hard Recovery Mean: {aug_metrics['hard_recs_mean']:.2f}%  Std: {aug_metrics['hard_recs_std']:.2f}%")
        print(f"ZCA Whitening Augmentation Improvement Mean: {aug_metrics['aug_imps_mean']:.2f}%  Std: {aug_metrics['aug_imps_std']:.2f}%")
        print(f"ZCA Whitening Augmentation Metrics Mean: {aug_metrics['aug_metrics_mean']:.2f}%  Std: {aug_metrics['aug_metrics_std']:.2f}%")
        return aug_metrics

        
class Mixup_Augmentation_Metrics(Augmentation_Metrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_func = Mixup_Augmentation(self.device)
        
    def transform(self, images):
        return self.aug_func(images)
    
    def compute_metrics(self, images, labels=None):
        aug_metrics = super().compute_metrics(images, labels=labels)
        print(f"Mixup Hard Recovery Mean: {aug_metrics['hard_recs_mean']:.2f}%  Std: {aug_metrics['hard_recs_std']:.2f}%")
        print(f"Mixup Augmentation Improvement Mean: {aug_metrics['aug_imps_mean']:.2f}%  Std: {aug_metrics['aug_imps_std']:.2f}%")
        print(f"Mixup Augmentation Metrics Mean: {aug_metrics['aug_metrics_mean']:.2f}%  Std: {aug_metrics['aug_metrics_std']:.2f}%")
        return aug_metrics


class Cutmix_Augmentation_Metrics(Augmentation_Metrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_func = Cutmix_Augmentation(self.device)

    def transform(self, images):
        return self.aug_func(images)

    def compute_metrics(self, images, labels=None):
        aug_metrics = super().compute_metrics(images, labels=labels)
        print(f"Cutmix Hard Recovery Mean: {aug_metrics['hard_recs_mean']:.2f}%  Std: {aug_metrics['hard_recs_std']:.2f}%")
        print(f"Cutmix Augmentation Improvement Mean: {aug_metrics['aug_imps_mean']:.2f}%  Std: {aug_metrics['aug_imps_std']:.2f}%")
        print(f"Cutmix Augmentation Metrics Mean: {aug_metrics['aug_metrics_mean']:.2f}%  Std: {aug_metrics['aug_metrics_std']:.2f}%")
        return aug_metrics
