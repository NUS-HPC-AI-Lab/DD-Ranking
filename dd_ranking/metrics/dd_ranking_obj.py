import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List
from torch import Tensor
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from dd_ranking.utils import build_model
from dd_ranking.utils import TensorDataset, get_random_images, get_dataset
from dd_ranking.utils import set_seed
from dd_ranking.hard_label import compute_hard_label_metrics
from dd_ranking.train import train_one_epoch, validate


class DD_Ranking_Objective:

    def __init__(self, dataset: str="CIFAR10", real_data_path: str=None, num_classes: int=10, 
                 ipc: int=1, model_name: str=None, use_default_transform: bool=True, num_eval: int=5, 
                 num_epochs: int=100, lr: float=0.01, batch_size: int=256,
                 custom_transform: transforms.Compose=None, device: str="cuda"):
        channel, im_size, num_classes, dst_train, dst_test, class_map, class_map_inv = get_dataset(dataset, real_data_path)
        self.images_train, self.labels_train, self.class_indices_train = self.load_real_data(dst_train, class_map, num_classes)
        self.test_loader = DataLoader(dst_test, batch_size=batch_size, shuffle=False)

        # data info
        self.im_size = im_size
        self.num_classes = num_classes
        self.ipc = ipc

        # training info
        self.num_eval = num_eval
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device
    
    def load_real_data(self, dataset, class_map, num_classes):
        images_all = []
        labels_all = []
        class_indices = [[] for c in range(num_classes)]
        for i, (image, label) in enumerate(dataset):
            images_all.append(torch.unsqueeze(image, 0))
            labels_all.append(class_map[label].item())
        images_all = torch.cat(images_all, dim=0)
        labels_all = torch.tensor(labels_all)
        for i, label in enumerate(labels_all):
            class_indices.append(i)
        
        return images_all, labels_all, class_indices
    
    def compute_metrics(self):
        pass


class Soft_Label_Objective(DD_Ranking_Objective):
    def __init__(self, soft_labels: Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_labels = soft_labels
        self.soft_label_dataset = TensorDataset(self.syn_images.detach().clone(), self.soft_labels.detach().clone())
        self.teacher_model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=True, device=self.device)
        self.teacher_model.eval()

    @staticmethod
    def SoftCrossEntropy(inputs, target):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch_size = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch_size
        return loss

    def compute_hard_label_metrics(self, model, images, hard_labels=None):
        if hard_labels is None:
            hard_labels = torch.tensor([np.ones(self.ipc) * i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False).view(-1)
        hard_label_dataset = TensorDataset(images, hard_labels)
        train_loader = DataLoader(hard_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs * len(train_loader))

        best_acc1 = 0
        for epoch in range(self.num_epochs):
            train_one_epoch(model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device)
            metric = validate(model, self.test_loader, device=self.device)
            if metric['top1'] > best_acc1:
                best_acc1 = metric['top1']

        return best_acc1
        
    def compute_soft_label_metrics(self, model, images, soft_labels):
        soft_label_dataset = TensorDataset(images, soft_labels)
        train_loader = DataLoader(soft_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = self.SoftCrossEntropy
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs * len(train_loader))
        
        best_acc1 = 0
        for epoch in range(self.num_epochs):
            train_one_epoch(model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device)
            metric = validate(model, self.test_loader, device=self.device)
            if metric['top1'] > best_acc1:
                best_acc1 = metric['top1']
        
        return best_acc1

    def generate_soft_labels_for_random_data(self, random_images):
        hard_labels = torch.tensor([np.ones(self.ipc) * i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False).view(-1)
        random_dataset = TensorDataset(random_images, hard_labels)
        train_loader = DataLoader(random_dataset, batch_size=self.batch_size, shuffle=False)
        soft_labels = []
        with torch.no_grad():
            for images, _ in train_loader:
                images = images.to(self.device)
                soft_labels.append(self.teacher_model(images).detach().cpu())
        soft_labels = torch.cat(soft_labels, dim=0)
        return soft_labels
    
    def compute_metrics(self, syn_images, soft_labels):
        obj_metrics = []
        for i in range(self.num_eval):
            set_seed()
            print(f"{i+1}th Evaluation")

            print("Caculating syn data hard label metrics...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            syn_data_hard_label_acc = self.compute_hard_label_metrics(model, syn_images, hard_labels=None)
            del model

            print("Caculating full data hard label metrics...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            full_data_hard_label_acc = self.compute_hard_label_metrics(model, self.images_train, hard_labels=self.labels_train)
            del model

            print("Caculating syn data soft label metrics...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            syn_data_soft_label_acc = self.compute_soft_label_metrics(model, syn_images, soft_labels)
            del model
            
            print("Caculating random data soft label metrics...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            random_images = get_random_images(self.images_train, self.class_indices_train, self.ipc)
            random_data_soft_labels = self.generate_soft_labels_for_random_data(random_images)
            random_data_soft_label_acc = self.compute_soft_label_metrics(model, random_images, random_data_soft_labels)
            del model

            numerator = 1.00 * (syn_data_soft_label_acc - random_data_soft_label_acc)
            denominator = 1.00 * (full_data_hard_label_acc - syn_data_hard_label_acc)
            obj_metrics.append(numerator / denominator)
        obj_metrics_mean = np.mean(obj_metrics)
        obj_metrics_std = np.std(obj_metrics)

        print(f"Soft Label Objective Metrics Mean: {obj_metrics_mean * 100:.2f}%  Std: {obj_metrics_std * 100:.2f}%")
        return obj_metrics_mean, obj_metrics_std


class KL_Divergence_Objective(DD_Ranking_Objective):
    def __init__(self, teacher_model: str, temperature: float=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.teacher_model = self.build_model(self.teacher_model)

    @staticmethod
    def KLDivLoss(stu_outputs, tea_outputs):
        stu_probs = F.log_softmax(stu_outputs / self.temperature, dim=1)
        tea_probs = F.log_softmax(tea_outputs / self.temperature, dim=1)
        loss = F.kl_div(stu_probs, tea_probs, reduction='batchmean') * (self.temperature ** 2)
        return loss
    
    def compute_hard_label_metrics(self, model, images, hard_labels=None):
        if hard_labels is None:
            hard_labels = torch.tensor([np.ones(self.ipc) * i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False).view(-1)
        hard_label_dataset = TensorDataset(images, hard_labels)
        train_loader = DataLoader(hard_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs * len(train_loader))

        best_acc1 = 0
        for epoch in range(self.num_epochs):
            train_one_epoch(model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device)
            metric = validate(model, self.test_loader, device=self.device)
            if metric['top1'] > best_acc1:
                best_acc1 = metric['top1']

        return best_acc1
        
    def compute_kl_divergence_metrics(self, model, images, labels=None):
        if labels is None:
            labels = torch.tensor([np.ones(self.ipc) * i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False).view(-1)
        soft_label_dataset = TensorDataset(images, labels)
        train_loader = DataLoader(soft_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = self.KLDivLoss
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs * len(train_loader))

        best_acc1 = 0
        for epoch in range(self.num_epochs):
            train_one_epoch(model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device)
            metric = validate(model, self.test_loader, device=self.device)
            if metric['top1'] > best_acc1:
                best_acc1 = metric['top1']
        
        return best_acc1
    
    def compute_metrics(self, images, labels=None):
        obj_metrics = []
        for i in range(self.num_eval):
            set_seed()
            print(f"{i+1}th Evaluation")

            print("Caculating syn data hard label metrics...")
            model = self.build_model(self.model_name)
            syn_data_hard_label_acc = self.compute_hard_label_metrics(model, images, hard_labels=None)
            del model

            print("Caculating full data hard label metrics...")
            model = self.build_model(self.model_name)
            full_data_hard_label_acc = self.compute_hard_label_metrics(model, self.images_train, hard_labels=self.labels_train)
            del model

            print("Caculating syn data kl divergence metrics...")
            model = self.build_model(self.model_name)
            syn_data_kl_divergence_acc = self.compute_kl_divergence_metrics(model, images, labels=labels)
            del model

            print("Caculating random data kl divergence metrics...")
            model = self.build_model(self.model_name)
            random_images = get_random_images(self.images_train, self.class_indices_train, self.ipc)
            random_data_kl_divergence_acc = self.compute_kl_divergence_metrics(model, random_images, labels=labels)
            del model

            numerator = 1.00 * (syn_data_kl_divergence_acc - random_data_kl_divergence_acc)
            denominator = 1.00 * (full_data_hard_label_acc - syn_data_hard_label_acc)
            obj_metrics.append(numerator / denominator)
        obj_metrics_mean = np.mean(obj_metrics)
        obj_metrics_std = np.std(obj_metrics)

        print(f"KL Divergence Objective Metrics Mean: {obj_metrics_mean * 100:.2f}%  Std: {obj_metrics_std * 100:.2f}%")
        return obj_metrics_mean, obj_metrics_std


if __name__ == "__main__":
    images = torch.randn(10, 3, 32, 32)
    obj = DD_Ranking_Objective(images=images, model_name="ConvNet")
    print(obj.syn_images.shape)
    print(obj.compute_metrics())