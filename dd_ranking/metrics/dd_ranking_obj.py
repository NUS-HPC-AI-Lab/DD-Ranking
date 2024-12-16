import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from dd_ranking.utils.utils import build_model
from dd_ranking.utils.utils import TensorDataset, get_random_images, get_dataset
from dd_ranking.utils.utils import set_seed
from dd_ranking.utils.utils import train_one_epoch, validate
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from dd_ranking.utils.utils import build_model
from dd_ranking.utils.utils import TensorDataset, get_random_images, get_dataset
from dd_ranking.utils.utils import set_seed
from dd_ranking.utils.utils import train_one_epoch, validate


class DD_Ranking_Objective:

    def __init__(self, dataset: str, real_data_path: str, ipc: int, model_name: str, 
                 use_default_transform: bool=True, num_eval: int=5, im_size: tuple=(32, 32),
    def __init__(self, dataset: str, real_data_path: str, ipc: int, model_name: str, 
                 use_default_transform: bool=True, num_eval: int=5, im_size: tuple=(32, 32),
                 num_epochs: int=100, lr: float=0.01, batch_size: int=256,
                 custom_transform: transforms.Compose=None, device: str="cuda"):
        channel, im_size, num_classes, dst_train, dst_test, class_map, class_map_inv = get_dataset(dataset, real_data_path, im_size, custom_transform)
        channel, im_size, num_classes, dst_train, dst_test, class_map, class_map_inv = get_dataset(dataset, real_data_path, im_size, custom_transform)
        self.images_train, self.labels_train, self.class_indices_train = self.load_real_data(dst_train, class_map, num_classes)
        self.test_loader = DataLoader(dst_test, batch_size=batch_size, shuffle=False)

        # data info
        self.im_size = im_size
        self.num_classes = num_classes
        self.ipc = ipc

        # training info
        self.model_name = model_name
        self.model_name = model_name
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
            labels_all.append(class_map[label])
            labels_all.append(class_map[label])
        images_all = torch.cat(images_all, dim=0)
        labels_all = torch.tensor(labels_all)
        for i, label in enumerate(labels_all):
            class_indices.append(i)
        
        return images_all, labels_all, class_indices
    
    def compute_metrics(self):
        pass


class Soft_Label_Objective(DD_Ranking_Objective):
    def __init__(self, dataset: str, real_data_path: str, ipc: int, model_name: str, device: str="cuda", *args, **kwargs):
        super().__init__(dataset=dataset, real_data_path=real_data_path, ipc=ipc, model_name=model_name, device=device, *args, **kwargs)

        pretrained_model_path = get_pretrained_model_path(model_name, dataset, ipc)
        self.teacher_model = build_model(model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=True, device=self.device, model_path=pretrained_model_path)
    def __init__(self, dataset: str, real_data_path: str, ipc: int, model_name: str, device: str="cuda", *args, **kwargs):
        super().__init__(dataset=dataset, real_data_path=real_data_path, ipc=ipc, model_name=model_name, device=device, *args, **kwargs)

        pretrained_model_path = get_pretrained_model_path(model_name, dataset, ipc)
        self.teacher_model = build_model(model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=True, device=self.device, model_path=pretrained_model_path)
        self.teacher_model.eval()

    @staticmethod
    def SoftCrossEntropy(inputs, target):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch_size = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch_size
        return loss

    def hyper_param_search_for_hard_label(self, model, images, hard_labels=None):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            acc = self.compute_hard_label_metrics(model, images, lr, hard_labels=hard_labels)
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
        return best_acc, best_lr

    def hyper_param_search_for_soft_label(self, model, images, soft_labels=None):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            acc = self.compute_soft_label_metrics(model, images, lr, soft_labels=soft_labels)
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
        return best_acc, best_lr
    
    def compute_hard_label_metrics(self, model, images, lr, hard_labels=None):
        if hard_labels is None:
            hard_labels = torch.tensor([np.ones(self.ipc) * i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False).view(-1)
        hard_label_dataset = TensorDataset(images, hard_labels)
        train_loader = DataLoader(hard_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = StepLR(optimizer, step_size=self.num_epochs // 4, gamma=0.5)

        best_acc1 = 0
        for epoch in range(self.num_epochs):
            train_one_epoch(epoch, model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device)
            train_one_epoch(epoch, model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device)
            metric = validate(model, self.test_loader, device=self.device)
            if metric['top1'] > best_acc1:
                best_acc1 = metric['top1']

        return best_acc1
        
    def compute_soft_label_metrics(self, model, images, lr, soft_labels=None):
        if soft_labels is None:
            soft_labels = self.generate_soft_labels(images)
        soft_label_dataset = TensorDataset(images, soft_labels)
        train_loader = DataLoader(soft_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = self.SoftCrossEntropy
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = StepLR(optimizer, step_size=self.num_epochs // 4, gamma=0.5)
        
        best_acc1 = 0
        for epoch in range(self.num_epochs):
            train_one_epoch(epoch, model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device)
            train_one_epoch(epoch, model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device)
            metric = validate(model, self.test_loader, device=self.device)
            if metric['top1'] > best_acc1:
                best_acc1 = metric['top1']
        
        return best_acc1

    def generate_soft_labels(self, images):
        hard_labels = torch.tensor([np.ones(self.ipc) * i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False).view(-1)
        dataset = TensorDataset(images, hard_labels)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        soft_labels = []
        with torch.no_grad():
            for images, _ in train_loader:
                images = images.to(self.device)
                soft_labels.append(self.teacher_model(images).detach().cpu())
        soft_labels = torch.cat(soft_labels, dim=0)
        return soft_labels
    
    def compute_metrics(self, syn_images, syn_lr, soft_labels):

        obj_metrics = []
        for i in range(self.num_eval):
            set_seed()
            print(f"############### {i+1}th Evaluation ###############")
            print(f"############### {i+1}th Evaluation ###############")

            print("Caculating syn data hard label metrics...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            syn_data_hard_label_acc = self.hyper_param_search_for_hard_label(model, syn_images)
            del model

            print("Caculating full data hard label metrics...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            full_data_hard_label_acc = self.hyper_param_search_for_hard_label(model, self.images_train, hard_labels=self.labels_train)
            del model

            print("Caculating syn data soft label metrics...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            syn_data_soft_label_acc = self.compute_soft_label_metrics(model, syn_images, lr=syn_lr, soft_labels=soft_labels)
            del model
            
            print("Caculating random data soft label metrics...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            random_images = get_random_images(self.images_train, self.class_indices_train, self.ipc)
            random_data_soft_label_acc = self.hyper_param_search_for_soft_label(model, random_images)
            del model

            print("syn_data_soft_label_acc: ", syn_data_soft_label_acc)
            print("random_data_soft_label_acc: ", random_data_soft_label_acc)
            print("full_data_hard_label_acc: ", full_data_hard_label_acc)
            print("syn_data_hard_label_acc: ", syn_data_hard_label_acc)

            print("syn_data_soft_label_acc: ", syn_data_soft_label_acc)
            print("random_data_soft_label_acc: ", random_data_soft_label_acc)
            print("full_data_hard_label_acc: ", full_data_hard_label_acc)
            print("syn_data_hard_label_acc: ", syn_data_hard_label_acc)

            numerator = 1.00 * (syn_data_soft_label_acc - random_data_soft_label_acc)
            denominator = 1.00 * (full_data_hard_label_acc - syn_data_hard_label_acc)
            obj_metrics.append(numerator / denominator)
        obj_metrics_mean = np.mean(obj_metrics)
        obj_metrics_std = np.std(obj_metrics)

        print(f"Soft Label Objective Metrics Mean: {obj_metrics_mean * 100:.2f}%  Std: {obj_metrics_std * 100:.2f}%")
        return obj_metrics_mean, obj_metrics_std


class KL_Divergence_Objective(DD_Ranking_Objective):
    def __init__(self, dataset: str, real_data_path: str, ipc: int, stu_model_name: str, 
                 tea_model_name: str, temperature: float=1.0, *args, **kwargs):
        super().__init__(dataset=dataset, real_data_path=real_data_path, ipc=ipc, model_name=stu_model_name, *args, **kwargs)
        self.tea_model_name = tea_model_name
    def __init__(self, dataset: str, real_data_path: str, ipc: int, stu_model_name: str, 
                 tea_model_name: str, temperature: float=1.0, *args, **kwargs):
        super().__init__(dataset=dataset, real_data_path=real_data_path, ipc=ipc, model_name=stu_model_name, *args, **kwargs)
        self.tea_model_name = tea_model_name
        self.temperature = temperature

        pretrained_model_path = get_pretrained_model_path(tea_model_name, dataset, ipc)
        self.teacher_model = build_model(tea_model_name, num_classes=self.num_classes, im_size=self.im_size, 
                                         pretrained=True, device=self.device, model_path=pretrained_model_path)

        pretrained_model_path = get_pretrained_model_path(tea_model_name, dataset, ipc)
        self.teacher_model = build_model(tea_model_name, num_classes=self.num_classes, im_size=self.im_size, 
                                         pretrained=True, device=self.device, model_path=pretrained_model_path)

    @staticmethod
    def KLDivLoss(stu_outputs, tea_outputs, temperature=1.0):
        stu_probs = F.log_softmax(stu_outputs / temperature, dim=1)
        tea_probs = F.log_softmax(tea_outputs / temperature, dim=1)
        loss = F.kl_div(stu_probs, tea_probs, reduction='batchmean') * (temperature ** 2)
    def KLDivLoss(stu_outputs, tea_outputs, temperature=1.0):
        stu_probs = F.log_softmax(stu_outputs / temperature, dim=1)
        tea_probs = F.log_softmax(tea_outputs / temperature, dim=1)
        loss = F.kl_div(stu_probs, tea_probs, reduction='batchmean') * (temperature ** 2)
        return loss

    def hyper_param_search_for_hard_label(self, model, images, hard_labels=None):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            acc = self.compute_hard_label_metrics(model, images, lr, hard_labels=hard_labels)
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
        return best_acc, best_lr
    
    def hyper_param_search_for_kl_divergence(self, model, images, labels=None):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            acc = self.compute_kl_divergence_metrics(model, images, lr, labels=labels)
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
        return best_acc, best_lr
    
    def compute_hard_label_metrics(self, model, images, lr, hard_labels=None):
        if hard_labels is None:
            hard_labels = torch.tensor([np.ones(self.ipc) * i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False).view(-1)
        hard_label_dataset = TensorDataset(images, hard_labels)
        train_loader = DataLoader(hard_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = StepLR(optimizer, step_size=self.num_epochs // 4, gamma=0.5)

        best_acc1 = 0
        for epoch in range(self.num_epochs):
            train_one_epoch(epoch, model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device, temperature=self.temperature)
            train_one_epoch(epoch, model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device, temperature=self.temperature)
            metric = validate(model, self.test_loader, device=self.device)
            if metric['top1'] > best_acc1:
                best_acc1 = metric['top1']

        return best_acc1
        
    def compute_kl_divergence_metrics(self, model, images, lr, labels=None):
        if labels is None:
            labels = torch.tensor([np.ones(self.ipc) * i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False).view(-1)
        soft_label_dataset = TensorDataset(images, labels)
        train_loader = DataLoader(soft_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = self.KLDivLoss
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = StepLR(optimizer, step_size=self.num_epochs // 4, gamma=0.5)

        best_acc1 = 0
        for epoch in range(self.num_epochs):
            train_one_epoch(epoch, model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device, temperature=self.temperature)
            train_one_epoch(epoch, model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device, temperature=self.temperature)
            metric = validate(model, self.test_loader, device=self.device)
            if metric['top1'] > best_acc1:
                best_acc1 = metric['top1']
        
        return best_acc1
    
    def compute_metrics(self, images, syn_lr, labels=None):

        obj_metrics = []
        for i in range(self.num_eval):
            set_seed()
            print(f"############### {i+1}th Evaluation ###############")
            print(f"############### {i+1}th Evaluation ###############")

            print("Caculating syn data hard label metrics...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            syn_data_hard_label_acc = self.hyper_param_search_for_hard_label(model, images, hard_labels=None)
            del model

            print("Caculating full data hard label metrics...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            full_data_hard_label_acc = self.hyper_param_search_for_hard_label(model, self.images_train, hard_labels=self.labels_train)
            del model

            print("Caculating syn data kl divergence metrics...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            syn_data_kl_divergence_acc = self.compute_kl_divergence_metrics(model, images, lr=syn_lr, labels=labels)
            del model

            print("Caculating random data kl divergence metrics...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            random_images = get_random_images(self.images_train, self.class_indices_train, self.ipc)
            random_data_kl_divergence_acc = self.hyper_param_search_for_kl_divergence(model, random_images, labels=labels)
            del model
            
            print("syn_data_kl_divergence_acc: ", syn_data_kl_divergence_acc)
            print("random_data_kl_divergence_acc: ", random_data_kl_divergence_acc)
            print("full_data_hard_label_acc: ", full_data_hard_label_acc)
            print("syn_data_hard_label_acc: ", syn_data_hard_label_acc)
            
            print("syn_data_kl_divergence_acc: ", syn_data_kl_divergence_acc)
            print("random_data_kl_divergence_acc: ", random_data_kl_divergence_acc)
            print("full_data_hard_label_acc: ", full_data_hard_label_acc)
            print("syn_data_hard_label_acc: ", syn_data_hard_label_acc)

            numerator = 1.00 * (syn_data_kl_divergence_acc - random_data_kl_divergence_acc)
            denominator = 1.00 * (full_data_hard_label_acc - syn_data_hard_label_acc)
            obj_metrics.append(numerator / denominator)
        obj_metrics_mean = np.mean(obj_metrics)
        obj_metrics_std = np.std(obj_metrics)

        print(f"KL Divergence Objective Metrics Mean: {obj_metrics_mean:.2f}  Std: {obj_metrics_std:.2f}")
        print(f"KL Divergence Objective Metrics Mean: {obj_metrics_mean:.2f}  Std: {obj_metrics_std:.2f}")
        return obj_metrics_mean, obj_metrics_std


if __name__ == "__main__":
    images = torch.randn(10, 3, 32, 32)
    obj = DD_Ranking_Objective(dataset="CIFAR10", real_data_path=None, ipc=1, model_name="ConvNet")
    print(obj.compute_metrics())