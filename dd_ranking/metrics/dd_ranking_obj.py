import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from dd_ranking.utils.utils import build_model, get_pretrained_model_path
from dd_ranking.utils.utils import TensorDataset, get_random_images, get_dataset
from dd_ranking.utils.utils import set_seed
from dd_ranking.utils.utils import train_one_epoch, train_one_epoch_dc, validate, validate_dc


class DD_Ranking_Objective:

    def __init__(self, dataset: str, real_data_path: str, ipc: int, model_name: str, 
                 num_eval: int=5, im_size: tuple=(32, 32), num_epochs: int=300, batch_size: int=256, device: str="cuda"):

        channel, im_size, num_classes, dst_train, dst_test, class_map, class_map_inv = get_dataset(dataset, real_data_path, im_size)
        self.images_train, self.labels_train, self.class_indices_train = self.load_real_data(dst_train, class_map, num_classes)
        self.test_loader = DataLoader(dst_test, batch_size=batch_size, shuffle=False)

        # data info
        self.im_size = im_size
        self.num_classes = num_classes
        self.ipc = ipc

        # training info
        self.num_eval = num_eval
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.default_lr = 0.01
        self.test_interval = 10
        self.device = device
    
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
    
    def compute_metrics(self):
        pass


class Soft_Label_Objective(DD_Ranking_Objective):
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

    def hyper_param_search_for_hard_label(self, images, hard_labels=None):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            print(f"Searching lr:{lr} for hard label...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            acc = self.compute_hard_label_metrics(model, images, lr, hard_labels=hard_labels)
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr

    def hyper_param_search_for_soft_label(self, images, soft_labels=None):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            print(f"Searching lr:{lr} for soft label...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            acc = self.compute_soft_label_metrics(model, images, lr, soft_labels=soft_labels)
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr
    
    def compute_hard_label_metrics(self, model, images, lr, hard_labels=None):
        if hard_labels is None:
            hard_labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), dtype=torch.long, requires_grad=False).view(-1)
        hard_label_dataset = TensorDataset(images, hard_labels)
        train_loader = DataLoader(hard_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = StepLR(optimizer, step_size=self.num_epochs // 2 + 1, gamma=0.1)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_one_epoch(epoch, model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device)
            if epoch % self.test_interval == 0:
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
        lr_scheduler = StepLR(optimizer, step_size=self.num_epochs // 2 + 1, gamma=0.1)
        
        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_one_epoch(epoch, model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device)
            if epoch % self.test_interval == 0:
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
    
    def compute_metrics(self, syn_images, soft_labels=None, hard_labels=None, syn_lr=None):

        obj_metrics = []
        for i in range(self.num_eval):
            set_seed()
            print(f"########################### {i+1}th Evaluation ###########################")

            print("Caculating syn data hard label metrics...")
            syn_data_hard_label_acc, best_lr = self.hyper_param_search_for_hard_label(syn_images, hard_labels=hard_labels)
            print(f"Syn data hard label acc: {syn_data_hard_label_acc * 100:.2f}%")

            print("Caculating full data hard label metrics...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            full_data_hard_label_acc = self.compute_hard_label_metrics(model, self.images_train, lr=self.default_lr, hard_labels=self.labels_train)
            del model
            print(f"Full data hard label acc: {full_data_hard_label_acc * 100:.2f}%")

            print("Caculating syn data soft label metrics...")
            if syn_lr:
                model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
                syn_data_soft_label_acc = self.compute_soft_label_metrics(model, syn_images, lr=syn_lr, soft_labels=soft_labels)
                del model
            else:
                syn_data_soft_label_acc, best_lr = self.hyper_param_search_for_soft_label(syn_images, soft_labels=soft_labels)
            
            print(f"Syn data soft label acc: {syn_data_soft_label_acc * 100:.2f}%")

            print("Caculating random data soft label metrics...")
            random_images, random_labels = get_random_images(self.images_train, self.labels_train, self.class_indices_train, self.ipc)
            random_data_soft_label_acc, best_lr = self.hyper_param_search_for_soft_label(random_images, soft_labels=None)
            print(f"Random data soft label acc: {random_data_soft_label_acc * 100:.2f}%")

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
        self.temperature = temperature

        pretrained_model_path = get_pretrained_model_path(tea_model_name, dataset, ipc)
        self.teacher_model = build_model(tea_model_name, num_classes=self.num_classes, im_size=self.im_size, 
                                         pretrained=True, device=self.device, model_path=pretrained_model_path)

    @staticmethod
    def KLDivLoss(stu_outputs, tea_outputs, temperature=1.0):
        kl = torch.nn.KLDivLoss(reduction='batchmean')
        stu_probs = F.log_softmax(stu_outputs / temperature, dim=1)
        tea_probs = F.softmax(tea_outputs / temperature, dim=1)
        loss = kl(stu_probs, tea_probs)
        return loss
    
    def hyper_param_search_for_hard_label(self, images, hard_labels=None):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]

        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            print(f"Searching {lr} for hard label...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            acc = self.compute_hard_label_metrics(model, images, lr, hard_labels=hard_labels)
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
        return best_acc, best_lr
    
    def hyper_param_search_for_kl_divergence(self, images, labels=None):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            print(f"Searching {lr} for kl divergence...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            acc = self.compute_kl_divergence_metrics(model, images, lr, labels=labels)
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr

    def compute_hard_label_metrics(self, model, images, lr, hard_labels=None):
        if hard_labels is None:
            hard_labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), dtype=torch.long, requires_grad=False).view(-1)
        hard_label_dataset = TensorDataset(images, hard_labels)
        train_loader = DataLoader(hard_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = StepLR(optimizer, step_size=self.num_epochs // 2 + 1, gamma=0.1)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_one_epoch(epoch, model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device, temperature=self.temperature)
            if epoch % self.test_interval == 0:
                metric = validate(model, self.test_loader, device=self.device)
                if metric['top1'] > best_acc1:
                    best_acc1 = metric['top1']

        return best_acc1

    def compute_kl_divergence_metrics(self, model, images, lr, labels=None):
        if labels is None:
            labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), dtype=torch.long, requires_grad=False).view(-1)
        soft_label_dataset = TensorDataset(images, labels)
        train_loader = DataLoader(soft_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = self.KLDivLoss
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = StepLR(optimizer, step_size=self.num_epochs // 2 + 1, gamma=0.1)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_one_epoch(epoch, model, train_loader, loss_fn, optimizer, tea_model=self.teacher_model, lr_scheduler=lr_scheduler, device=self.device, temperature=self.temperature)
            if epoch % self.test_interval == 0:
                metric = validate(model, self.test_loader, device=self.device)
                if metric['top1'] > best_acc1:
                    best_acc1 = metric['top1']
        
        return best_acc1
    
    def compute_metrics(self, images, hard_labels=None, syn_lr=None):

        obj_metrics = []
        for i in range(self.num_eval):
            set_seed()
            print(f"########################### {i+1}th Evaluation ###########################")

            print("Caculating syn data hard label metrics...")
            syn_data_hard_label_acc, best_lr = self.hyper_param_search_for_hard_label(images, hard_labels=hard_labels)
            print(f"Syn data hard label acc: {syn_data_hard_label_acc * 100:.2f}%")

            print("Caculating full data hard label metrics...")
            model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
            full_data_hard_label_acc = self.compute_hard_label_metrics(model, self.images_train, lr=self.default_lr, hard_labels=self.labels_train)
            del model
            print(f"Full data hard label acc: {full_data_hard_label_acc * 100:.2f}%")

            print("Caculating syn data kl divergence metrics...")
            if syn_lr:
                model = build_model(self.model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
                syn_data_kl_divergence_acc = self.compute_kl_divergence_metrics(model, images, lr=syn_lr, labels=hard_labels)
                del model
            else:
                syn_data_kl_divergence_acc, best_lr = self.hyper_param_search_for_kl_divergence(images, labels=hard_labels)
            print(f"Syn data kl divergence acc: {syn_data_kl_divergence_acc * 100:.2f}%")

            print("Caculating random data kl divergence metrics...")
            random_images, random_labels = get_random_images(self.images_train, self.labels_train, self.class_indices_train, self.ipc)
            random_data_kl_divergence_acc, best_lr = self.hyper_param_search_for_kl_divergence(random_images, labels=random_labels)
            print(f"Random data kl divergence acc: {random_data_kl_divergence_acc * 100:.2f}%")

            numerator = 1.00 * (syn_data_kl_divergence_acc - random_data_kl_divergence_acc)
            denominator = 1.00 * (full_data_hard_label_acc - syn_data_hard_label_acc)
            obj_metrics.append(numerator / denominator)
        obj_metrics_mean = np.mean(obj_metrics)
        obj_metrics_std = np.std(obj_metrics)

        print(f"KL Divergence Objective Metrics Mean: {obj_metrics_mean:.2f}  Std: {obj_metrics_std:.2f}")
        return obj_metrics_mean, obj_metrics_std


if __name__ == "__main__":
    images = torch.randn(10, 3, 32, 32)
    obj = DD_Ranking_Objective(dataset="CIFAR10", real_data_path=None, ipc=1, model_name="ConvNet")
    print(obj.compute_metrics())