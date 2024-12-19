import os
import time
import torch
import random
import numpy as np
import torch.nn.functional as F
from typing import List
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchvision import transforms, datasets
from dd_ranking.utils import build_model, get_pretrained_model_path
from dd_ranking.utils import TensorDataset, get_random_images, get_dataset
from dd_ranking.utils import set_seed
from dd_ranking.utils import train_one_epoch, train_one_epoch_dc, validate, validate_dc
from dd_ranking.loss import SoftCrossEntropyLoss, KLDivergenceLoss
from dd_ranking.aug import DSA_Augmentation, Mixup_Augmentation, Cutmix_Augmentation, ZCA_Whitening_Augmentation


class Unified_Evaluator:

    def __init__(self, 
        dataset: str, 
        real_data_path: str, 
        ipc: int,
        model_name: str,
        soft_label_mode: str,
        criterion: str,
        aug_name: str,
        func_names: list=None,
        params: dict=None,
        num_eval: int=5,
        im_size: tuple=(32, 32), 
        num_epochs: int=300, 
        batch_size: int=256, 
        device: str="cuda"
    ):

        channel, im_size, num_classes, dst_train, dst_test, class_map, class_map_inv = get_dataset(dataset, real_data_path, im_size)
        self.num_classes = num_classes
        self.im_size = im_size
        self.test_loader = DataLoader(dst_test, batch_size=batch_size, shuffle=False)

        self.ipc = ipc
        self.model_name = model_name
        self.soft_label_mode = soft_label_mode
        self.criterion = criterion
        
        self.num_eval = num_eval
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device

        pretrained_model_path = get_pretrained_model_path(model_name, dataset, ipc)
        self.teacher_model = build_model(
            model_name=model_name, 
            num_classes=num_classes, 
            im_size=self.im_size, 
            pretrained=True, 
            device=self.device, 
            model_path=pretrained_model_path
        )
        self.teacher_model.eval()

        if aug_name == 'None':
            self.aug_func = None
        elif aug_name == 'DSA':
            self.aug_func = DSA_Augmentation(func_names, params)
        elif aug_name == 'ZCA':
            self.aug_func = ZCA_Whitening_Augmentation(params)
        elif aug_name == 'Mixup':
            self.aug_func = Mixup_Augmentation(params)
        elif aug_name == 'Cutmix':
            self.aug_func = Cutmix_Augmentation(params)
    
    @staticmethod
    def SoftCrossEntropyLoss(inputs, target):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch_size = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch_size
        return loss
    
    @staticmethod
    def KLDivLoss(stu_outputs, tea_outputs, temperature=1.0):
        kl = torch.nn.KLDivLoss(reduction='batchmean')
        stu_probs = F.log_softmax(stu_outputs / temperature, dim=1)
        tea_probs = F.softmax(tea_outputs / temperature, dim=1)
        loss = kl(stu_probs, tea_probs) * (temperature ** 2)
        return loss

    def generate_soft_labels(self, images):
        batches = torch.split(images, self.batch_size)
        soft_labels = []
        with torch.no_grad():
            for image_batch in batches:
                image_batch = image_batch.to(self.device)
                soft_labels.append(self.teacher_model(image_batch).detach().cpu())
        soft_labels = torch.cat(soft_labels, dim=0)
        return soft_labels

    def hyper_param_search(self, loader):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        best_acc = 0
        best_lr = None
        for lr in lr_list:
            print(f"Searching lr: {lr}")
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False, 
                device=self.device
            )
            train_one_epoch(
                model=model, 
                loader=loader, 
                lr=lr,
                num_epochs=self.num_epochs, 
                device=self.device
            )
            acc = validate(
                model=model, 
                loader=loader, 
                device=self.device
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
        return best_acc, best_lr
    
    def compute_metrics_helper(self, model, loader, lr):
        if self.criterion == 'CE':
            loss_fn = CrossEntropyLoss()
        elif self.criterion == 'KL':
            loss_fn = KLDivLoss()
        elif self.criterion == 'SCE':
            loss_fn = SoftCrossEntropyLoss()
        
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=self.num_epochs//2, gamma=0.1)
        
        best_acc = 0
        for epoch in range(self.num_epochs):
            train_one_epoch(
                model=model, 
                loader=loader, 
                optimizer=optimizer, 
                loss_fn=loss_fn,
                soft_label_mode=self.soft_label_mode,
                aug_func=self.aug_func.apply_augmentation,
                lr_scheduler=scheduler, 
                device=self.device
            )
            acc = validate(
                model=model, 
                loader=loader, 
                device=self.device
            )
            if acc > best_acc:
                best_acc = acc
        return best_acc
        
    def compute_metrics(self, images, labels, syn_lr=None):
        syn_dataset = TensorDataset(images, labels)
        syn_loader = DataLoader(syn_dataset, batch_size=self.batch_size, shuffle=True)

        accs = []
        for i in range(self.num_eval):
            print(f"########################### {i+1}th Evaluation ###########################")
            if syn_lr:
                model = build_model(
                    model_name=self.model_name, 
                    num_classes=self.num_classes, 
                    im_size=self.im_size, 
                    pretrained=False, 
                    device=self.device
                )
                syn_data_acc = self.compute_metrics_helper(
                    model=model, 
                    loader=syn_loader, 
                    lr=syn_lr
                )
                del model
            else:
                syn_data_acc, best_lr = self.hyper_param_search(syn_loader)
            accs.append(syn_data_acc)
        
        accs_mean = np.mean(accs)
        accs_std = np.std(accs)
        return accs_mean, accs_std
