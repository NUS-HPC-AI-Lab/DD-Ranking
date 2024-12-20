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
from dd_ranking.utils import build_model, get_pretrained_model_path
from dd_ranking.utils import TensorDataset, get_random_images, get_dataset
from dd_ranking.utils import set_seed, save_results
from dd_ranking.utils import train_one_epoch, train_one_epoch_dc, validate, validate_dc
from dd_ranking.loss import SoftCrossEntropyLoss, KLDivergenceLoss


class Soft_Label_Objective_Metrics:

    def __init__(self, dataset: str, real_data_path: str, ipc: int, model_name: str, soft_label_mode: str='S',
                 num_eval: int=5, im_size: tuple=(32, 32), num_epochs: int=300, batch_size: int=256, save_path: str=None, device: str="cuda"):

        channel, im_size, num_classes, dst_train, dst_test, class_map, class_map_inv = get_dataset(dataset, real_data_path, im_size)
        self.images_train, self.labels_train, self.class_indices_train = self.load_real_data(dst_train, class_map, num_classes)
        self.test_loader = DataLoader(dst_test, batch_size=batch_size, shuffle=False)

        self.soft_label_mode = soft_label_mode
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

        if not save_path:
            save_path = f"./results/{dataset}/{model_name}/ipc{ipc}/obj_scores.csv"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        self.save_path = save_path

        # teacher model
        pretrained_model_path = get_pretrained_model_path(model_name, dataset, ipc)
        self.teacher_model = build_model(model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=True, device=self.device, model_path=pretrained_model_path)
        self.teacher_model.eval()

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


class SCE_Objective_Metrics(Soft_Label_Objective_Metrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def SoftCrossEntropy(inputs, target):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch_size = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch_size
        return loss

    def hyper_param_search_for_hard_label(self, images, hard_labels):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            print(f"Searching lr:{lr} for hard label...")
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False, 
                device=self.device
            )
            acc = self.compute_hard_label_metrics(
                model=model, 
                images=images, 
                lr=lr, 
                hard_labels=hard_labels
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr

    def hyper_param_search_for_soft_label(self, images, soft_labels):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            print(f"Searching lr:{lr} for soft label...")
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False, 
                device=self.device
            )
            acc = self.compute_soft_label_metrics(
                model=model, 
                images=images, 
                lr=lr, 
                soft_labels=soft_labels
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr
    
    def compute_hard_label_metrics(self, model, images, lr, hard_labels):
        
        hard_label_dataset = TensorDataset(images, hard_labels)
        train_loader = DataLoader(hard_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = StepLR(optimizer, step_size=self.num_epochs // 2 + 1, gamma=0.1)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_one_epoch(
                epoch=epoch, 
                stu_model=model, 
                loader=train_loader, 
                loss_fn=loss_fn, 
                optimizer=optimizer, 
                lr_scheduler=lr_scheduler, 
                tea_model=self.teacher_model, 
                device=self.device
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
        
    def compute_soft_label_metrics(self, model, images, lr, soft_labels):

        soft_label_dataset = TensorDataset(images, soft_labels)
        train_loader = DataLoader(soft_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = SoftCrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = StepLR(optimizer, step_size=self.num_epochs // 2 + 1, gamma=0.1)
        
        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_one_epoch(
                epoch=epoch, 
                stu_model=model, 
                loader=train_loader, 
                loss_fn=loss_fn, 
                optimizer=optimizer, 
                soft_label_mode=self.soft_label_mode, 
                lr_scheduler=lr_scheduler, 
                tea_model=self.teacher_model, 
                device=self.device
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

    def generate_soft_labels(self, images):
        batches = torch.split(images, self.batch_size)
        soft_labels = []
        with torch.no_grad():
            for image_batch in batches:
                image_batch = image_batch.to(self.device)
                soft_labels.append(self.teacher_model(image_batch).detach().cpu())
        soft_labels = torch.cat(soft_labels, dim=0)
        return soft_labels
    
    def compute_metrics(self, syn_images, soft_labels=None, hard_labels=None, syn_lr=None):
        if soft_labels is None:
            soft_labels = self.generate_soft_labels(syn_images)
        if hard_labels is None:
            hard_labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), dtype=torch.long, requires_grad=False).view(-1)

        obj_metrics = []
        hard_recs = []
        soft_imps = []
        for i in range(self.num_eval):
            set_seed()
            print(f"########################### {i+1}th Evaluation ###########################")

            print("Caculating syn data hard label metrics...")
            syn_data_hard_label_acc, best_lr = self.hyper_param_search_for_hard_label(
                images=syn_images, 
                hard_labels=hard_labels
            )
            print(f"Syn data hard label acc: {syn_data_hard_label_acc:.2f}%")

            print("Caculating full data hard label metrics...")
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False, 
                device=self.device
            )
            full_data_hard_label_acc = self.compute_hard_label_metrics(
                model=model, 
                images=self.images_train, 
                lr=self.default_lr, 
                hard_labels=self.labels_train
            )
            del model
            print(f"Full data hard label acc: {full_data_hard_label_acc:.2f}%")

            print("Caculating syn data soft label metrics...")
            if syn_lr:
                model = build_model(
                    model_name=self.model_name, 
                    num_classes=self.num_classes, 
                    im_size=self.im_size, 
                    pretrained=False, 
                    device=self.device
                )
                syn_data_soft_label_acc = self.compute_soft_label_metrics(
                    model=model, 
                    images=syn_images, 
                    lr=syn_lr, 
                    soft_labels=soft_labels
                )
                del model
            else:
                syn_data_soft_label_acc, best_lr = self.hyper_param_search_for_soft_label(
                    images=syn_images, 
                    soft_labels=soft_labels
                )
            
            print(f"Syn data soft label acc: {syn_data_soft_label_acc:.2f}%")

            print("Caculating random data soft label metrics...")
            random_images, _ = get_random_images(self.images_train, self.labels_train, self.class_indices_train, self.ipc)
            random_data_soft_labels = self.generate_soft_labels(random_images)
            random_data_soft_label_acc, best_lr = self.hyper_param_search_for_soft_label(
                images=random_images, 
                soft_labels=random_data_soft_labels
            )
            print(f"Random data soft label acc: {random_data_soft_label_acc:.2f}%")

            hard_rec = 1.00 * (full_data_hard_label_acc - syn_data_hard_label_acc)
            soft_imp = 1.00 * (syn_data_soft_label_acc - random_data_soft_label_acc)

            hard_recs.append(hard_rec)
            soft_imps.append(soft_imp)
            obj_metrics.append(soft_imp / hard_rec)
        
        results_to_save = {
            "hard_recs": hard_recs,
            "soft_imps": soft_imps,
            "obj_metrics": obj_metrics
        }
        save_results(results_to_save, self.save_path)

        hard_recs_mean = np.mean(hard_recs)
        hard_recs_std = np.std(hard_recs)
        soft_imps_mean = np.mean(soft_imps)
        soft_imps_std = np.std(soft_imps)
        obj_metrics_mean = np.mean(obj_metrics)
        obj_metrics_std = np.std(obj_metrics)

        print(f"SCE Hard Recovery Mean: {hard_recs_mean:.2f}%  Std: {hard_recs_std:.2f}")
        print(f"SCE Soft Improvement Mean: {soft_imps_mean:.2f}%  Std: {soft_imps_std:.2f}")
        print(f"SCE Objective Metrics Mean: {obj_metrics_mean:.2f}%  Std: {obj_metrics_std:.2f}")
        return {
            "hard_recs_mean": hard_recs_mean,
            "hard_recs_std": hard_recs_std,
            "soft_imps_mean": soft_imps_mean,
            "soft_imps_std": soft_imps_std,
            "obj_metrics_mean": obj_metrics_mean,
            "obj_metrics_std": obj_metrics_std
        }


class KL_Objective_Metrics(Soft_Label_Objective_Metrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
    
    def hyper_param_search_for_hard_label(self, images, hard_labels):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]

        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            print(f"Searching {lr} for hard label...")
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False, 
                device=self.device
            )
            acc = self.compute_hard_label_metrics(
                model=model, 
                images=images, 
                lr=lr, 
                hard_labels=hard_labels
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
        return best_acc, best_lr
    
    def hyper_param_search_for_kl_divergence(self, images, soft_labels):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            print(f"Searching {lr} for kl divergence...")
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False, 
                device=self.device
            )
            acc = self.compute_kl_divergence_metrics(
                model=model, 
                images=images, 
                lr=lr, 
                soft_labels=soft_labels
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr

    def compute_hard_label_metrics(self, model, images, lr, hard_labels):
        hard_label_dataset = TensorDataset(images, hard_labels)
        train_loader = DataLoader(hard_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = StepLR(optimizer, step_size=self.num_epochs // 2 + 1, gamma=0.1)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_one_epoch(
                epoch=epoch, 
                stu_model=model, 
                loader=train_loader, 
                loss_fn=loss_fn, 
                optimizer=optimizer, 
                soft_label_mode=self.soft_label_mode, 
                lr_scheduler=lr_scheduler, 
                tea_model=self.teacher_model, 
                device=self.device
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

    def compute_kl_divergence_metrics(self, model, images, lr, soft_labels):

        soft_label_dataset = TensorDataset(images, soft_labels)
        train_loader = DataLoader(soft_label_dataset, batch_size=self.batch_size, shuffle=True)

        loss_fn = KLDivergenceLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = StepLR(optimizer, step_size=self.num_epochs // 2 + 1, gamma=0.1)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_one_epoch(
                epoch=epoch, 
                stu_model=model, 
                loader=train_loader, 
                loss_fn=loss_fn, 
                optimizer=optimizer, 
                soft_label_mode=self.soft_label_mode, 
                lr_scheduler=lr_scheduler, 
                tea_model=self.teacher_model, 
                device=self.device
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
    
    def compute_metrics(self, images, soft_labels=None, hard_labels=None):
        if soft_labels is None:
            soft_labels = self.generate_soft_labels(syn_images)
        if hard_labels is None:
            hard_labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), dtype=torch.long, requires_grad=False).view(-1)

        hard_recs = []
        soft_imps = []
        obj_metrics = []
        for i in range(self.num_eval):
            set_seed()
            print(f"########################### {i+1}th Evaluation ###########################")

            print("Caculating syn data hard label metrics...")
            syn_data_hard_label_acc, best_lr = self.hyper_param_search_for_hard_label(
                images=images, 
                hard_labels=hard_labels
            )
            print(f"Syn data hard label acc: {syn_data_hard_label_acc:.2f}%")

            print("Caculating full data hard label metrics...")
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False, 
                device=self.device
            )
            full_data_hard_label_acc = self.compute_hard_label_metrics(
                model=model, 
                images=self.images_train, 
                lr=self.default_lr, 
                hard_labels=self.labels_train
            )
            del model
            print(f"Full data hard label acc: {full_data_hard_label_acc:.2f}%")

            print("Caculating syn data kl divergence metrics...")
            syn_data_kl_divergence_acc, best_lr = self.hyper_param_search_for_kl_divergence(
                images=images, 
                soft_labels=soft_labels
            )
            print(f"Syn data kl divergence acc: {syn_data_kl_divergence_acc:.2f}%")

            print("Caculating random data kl divergence metrics...")
            random_images, _ = get_random_images(
                images_all=self.images_train, 
                labels_all=self.labels_train, 
                class_indices=self.class_indices_train, 
                n_images_per_class=self.ipc
            )
            random_data_soft_labels = self.generate_soft_labels(random_images)
            random_data_kl_divergence_acc, best_lr = self.hyper_param_search_for_kl_divergence(
                images=random_images, 
                soft_labels=random_data_soft_labels
            )
            print(f"Random data kl divergence acc: {random_data_kl_divergence_acc:.2f}%")

            hard_rec = 1.00 * (full_data_hard_label_acc - syn_data_hard_label_acc)
            soft_imp = 1.00 * (syn_data_kl_divergence_acc - random_data_kl_divergence_acc)
            obj_metrics.append(soft_imp / hard_rec)

            hard_recs.append(hard_rec)
            soft_imps.append(soft_imp)

        results_to_save = {
            "hard_recs": hard_recs,
            "soft_imps": soft_imps,
            "obj_metrics": obj_metrics
        }
        save_results(results_to_save, self.save_path)

        hard_recs_mean = np.mean(hard_recs)
        hard_recs_std = np.std(hard_recs)
        soft_imps_mean = np.mean(soft_imps)
        soft_imps_std = np.std(soft_imps)
        obj_metrics_mean = np.mean(obj_metrics)
        obj_metrics_std = np.std(obj_metrics)

        print(f"KL Divergence Hard Recovery Mean: {hard_recs_mean:.2f}  Std: {hard_recs_std:.2f}")
        print(f"KL Divergence Soft Improvement Mean: {soft_imps_mean:.2f}  Std: {soft_imps_std:.2f}")
        print(f"KL Divergence Objective Metrics Mean: {obj_metrics_mean:.2f}  Std: {obj_metrics_std:.2f}")
        return {
            "hard_recs_mean": hard_recs_mean,
            "hard_recs_std": hard_recs_std,
            "soft_imps_mean": soft_imps_mean,
            "soft_imps_std": soft_imps_std,
            "obj_metrics_mean": obj_metrics_mean,
            "obj_metrics_std": obj_metrics_std
        }
