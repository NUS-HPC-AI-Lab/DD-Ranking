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
from ddranking.utils import build_model, get_pretrained_model_path
from ddranking.utils import TensorDataset, get_random_images, get_dataset, save_results
from ddranking.utils import set_seed, train_one_epoch, validate, get_optimizer, get_lr_scheduler
from ddranking.aug import DSA, Mixup, Cutmix, ZCAWhitening
from ddranking.config import Config


class HardLabelEvaluator:

    def __init__(self, config: Config=None, dataset: str='CIFAR10', real_data_path: str='./dataset/', ipc: int=10, 
                 model_name: str='ConvNet-3', data_aug_func: str='cutmix', aug_params: dict={'cutmix_p': 1.0}, optimizer: str='sgd', 
                 lr_scheduler: str='step', weight_decay: float=0.0005, momentum: float=0.9, use_zca: bool=False, num_eval: int=5, 
                 im_size: tuple=(32, 32), num_epochs: int=300, real_batch_size: int=256, syn_batch_size: int=256, use_torchvision: bool=False,
                 default_lr: float=0.01, num_workers: int=4, save_path: str=None, custom_train_trans=None, custom_val_trans=None, device: str="cuda"):
        
        if config is not None:
            self.config = config
            dataset = self.config.get('dataset')
            real_data_path = self.config.get('real_data_path')
            ipc = self.config.get('ipc')
            model_name = self.config.get('model_name')
            data_aug_func = self.config.get('data_aug_func')
            aug_params = self.config.get('aug_params')
            optimizer = self.config.get('optimizer')
            lr_scheduler = self.config.get('lr_scheduler')
            weight_decay = self.config.get('weight_decay')
            momentum = self.config.get('momentum')
            num_eval = self.config.get('num_eval')
            im_size = self.config.get('im_size')
            num_epochs = self.config.get('num_epochs')
            real_batch_size = self.config.get('real_batch_size')
            syn_batch_size = self.config.get('syn_batch_size')
            default_lr = self.config.get('default_lr')
            save_path = self.config.get('save_path')
            use_zca = self.config.get('use_zca')
            use_torchvision = self.config.get('use_torchvision')
            custom_train_trans = self.config.get('custom_train_trans')
            custom_val_trans = self.config.get('custom_val_trans')
            num_workers = self.config.get('num_workers')
            device = self.config.get('device')

        channel, im_size, num_classes, dst_train, dst_test_real, dst_test_syn, class_map, class_map_inv = get_dataset(dataset, 
                                                                                                                      real_data_path, 
                                                                                                                      im_size, 
                                                                                                                      use_zca,
                                                                                                                      custom_val_trans,
                                                                                                                      device)
        # self.images_train, self.labels_train, self.class_indices_train = self.load_real_data(dst_train, class_map, num_classes)
        self.class_indices = self.get_class_indices(dst_train, class_map, num_classes)
        self.dst_train = dst_train

        self.test_loader_real = DataLoader(dst_test_real, batch_size=real_batch_size, num_workers=num_workers, shuffle=False)
        self.test_loader_syn = DataLoader(dst_test_syn, batch_size=syn_batch_size, num_workers=num_workers, shuffle=False)

        # data info
        self.im_size = im_size
        self.num_classes = num_classes
        self.ipc = ipc
        self.custom_train_trans = custom_train_trans

        # training info
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.num_eval = num_eval
        self.model_name = model_name
        self.real_batch_size = real_batch_size
        self.syn_batch_size = syn_batch_size
        self.num_epochs = num_epochs
        self.default_lr = default_lr
        self.num_workers = num_workers
        self.test_interval = 10
        self.use_torchvision = use_torchvision
        self.device = device

        if data_aug_func == 'dsa':
            self.aug_func = DSA(aug_params)
        elif data_aug_func == 'zca':
            self.aug_func = ZCAWhitening(aug_params)
        elif data_aug_func == 'mixup':
            self.aug_func = Mixup(aug_params)
        elif data_aug_func == 'cutmix':
            self.aug_func = Cutmix(aug_params)
        else:
            self.aug_func = None

        if not save_path:
            save_path = f"./results/{dataset}/{model_name}/ipc{ipc}/obj_scores.csv"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        self.save_path = save_path

    def load_real_data(self, dataset, class_map, num_classes):
        images_all = []
        labels_all = []
        class_indices = [[] for c in range(num_classes)]
        for i, (image, label) in enumerate(dataset):
            if torch.is_tensor(label):
                label = label.item()
            images_all.append(torch.unsqueeze(image, 0))
            labels_all.append(class_map[label])
        images_all = torch.cat(images_all, dim=0)
        labels_all = torch.tensor(labels_all)
        for i, label in enumerate(labels_all):
            class_indices[label].append(i)
        
        return images_all, labels_all, class_indices
    
    def get_class_indices(self, dataset, class_map, num_classes):
        class_indices = [[] for c in range(num_classes)]
        for i, (_, label) in enumerate(dataset):
            if torch.is_tensor(label):
                label = label.item()
            true_label = class_map[label]
            class_indices[true_label].append(i)
        return class_indices
    
    def hyper_param_search_for_hard_label(self, image_tensor, image_path, hard_labels, mode='real'):
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
                use_torchvision=self.use_torchvision,
                device=self.device
            )
            acc = self.compute_hard_label_metrics(
                model=model, 
                image_tensor=image_tensor,
                image_path=image_path,
                lr=lr,
                hard_labels=hard_labels,
                mode=mode
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr

    def compute_hard_label_metrics(self, model, image_tensor, image_path, lr, hard_labels, mode='real'):
        if mode == 'real':
            hard_label_dataset = self.dst_train
        else:
            if image_tensor is None:
                hard_label_dataset = datasets.ImageFolder(root=image_path, transform=self.custom_train_trans)
            else:
                hard_label_dataset = TensorDataset(image_tensor, hard_labels)
        train_loader = DataLoader(hard_label_dataset, batch_size=self.real_batch_size if mode == 'real' else self.syn_batch_size, 
                                  num_workers=self.num_workers, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = get_optimizer(self.optimizer, model, lr, self.weight_decay, self.momentum)
        lr_scheduler = get_lr_scheduler(self.lr_scheduler, optimizer, self.num_epochs)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_one_epoch(
                epoch=epoch, 
                stu_model=model, 
                loader=train_loader, 
                loss_fn=loss_fn, 
                optimizer=optimizer,
                aug_func=self.aug_func,
                lr_scheduler=lr_scheduler, 
                device=self.device
            )
            if epoch > 0.8 * self.num_epochs and (epoch + 1) % self.test_interval == 0:
                metric = validate(
                    model=model, 
                    loader=self.test_loader,
                    device=self.device
                )
                if metric['top1'] > best_acc1:
                    best_acc1 = metric['top1']

        return best_acc1
    
    def compute_metrics(self, image_tensor: Tensor=None, image_path: str=None, hard_labels: Tensor=None, syn_lr: float=None):
        if image_tensor is None and image_path is None:
            raise ValueError("Either image_tensor or image_path must be provided")

        if not hard_labels:
            hard_labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), dtype=torch.long, requires_grad=False).view(-1)

        hard_label_recovery = []
        improvement_over_random = []
        for i in range(self.num_eval):
            set_seed()
            print(f"########################### {i+1}th Evaluation ###########################")

            print("Caculating syn data hard label metrics...")
            if syn_lr:
                model = build_model(
                    model_name=self.model_name, 
                    num_classes=self.num_classes, 
                    im_size=self.im_size, 
                    pretrained=False,
                    use_torchvision=self.use_torchvision,
                    device=self.device
                )
                syn_data_hard_label_acc = self.compute_hard_label_metrics(
                    model=model, 
                    image_tensor=image_tensor,
                    image_path=image_path,
                    lr=syn_lr, 
                    hard_labels=hard_labels,
                    mode='syn'
                )
                del model
            else:
                syn_data_hard_label_acc, best_lr = self.hyper_param_search_for_hard_label(
                    image_tensor=image_tensor,
                    image_path=image_path,
                    hard_labels=hard_labels,
                    mode='syn'
                )
            print(f"Syn data hard label acc: {syn_data_hard_label_acc:.2f}%")

            print("Caculating full data hard label metrics...")
            model = build_model(
                model_name=self.model_name,
                num_classes=self.num_classes, 
                im_size=self.im_size,
                pretrained=False,
                use_torchvision=self.use_torchvision,
                device=self.device
            )
            full_data_hard_label_acc = self.compute_hard_label_metrics(
                model=model, 
                image_tensor=self.images_train,
                image_path=None,
                lr=self.default_lr, 
                hard_labels=self.labels_train,
                mode='real'
            )
            del model
            print(f"Full data hard label acc: {full_data_hard_label_acc:.2f}%")

            print("Caculating random data hard label metrics...")
            random_images, random_data_hard_labels = get_random_images(self.images_train, self.labels_train, self.class_indices_train, self.ipc)
            random_data_hard_label_acc, best_lr = self.hyper_param_search_for_hard_label(
                image_tensor=random_images,
                image_path=None,
                hard_labels=random_data_hard_labels,
                mode='syn'
            )
            print(f"Random data hard label acc: {random_data_hard_label_acc:.2f}%")

            hlr = 1.00 * (full_data_hard_label_acc - syn_data_hard_label_acc)
            ior = 1.00 * (syn_data_hard_label_acc - random_data_hard_label_acc)

            hard_label_recovery.append(hlr)
            improvement_over_random.append(ior)
        
        results_to_save = {
            "hard_label_recovery": hard_label_recovery,
            "improvement_over_random": improvement_over_random
        }
        save_results(results_to_save, self.save_path)

        hard_label_recovery_mean = np.mean(hard_label_recovery)
        hard_label_recovery_std = np.std(hard_label_recovery)
        improvement_over_random_mean = np.mean(improvement_over_random)
        improvement_over_random_std = np.std(improvement_over_random)

        print(f"Hard Label Recovery Mean: {hard_label_recovery_mean:.2f}%  Std: {hard_label_recovery_std:.2f}")
        print(f"Improvement Over Random Mean: {improvement_over_random_mean:.2f}%  Std: {improvement_over_random_std:.2f}")
        return {
            "hard_label_recovery_mean": hard_label_recovery_mean,
            "hard_label_recovery_std": hard_label_recovery_std,
            "improvement_over_random_mean": improvement_over_random_mean,
            "improvement_over_random_std": improvement_over_random_std
        }
