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
from ddranking.utils import TensorDataset, get_random_data_tensors, get_random_data_path_from_cifar, get_random_data_path, get_dataset, save_results, setup_dist
from ddranking.utils import set_seed, train_one_epoch, validate, get_optimizer, get_lr_scheduler
from ddranking.aug import DSA, Mixup, Cutmix, ZCAWhitening
from ddranking.config import Config
from ddranking.utils import logging, broadcast_string
from ddranking.utils import REAL_DATA_ACC_CACHE, REAL_DATA_TRAINING_CONFIG


class LabelRobustScoreHard:

    def __init__(self, config: Config=None, dataset: str='CIFAR10', real_data_path: str='./dataset/', ipc: int=10, model_name: str='ConvNet-3', 
                 data_aug_func: str='cutmix', aug_params: dict={'cutmix_p': 1.0}, optimizer: str='sgd', lr_scheduler: str='step', 
                 step_size: int=None, weight_decay: float=0.0005, momentum: float=0.9, use_zca: bool=False, num_eval: int=5, 
                 im_size: tuple=(32, 32), num_epochs: int=300, real_batch_size: int=256, syn_batch_size: int=256, use_torchvision: bool=False,
                 num_workers: int=4, save_path: str=None, custom_train_trans=None, custom_val_trans=None, device: str="cuda", 
                 dist: bool=False, random_data_format: str='tensor', random_data_path: str=None, eval_full_data: bool=False):
        
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
            step_size = self.config.get('step_size')
            num_eval = self.config.get('num_eval')
            im_size = self.config.get('im_size')
            num_epochs = self.config.get('num_epochs')
            real_batch_size = self.config.get('real_batch_size')
            syn_batch_size = self.config.get('syn_batch_size')
            save_path = self.config.get('save_path')
            use_zca = self.config.get('use_zca')
            use_torchvision = self.config.get('use_torchvision')
            custom_train_trans = self.config.get('custom_train_trans')
            custom_val_trans = self.config.get('custom_val_trans')
            num_workers = self.config.get('num_workers')
            device = self.config.get('device')
            dist = self.config.get('dist', False)
            random_data_format = self.config.get('random_data_format', 'tensors')
            random_data_path = self.config.get('random_data_path')
            eval_full_data = self.config.get('eval_full_data', False)

        self.use_dist = dist

        if self.use_dist:
            setup_dist(self)
        else:
            self.rank = 0
            self.gpu = 0
            self.world_size = 1

        real_batch_size = real_batch_size // self.world_size
        syn_batch_size = syn_batch_size // self.world_size
    
        self.device = device

        channel, im_size, mean, std, num_classes, dst_train, dst_test_real, dst_test_syn, class_map, class_map_inv = get_dataset(
            dataset, 
            real_data_path, 
            im_size, 
            use_zca,
            custom_val_trans,
            device
        )
        self.dataset = dataset
        self.class_map = class_map
        self.class_to_indices = self._get_class_indices(dst_train, class_map, num_classes)
        if dataset not in ['CIFAR10', 'CIFAR100']:
            self.class_to_samples = self._get_class_to_samples(dst_train, class_map, num_classes)
        self.dst_train = dst_train

        if self.use_dist:
            test_sampler_real = torch.utils.data.distributed.DistributedSampler(dst_test_real)
            test_sampler_syn = torch.utils.data.distributed.DistributedSampler(dst_test_syn)
        else:
            test_sampler_real = torch.utils.data.RandomSampler(dst_test_real)
            test_sampler_syn = torch.utils.data.RandomSampler(dst_test_syn)

        self.test_loader_real = DataLoader(dst_test_real, batch_size=real_batch_size, num_workers=num_workers, sampler=test_sampler_real)
        self.test_loader_syn = DataLoader(dst_test_syn, batch_size=syn_batch_size, num_workers=num_workers, sampler=test_sampler_syn)

        # data info
        self.im_size = im_size
        self.num_classes = num_classes
        self.ipc = ipc
        self.custom_train_trans = custom_train_trans
        self.random_data_format = random_data_format
        self.random_data_path = random_data_path

        # training info
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.step_size = step_size
        self.num_eval = num_eval
        self.model_name = model_name
        self.real_batch_size = real_batch_size
        self.syn_batch_size = syn_batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.test_interval = 10
        self.use_torchvision = use_torchvision
        self.device = device
        self.eval_full_data = eval_full_data
        
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
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.save_path = save_path
    
    def _get_class_indices(self, dataset, class_map, num_classes):
        class_indices = [[] for c in range(num_classes)]
        for i, (_, label) in enumerate(dataset):
            if torch.is_tensor(label):
                label = label.item()
            true_label = class_map[label]
            class_indices[true_label].append(i)
        return class_indices
    
    def _get_class_to_samples(self, dataset, class_map, num_classes):
        class_to_samples = [[] for c in range(num_classes)]
        if isinstance(dataset, datasets.ImageFolder):
            for idx, (path, label) in enumerate(dataset.samples):
                if torch.is_tensor(label):
                    label = label.item()
                true_label = class_map[label]
                class_to_samples[true_label].append(path)
        elif isinstance(dataset, torch.utils.data.Subset):
            for i in range(len(dataset)):
                original_idx = dataset.indices[i]
                path, class_idx = dataset.dataset.samples[original_idx]
                true_label = class_map[class_idx]
                class_to_samples[true_label].append(path)
        else:
            raise ValueError(f"Dataset type {type(dataset)} not supported")
        return class_to_samples
    
    def _hyper_param_search_for_hard_label(self, image_tensor, image_path, hard_labels, mode='real'):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False,
                use_torchvision=self.use_torchvision,
                device=self.device
            )
            if self.use_dist:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
            acc = self._compute_hard_label_metrics(
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
    
    def _compute_hard_label_metrics_helper(self, image_tensor, image_path, hard_labels, lr, mode, hyper_param_search=False):
        if hyper_param_search:
            hard_label_acc, best_lr = self._hyper_param_search_for_hard_label(
                image_tensor=image_tensor,
                image_path=image_path,
                hard_labels=hard_labels,
                mode=mode
            )
        else:
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False,
                use_torchvision=self.use_torchvision,
                device=self.device
            )
            if self.use_dist:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
            hard_label_acc = self._compute_hard_label_metrics(
                model=model, 
                image_tensor=image_tensor,
                image_path=image_path,
                lr=lr, 
                hard_labels=hard_labels,
                mode=mode
            )
            best_lr = lr
        return hard_label_acc, best_lr

    def _compute_hard_label_metrics(self, model, image_tensor, image_path, lr, hard_labels, mode='real'):
        if mode == 'real':
            hard_label_dataset = self.dst_train
        else:
            if image_tensor is None:
                hard_label_dataset = datasets.ImageFolder(root=image_path, transform=self.custom_train_trans)
            else:
                hard_label_dataset = TensorDataset(image_tensor, hard_labels)
        
        if self.use_dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(hard_label_dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(hard_label_dataset)
        train_loader = DataLoader(hard_label_dataset, batch_size=self.real_batch_size if mode == 'real' else self.syn_batch_size, 
                                  num_workers=self.num_workers, sampler=train_sampler)

        loss_fn = torch.nn.CrossEntropyLoss()

        # We use default optimizer and lr scheduler to train a model on real data. These parameters are empirically set.
        if mode == 'real':
            real_data_training_config = REAL_DATA_TRAINING_CONFIG[f"{self.dataset}-{self.model_name}"]
            optimizer = get_optimizer(
                real_data_training_config['optimizer'], 
                model, 
                real_data_training_config['lr'], 
                real_data_training_config['weight_decay'], 
                real_data_training_config['momentum']
            )
            lr_scheduler = get_lr_scheduler(
                real_data_training_config['lr_scheduler'], 
                optimizer, 
                real_data_training_config['num_epochs'], 
                real_data_training_config['step_size']
            )
        else:
            optimizer = get_optimizer(self.optimizer, model, lr, self.weight_decay, self.momentum)
            lr_scheduler = get_lr_scheduler(self.lr_scheduler, optimizer, self.num_epochs, self.step_size)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_one_epoch(
                epoch=epoch, 
                stu_model=model, 
                loader=train_loader, 
                loss_fn=loss_fn, 
                optimizer=optimizer,
                aug_func=self.aug_func if mode == 'syn' else None,
                lr_scheduler=lr_scheduler,
                class_map=self.class_map if mode == 'real' else None,
                device=self.device
            )
            if (epoch + 1) % self.test_interval == 0:
                acc1 = validate(
                    model=model, 
                    loader=self.test_loader_real if mode == 'real' else self.test_loader_syn,
                    class_map=self.class_map,
                    device=self.device
                )
                if acc1 > best_acc1:
                    best_acc1 = acc1

        return best_acc1
    
    def _get_random_data_helper(self, eval_iter):
        if self.rank == 0:
            if self.random_data_format == 'tensor':
                random_data_tensors, random_data_hard_labels = get_random_data_tensors(self.dataset, self.dst_train, self.class_to_indices, 
                                                                                       self.ipc, self.class_map, eval_iter, self.random_data_path)
                random_data_tensors, random_data_hard_labels = random_data_tensors.to(self.device), random_data_hard_labels.to(self.device)
            elif self.random_data_format == 'image':
                if self.dataset in ['CIFAR10', 'CIFAR100']:
                    random_data_path = get_random_data_path_from_cifar(self.dataset, self.dst_train, self.class_to_indices, self.ipc, eval_iter, self.random_data_path)
                else:
                    random_data_path = get_random_data_path(self.dataset, self.class_to_samples, self.ipc, eval_iter, self.random_data_path)
            else:
                raise ValueError(f"Random data format {self.random_data_format} not supported")
        else:
            if self.random_data_format == 'tensor':
                random_data_tensors = torch.empty((self.num_classes * self.ipc, 3, self.im_size[0], self.im_size[1]), 
                                                    device=self.device)
                random_data_hard_labels = torch.empty((self.num_classes * self.ipc), dtype=torch.long, device=self.device)
            elif self.random_data_format == 'image':
                random_data_path = ""
            else:
                raise ValueError(f"Random data format {self.random_data_format} not supported")

        if self.use_dist:
            if self.random_data_format == 'tensor':
                torch.distributed.broadcast(random_data_tensors, src=0)
                torch.distributed.broadcast(random_data_hard_labels, src=0)
            elif self.random_data_format == 'image':
                random_data_path = broadcast_string(random_data_path, device=self.device, src=0)

        if self.random_data_format == 'tensor':
            return None, random_data_tensors, random_data_hard_labels
        else:
            return random_data_path, None, None
    
    def compute_metrics(self, image_tensor: Tensor=None, image_path: str=None, hard_labels: Tensor=None, syn_lr: float=None, lrs_lambda: float=0.5):
        if image_tensor is None and image_path is None:
            raise ValueError("Either image_tensor or image_path must be provided")

        if hard_labels is None:
            hard_labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), dtype=torch.long, requires_grad=False).view(-1)
        
        if torch.is_tensor(syn_lr):
            syn_lr = syn_lr.item()

        hard_label_recovery = []
        improvement_over_random = []
        lrs_list = []

        for i in range(self.num_eval):
            set_seed()
            logging(f"================ EVALUATION RUN {i+1}/{self.num_eval} ================")

            syn_data_hard_label_acc, best_lr = self._compute_hard_label_metrics_helper(
                image_tensor=image_tensor,
                image_path=image_path,
                hard_labels=hard_labels,
                lr=syn_lr,
                mode='syn',
                hyper_param_search=True if syn_lr is None else False
            )
            logging(f"Syn data hard label acc: {syn_data_hard_label_acc:.2f}%")

            if self.eval_full_data:
                full_data_hard_label_acc, best_lr = self._compute_hard_label_metrics_helper(
                    image_tensor=None,
                    image_path=None,
                    hard_labels=None,
                    lr=None,
                    mode='real',
                    hyper_param_search=False
                )
                logging(f"Full data hard label acc: {full_data_hard_label_acc:.2f}%")
            else:
                key = f"{self.dataset}-{self.model_name}"
                if key not in REAL_DATA_ACC_CACHE:
                    raise ValueError(f"Full data acc for {key} not found in cache. Please set eval_full_data to True.")
                full_data_hard_label_acc = REAL_DATA_ACC_CACHE[key]
            
            random_data_path, random_data_tensors, random_data_hard_labels = self._get_random_data_helper(eval_iter=i)            
            random_data_hard_label_acc, best_lr = self._compute_hard_label_metrics_helper(
                image_tensor=random_data_tensors,
                image_path=random_data_path,
                hard_labels=random_data_hard_labels,
                lr=None,
                mode='syn',
                hyper_param_search=True
            )
            logging(f"Random data hard label acc: {random_data_hard_label_acc:.2f}%")

            hlr = 1.00 * (full_data_hard_label_acc - syn_data_hard_label_acc)
            ior = 1.00 * (syn_data_hard_label_acc - random_data_hard_label_acc)
            alpha = lrs_lambda * ior - (1 - lrs_lambda) * hlr
            lrs = (np.exp(alpha) - np.exp(-1)) / (np.exp(1) - np.exp(-1))
            lrs_list.append(lrs)

            hard_label_recovery.append(hlr)
            improvement_over_random.append(ior)
            lrs_list.append(lrs)

        if self.use_dist:
            hard_label_recovery_tensor = torch.tensor(hard_label_recovery, device=self.device)
            improvement_over_random_tensor = torch.tensor(improvement_over_random, device=self.device)
            lrs_tensor = torch.tensor(lrs_list, device=self.device)
            
            torch.distributed.all_reduce(hard_label_recovery_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(improvement_over_random_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(lrs_tensor, op=torch.distributed.ReduceOp.SUM)

            hard_label_recovery = (hard_label_recovery_tensor / self.world_size).cpu().tolist()
            improvement_over_random = (improvement_over_random_tensor / self.world_size).cpu().tolist()
            lrs_list = (lrs_tensor / self.world_size).cpu().tolist()

        if self.rank == 0:
            results_to_save = {
                "hard_label_recovery": hard_label_recovery,
                "improvement_over_random": improvement_over_random,
                "lrs": lrs_list
            }
            save_results(results_to_save, self.save_path)

            hard_label_recovery_mean = np.mean(hard_label_recovery)
            hard_label_recovery_std = np.std(hard_label_recovery)
            improvement_over_random_mean = np.mean(improvement_over_random)
            improvement_over_random_std = np.std(improvement_over_random)
            lrs_mean = np.mean(lrs_list)
            lrs_std = np.std(lrs_list)

            logging(f"Hard Label Recovery Mean: {hard_label_recovery_mean:.2f}%  Std: {hard_label_recovery_std:.2f}")
            logging(f"Improvement Over Random Mean: {improvement_over_random_mean:.2f}%  Std: {improvement_over_random_std:.2f}")
            logging(f"Label-Robust Score Mean: {lrs_mean:.2f}%  Std: {lrs_std:.2f}")

        if self.use_dist:
            torch.distributed.destroy_process_group()

        return {
            "hard_label_recovery_mean": hard_label_recovery_mean,
            "hard_label_recovery_std": hard_label_recovery_std,
            "improvement_over_random_mean": improvement_over_random_mean,
            "improvement_over_random_std": improvement_over_random_std,
            "label_robust_score_mean": lrs_mean,
            "label_robust_score_std": lrs_std
        }
