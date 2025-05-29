import os
import time
import warnings
import torch
import random
import numpy as np
import torch.nn.functional as F
from typing import List
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision import transforms, datasets
from ddranking.utils import build_model, get_pretrained_model_path, get_dataset, TensorDataset, save_results, setup_dist
from ddranking.utils import set_seed, get_optimizer, get_lr_scheduler
from ddranking.utils import train_one_epoch, validate, logging
from ddranking.loss import SoftCrossEntropyLoss, KLDivergenceLoss, MSEGTLoss
from ddranking.aug import DSA, Mixup, Cutmix, ZCAWhitening
from ddranking.config import Config


class GeneralEvaluator:

    def __init__(self, config: Config=None, dataset: str='CIFAR10', real_data_path: str='./dataset', ipc: int=10, model_name: str='ConvNet-3', 
                 use_soft_label: bool=False, optimizer: str='adamw', lr_scheduler: str='cosine', weight_decay: float=0.01, momentum: float=0.9, 
                 step_size: int=None, data_aug_func: str='dsa', aug_params: dict=None, soft_label_mode: str='M', soft_label_criterion: str='kl', 
                 loss_fn_kwargs: dict=None, num_eval: int=5, im_size: tuple=(32, 32), num_epochs: int=300, batch_size: int=256, use_zca: bool=False, 
                 stu_use_torchvision: bool=False, tea_use_torchvision: bool=False, teacher_dir: str='./teacher_models', teacher_model_names: List[str]=None, 
                 custom_train_trans: transforms.Compose=None, custom_val_trans: transforms.Compose=None, num_workers: int=4, save_path: str=None, 
                 dist: bool=False, device: str="cuda"):

        if config is not None:
            self.config = config
            dataset = self.config.get('dataset', 'CIFAR10')
            real_data_path = self.config.get('real_data_path', './dataset')
            ipc = self.config.get('ipc', 10)
            model_name = self.config.get('model_name', 'ConvNet-3')
            use_soft_label = self.config.get('use_soft_label', False)
            soft_label_criterion = self.config.get('soft_label_criterion', 'sce')
            loss_fn_kwargs = self.config.get('loss_fn_kwargs', {'temperature': 1.0, 'scale_loss': False})
            data_aug_func = self.config.get('data_aug_func', 'dsa')
            aug_params = self.config.get('aug_params', {
                "prob_flip": 0.5,
                "ratio_rotate": 15.0,
                "saturation": 2.0,
                "brightness": 1.0,
                "contrast": 0.5,
                "ratio_scale": 1.2,
                "ratio_crop_pad": 0.125,
                "ratio_cutout": 0.5
            })
            soft_label_mode = self.config.get('soft_label_mode', 'M')
            optimizer = self.config.get('optimizer', 'adamw')
            lr_scheduler = self.config.get('lr_scheduler', 'cosine')
            step_size = self.config.get('step_size', None)
            weight_decay = self.config.get('weight_decay', 0.01)
            momentum = self.config.get('momentum', 0.9)
            num_eval = self.config.get('num_eval', 5)
            im_size = self.config.get('im_size', (32, 32))
            num_epochs = self.config.get('num_epochs', 300)
            real_batch_size = self.config.get('real_batch_size', 256)
            syn_batch_size = self.config.get('syn_batch_size', 256)
            default_lr = self.config.get('default_lr', 0.01)
            save_path = self.config.get('save_path', None)
            num_workers = self.config.get('num_workers', 4)
            stu_use_torchvision = self.config.get('stu_use_torchvision', False)
            tea_use_torchvision = self.config.get('tea_use_torchvision', False)
            teacher_dir = self.config.get('teacher_dir', './teacher_models')
            teacher_model_names = self.config.get('teacher_model_names', [model_name])
            custom_train_trans = self.config.get('custom_train_trans', None)
            custom_val_trans = self.config.get('custom_val_trans', None)
            device = self.config.get('device', 'cuda')
            dist = self.config.get('dist', False)

        self.use_dist = dist
        self.device = device
        # setup dist
        if self.use_dist:
            setup_dist(self)
        else:
            self.rank = 0
            self.gpu = 0
            self.world_size = 1
        
        batch_size = batch_size // self.world_size

        channel, im_size, mean, std, num_classes, dst_train, dst_test, _, class_map, class_map_inv = get_dataset(
            dataset, 
            real_data_path, 
            im_size, 
            use_zca,
            custom_val_trans,
            device
        )
        self.num_classes = num_classes
        self.im_size = im_size
        self.class_map = class_map

        if self.use_dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(dst_test)
        else:
            test_sampler = torch.utils.data.RandomSampler(dst_test)

        self.test_loader = DataLoader(dst_test, batch_size=batch_size, num_workers=num_workers, sampler=test_sampler)

        self.ipc = ipc
        self.model_name = model_name
        self.stu_use_torchvision = stu_use_torchvision
        self.tea_use_torchvision = tea_use_torchvision
        self.custom_train_trans = custom_train_trans
        self.use_soft_label = use_soft_label
        if use_soft_label:
            assert soft_label_mode is not None, "soft_label_mode must be provided if use_soft_label is True"
            assert soft_label_criterion is not None, "soft_label_criterion must be provided if use_soft_label is True"
            self.soft_label_mode = soft_label_mode
            self.soft_label_criterion = soft_label_criterion
            self.loss_fn_kwargs = loss_fn_kwargs
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.step_size = step_size
        self.num_eval = num_eval
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device

        if not save_path:
            save_path = f"./results/{dataset}/{model_name}/ipc{ipc}/eval_scores.csv"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.save_path = save_path

        if not use_torchvision:
            pretrained_model_paths = get_pretrained_model_path(teacher_dir, teacher_model_names, dataset)
        else:
            pretrained_model_paths = None

        self.teacher_models = [
            build_model(
                teacher_model_name,
                num_classes=self.num_classes,
                im_size=self.im_size,
                pretrained=True,
                device=self.device, 
                model_path=pretrained_model_path,
                use_torchvision=tea_use_torchvision
            )
            for teacher_model_name, pretrained_model_path in zip(teacher_model_names, pretrained_model_paths)
        ]
        for teacher_model in self.teacher_models:
            teacher_model.eval()
            teacher_model.to(self.device)

        if self.use_dist:
            self.teacher_models = [
                torch.nn.parallel.DistributedDataParallel(
                    teacher_model,
                    device_ids=[self.gpu]
                )
                for teacher_model in self.teacher_models
            ]

        if data_aug_func is None:
            self.aug_func = None
        elif data_aug_func == 'dsa':
            self.aug_func = DSA(aug_params)
        elif data_aug_func == 'mixup':
            self.aug_func = Mixup(aug_params)  
        elif data_aug_func == 'cutmix':
            self.aug_func = Cutmix(aug_params)
        else:
            raise ValueError(f"Invalid data augmentation function: {data_aug_func}")
    
    def _get_loss_fn(self):
        if self.use_soft_label:
            if self.soft_label_criterion == 'kl':
                return KLDivergenceLoss(**self.loss_fn_kwargs).to(self.device)
            elif self.soft_label_criterion == 'sce':
                return SoftCrossEntropyLoss(**self.loss_fn_kwargs).to(self.device)
            elif self.soft_label_criterion == 'mse_gt':
                return MSEGTLoss(**self.loss_fn_kwargs).to(self.device)
            else:   
                raise ValueError(f"Invalid soft label criterion: {self.soft_label_criterion}")
        else:
            return CrossEntropyLoss().to(self.device)
    
    def _hyper_param_search_for_hard_label(self, image_tensor, image_path, hard_labels):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False,
                use_torchvision=self.stu_use_torchvision,
                device=self.device
            )
            if self.use_dist:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])
            acc = self._compute_hard_label_metrics(
                model=model, 
                image_tensor=image_tensor,
                image_path=image_path,
                lr=lr, 
                hard_labels=hard_labels
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr

    def _hyper_param_search_for_soft_label(self, image_tensor, image_path, soft_labels):
        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        
        best_acc = 0
        best_lr = 0
        for lr in lr_list:
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size,
                pretrained=False,
                use_torchvision=self.stu_use_torchvision,
                device=self.device
            )
            if self.use_dist:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])
            acc = self._compute_soft_label_metrics(
                model=model, 
                image_tensor=image_tensor,
                image_path=image_path,
                lr=lr, 
                soft_labels=soft_labels
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr
    
    def _compute_hard_label_metrics(self, model, image_tensor, image_path, lr, hard_labels):
        
        if image_tensor is None:
            hard_label_dataset = datasets.ImageFolder(root=image_path, transform=self.custom_train_trans)
        else:
            hard_label_dataset = TensorDataset(image_tensor, hard_labels)

        if self.use_dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(hard_label_dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(hard_label_dataset)
        train_loader = DataLoader(hard_label_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=train_sampler)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = get_optimizer(self.optimizer, model, lr, self.weight_decay, self.momentum)
        lr_scheduler = get_lr_scheduler(self.lr_scheduler, optimizer, self.num_epochs, self.step_size)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs, desc="Training with hard labels", disable=self.rank != 0):
            train_one_epoch(
                epoch=epoch, 
                stu_model=model, 
                loader=train_loader,
                loss_fn=loss_fn, 
                optimizer=optimizer,
                aug_func=self.aug_func,
                lr_scheduler=lr_scheduler,
                class_map=self.class_map,
                tea_models=self.teacher_models, 
                device=self.device
            )
            if (epoch + 1) % self.test_interval == 0:
                acc1 = validate(
                    model=model, 
                    loader=self.test_loader,
                    class_map=self.class_map,
                    device=self.device
                )
                if acc1 > best_acc1:
                    best_acc1 = acc1

        return best_acc1
        
    def _compute_soft_label_metrics(self, model, image_tensor, image_path, lr, soft_labels):
        if soft_labels is None:
            labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), dtype=torch.long, requires_grad=False).view(-1)
        else:
            labels = soft_labels
            
        if image_tensor is None:
            soft_label_dataset = datasets.ImageFolder(root=image_path, transform=self.custom_train_trans)
        else:
            soft_label_dataset = TensorDataset(image_tensor, labels)

        if self.use_dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(soft_label_dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(soft_label_dataset)
        train_loader = DataLoader(soft_label_dataset, batch_size=self.syn_batch_size, num_workers=self.num_workers, sampler=train_sampler)

        if self.soft_label_criterion == 'sce':
            loss_fn = SoftCrossEntropyLoss(**self.loss_fn_kwargs).to(self.device)
        elif self.soft_label_criterion == 'kl':
            loss_fn = KLDivergenceLoss(**self.loss_fn_kwargs).to(self.device)
        elif self.soft_label_criterion == 'mse_gt':
            loss_fn = MSEGTLoss(**self.loss_fn_kwargs).to(self.device)
        else:
            raise NotImplementedError(f"Soft label criterion {self.soft_label_criterion} not implemented")
        
        optimizer = get_optimizer(self.optimizer, model, lr, self.weight_decay, self.momentum)
        lr_scheduler = get_lr_scheduler(self.lr_scheduler, optimizer, self.num_epochs, self.step_size)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs, desc="Training with soft labels", disable=self.rank != 0):
            train_one_epoch(
                epoch=epoch, 
                stu_model=model,
                loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                aug_func=self.aug_func,
                soft_label_mode=self.soft_label_mode,
                lr_scheduler=lr_scheduler,
                class_map=self.class_map,
                tea_models=self.teacher_models,
                device=self.device
            )
            if (epoch + 1) % self.test_interval == 0:
                acc1 = validate(
                    model=model, 
                    loader=self.test_loader,
                    class_map=self.class_map,
                    device=self.device
                )
                if acc1 > best_acc1:
                    best_acc1 = acc1
        
        return best_acc1

    def _compute_hard_label_metrics_helper(self, model, image_tensor, image_path, hard_labels, lr, hyper_param_search=False):
        if hyper_param_search:
            warnings.warn("You are not providing learning rate for the evaluation. By default, we conduct hyper-parameter search for the best learning rate. \
                           To match your own results, we recommend you to provide the learning rate.")
            hard_label_acc, best_lr = self.hyper_param_search_for_hard_label(
                image_tensor=image_tensor,
                image_path=image_path,
                hard_labels=hard_labels
            )
        else:
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False,
                use_torchvision=self.stu_use_torchvision,
                device=self.device
            )
            if self.use_dist:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
            hard_label_acc = self._compute_hard_label_metrics(
                model=model, 
                image_tensor=image_tensor,
                image_path=image_path,
                lr=lr, 
                hard_labels=hard_labels
            )
            best_lr = lr
        return hard_label_acc, best_lr
    
    def _compute_soft_label_metrics_helper(self, image_tensor, image_path, soft_labels, lr, hyper_param_search=False):
        if hyper_param_search:
            warnings.warn("You are not providing learning rate for the evaluation. By default, we conduct hyper-parameter search for the best learning rate. \
                           To match your own results, we recommend you to provide the learning rate.")
            soft_label_acc, best_lr = self.hyper_param_search_for_soft_label(
                image_tensor=image_tensor,
                image_path=image_path,
                soft_labels=soft_labels
            )
        else:
            model = build_model(
                model_name=self.model_name, 
                num_classes=self.num_classes, 
                im_size=self.im_size, 
                pretrained=False,
                use_torchvision=self.stu_use_torchvision,
                device=self.device
            )
            if self.use_dist:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
            soft_label_acc = self._compute_soft_label_metrics(
                model=model,
                image_tensor=image_tensor,
                image_path=image_path,
                lr=lr,
                soft_labels=soft_labels
            )
            best_lr = lr
        return soft_label_acc, best_lr
        
    def compute_metrics(self, image_tensor: Tensor=None, image_path: str=None, labels: Tensor=None, syn_lr=None):
        if image_tensor is None and image_path is None:
            raise ValueError("Either image_tensor or image_path must be provided")
        
        if self.use_soft_label and self.soft_label_mode == 'S' and labels is None:
            raise ValueError("labels must be provided if soft_label_mode is 'S'")
        
        if torch.is_tensor(syn_lr):
            syn_lr = syn_lr.item()

        accs = []
        lrs = []
        for i in range(self.num_eval):
            set_seed()
            logging(f"================ EVALUATION RUN {i+1}/{self.num_eval} ================")
            if self.use_soft_label:
                acc, lr = self._compute_soft_label_metrics_helper(
                    image_tensor=image_tensor,
                    image_path=image_path,
                    soft_labels=labels,
                    lr=syn_lr,
                    hyper_param_search=True if syn_lr is None else False
                )
            else:
                acc, lr = self._compute_hard_label_metrics_helper(
                    image_tensor=image_tensor,
                    image_path=image_path,
                    hard_labels=labels,
                    lr=syn_lr,
                    hyper_param_search=True if syn_lr is None else False
                )
        
        if self.use_dist:
            accs_tensor = torch.tensor(accs, device=self.device)
            lrs_tensor = torch.tensor(lrs, device=self.device)
            
            torch.distributed.all_reduce(accs_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(lrs_tensor, op=torch.distributed.ReduceOp.SUM)
        
        if self.rank == 0:
            results_to_save = {
                "accs": accs,
                "lrs": lrs
            }
            save_results(results_to_save, self.save_path)

            accs_mean = np.mean(accs)
            accs_std = np.std(accs)

            print(f"Acc. Mean: {accs_mean:.2f}%  Std: {accs_std:.2f}")
        
        if self.use_dist:
            torch.distributed.destroy_process_group()
