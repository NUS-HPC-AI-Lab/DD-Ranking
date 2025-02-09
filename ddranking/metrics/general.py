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
from torchvision import transforms, datasets
from ddranking.utils import build_model, get_pretrained_model_path, get_dataset, TensorDataset, save_results, setup_dist
from ddranking.utils import set_seed, get_optimizer, get_lr_scheduler
from ddranking.utils import train_one_epoch, validate
from ddranking.loss import SoftCrossEntropyLoss, KLDivergenceLoss
from ddranking.aug import DSA, Mixup, Cutmix, ZCAWhitening
from ddranking.config import Config


class GeneralEvaluator:

    def __init__(self,
        config: Config=None,
        dataset: str='CIFAR10', 
        real_data_path: str='./dataset', 
        ipc: int=10,
        model_name: str='ConvNet-3',
        use_soft_label: bool=False,
        optimizer: str='sgd',
        lr_scheduler: str='step',
        data_aug_func: str='dsa',
        aug_params: dict=None,
        soft_label_mode: str='M',
        soft_label_criterion: str='kl',
        num_eval: int=5,
        im_size: tuple=(32, 32), 
        num_epochs: int=300,
        real_batch_size: int=256,
        syn_batch_size: int=256,
        weight_decay: float=0.0005,
        momentum: float=0.9,
        use_zca: bool=False,
        temperature: float=1.0,
        stu_use_torchvision: bool=False,
        tea_use_torchvision: bool=False,
        teacher_dir: str='./teacher_models',
        custom_train_trans: transforms.Compose=None,
        custom_val_trans: transforms.Compose=None,
        num_workers: int=4,
        save_path: str=None,
        dist: bool=False,
        device: str="cuda"
    ):

        if config is not None:
            self.config = config
            dataset = self.config.get('dataset', 'CIFAR10')
            real_data_path = self.config.get('real_data_path', './dataset')
            ipc = self.config.get('ipc', 10)
            model_name = self.config.get('model_name', 'ConvNet-3')
            use_soft_label = self.config.get('use_soft_label', False)
            soft_label_criterion = self.config.get('soft_label_criterion', 'sce')
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
            soft_label_mode = self.config.get('soft_label_mode', 'S')
            optimizer = self.config.get('optimizer', 'sgd')
            lr_scheduler = self.config.get('lr_scheduler', 'step')
            temperature = self.config.get('temperature', 1.0)
            weight_decay = self.config.get('weight_decay', 0.0005)
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
            custom_train_trans = self.config.get('custom_train_trans', None)
            custom_val_trans = self.config.get('custom_val_trans', None)
            device = self.config.get('device', 'cuda')
            dist = self.config.get('dist', False)

        self.use_dist = dist
        if dist:
            setup_dist(device)
            self.rank = setup_dist(device)
            self.world_size = torch.distributed.get_world_size()
            self.device = f'cuda:{self.rank}'

        channel, im_size, num_classes, dst_train, dst_test, class_map, class_map_inv = get_dataset(dataset, 
                                                                                                   real_data_path, 
                                                                                                   im_size,
                                                                                                   custom_train_trans,
                                                                                                   custom_val_trans,
                                                                                                   use_zca)
        self.num_classes = num_classes
        self.im_size = im_size
        self.real_test_loader = DataLoader(dst_test, batch_size=real_batch_size, num_workers=num_workers, shuffle=False)

        self.ipc = ipc
        self.model_name = model_name
        self.stu_use_torchvision = stu_use_torchvision
        self.custom_train_trans = custom_train_trans
        self.use_soft_label = use_soft_label
        if use_soft_label:
            assert soft_label_mode is not None, "soft_label_mode must be provided if use_soft_label is True"
            assert soft_label_criterion is not None, "soft_label_criterion must be provided if use_soft_label is True"
            self.soft_label_mode = soft_label_mode
            self.soft_label_criterion = soft_label_criterion
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.temperature = temperature

        self.num_eval = num_eval
        self.num_epochs = num_epochs
        self.syn_batch_size = syn_batch_size
        self.device = device

        if not save_path:
            save_path = f"./results/{dataset}/{model_name}/ipc{ipc}/eval_scores.csv"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        self.save_path = save_path

        if not use_torchvision:
            pretrained_model_path = get_pretrained_model_path(teacher_dir, model_name, dataset, ipc)
        else:
            pretrained_model_path = None

        self.teacher_model = build_model(
            model_name=model_name, 
            num_classes=num_classes, 
            im_size=self.im_size, 
            pretrained=True, 
            device=self.device, 
            model_path=pretrained_model_path,
            use_torchvision=tea_use_torchvision
        )
        self.teacher_model.eval()
        if self.use_dist:
            self.teacher_model = torch.nn.parallel.DistributedDataParallel(self.teacher_model, device_ids=[self.rank])

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

    def generate_soft_labels(self, images):
        batches = torch.split(images, self.syn_batch_size)
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
                use_torchvision=self.stu_use_torchvision,
                device=self.device
            )
            if self.use_dist:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])
            acc = self.compute_metrics_helper(
                model=model, 
                loader=loader,
                lr=lr
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
        return best_acc, best_lr
    
    def get_loss_fn(self):
        if self.use_soft_label:
            if self.soft_label_criterion == 'kl':
                return KLDivergenceLoss(temperature=self.temperature).to(self.device)
            elif self.soft_label_criterion == 'sce':
                return SoftCrossEntropyLoss(temperature=self.temperature).to(self.device)
            else:   
                raise ValueError(f"Invalid soft label criterion: {self.soft_label_criterion}")
        else:
            return CrossEntropyLoss().to(self.device)
    
    def compute_metrics_helper(self, model, loader, lr):
        loss_fn = self.get_loss_fn()
        
        optimizer = get_optimizer(model, self.optimizer, lr, self.weight_decay, self.momentum)
        scheduler = get_lr_scheduler(optimizer, self.lr_scheduler, self.num_epochs)
        
        best_acc = 0
        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs, desc="Training"):
            train_one_epoch(
                model=model, 
                loader=loader, 
                optimizer=optimizer, 
                loss_fn=loss_fn,
                soft_label_mode=self.soft_label_mode,
                aug_func=self.aug_func,
                lr_scheduler=scheduler,
                tea_model=self.teacher_model,
                logging=True,
                device=self.device
            )
            acc = validate(
                model=model, 
                loader=loader,
                logging=True,
                device=self.device
            )
            if acc > best_acc:
                best_acc = acc
        return best_acc
        
    def compute_metrics(self, image_tensor: Tensor=None, image_path: str=None, labels: Tensor=None, syn_lr=None):
        if image_tensor is None and image_path is None:
            raise ValueError("Either image_tensor or image_path must be provided")
        
        if self.use_soft_label and self.soft_label_mode == 'S' and labels is None:
            raise ValueError("labels must be provided if soft_label_mode is 'S'")
        
        if image_tensor is None:
            syn_dataset = datasets.ImageFolder(root=image_path, transform=self.custom_train_trans)
            if labels is not None:
                syn_dataset.samples = [(path, labels[idx]) for idx, (path, _) in enumerate(syn_dataset.samples)]
                syn_dataset.targets = labels
        else:
            if labels is not None:
                syn_dataset = TensorDataset(image_tensor, labels, transform=self.custom_train_trans)
            else:
                # use hard labels if labels are not provided
                default_labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), 
                                              dtype=torch.long, requires_grad=False).view(-1)
                syn_dataset = TensorDataset(image_tensor, default_labels, transform=self.custom_train_trans)

        syn_loader = DataLoader(syn_dataset, batch_size=self.syn_batch_size, shuffle=True, num_workers=4)

        accs = []
        lrs = []
        for i in range(self.num_eval):
            set_seed()
            if self.rank == 0:
                print(f"########################### {i+1}th Evaluation ###########################")
            if syn_lr:
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
                syn_data_acc = self.compute_metrics_helper(
                    model=model,
                    loader=syn_loader,
                    lr=syn_lr
                )
                del model
            else:
                syn_data_acc, best_lr = self.hyper_param_search(syn_loader)
            accs.append(syn_data_acc)
            if syn_lr:
                lrs.append(syn_lr)
            else:
                lrs.append(best_lr)
        
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

            print(f"Accs Mean: {accs_mean:.2f}%  Std: {accs_std:.2f}")
        
        if self.use_dist:
            torch.distributed.destroy_process_group()
