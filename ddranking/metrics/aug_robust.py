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
from ddranking.utils import TensorDataset, get_random_data_tensors, get_random_data_path, get_random_data_path_from_cifar, get_dataset
from ddranking.utils import save_results, setup_dist, logging, broadcast_string
from ddranking.utils import set_seed, train_one_epoch, validate, get_optimizer, get_lr_scheduler
from ddranking.loss import SoftCrossEntropyLoss, KLDivergenceLoss, MSEGTLoss
from ddranking.aug import DSA, Mixup, Cutmix, ZCAWhitening
from ddranking.config import Config


class AugmentationRobustScore:

    def __init__(self, config: Config=None, dataset: str='CIFAR10', real_data_path: str='./dataset/', ipc: int=10, model_name: str='ConvNet-3', 
                 data_aug_func: str='cutmix', aug_params: dict={'beta': 1.0}, label_type: str='soft', soft_label_mode: str='S', soft_label_criterion: str='kl', 
                 optimizer: str='sgd', lr_scheduler: str='step', loss_fn_kwargs: dict=None, step_size: int=None, weight_decay: float=0.0005, 
                 momentum: float=0.9, num_eval: int=5, im_size: tuple=(32, 32), num_epochs: int=300, use_zca: bool=False, batch_size: int=256, save_path: str=None, 
                 stu_use_torchvision: bool=False, tea_use_torchvision: bool=False, num_workers: int=4, teacher_dir: str='./teacher_models', 
                 teacher_model_names: List[str]=None, random_data_path: str=None, random_data_format: str='image', custom_train_trans: transforms.Compose=None, 
                 custom_val_trans: transforms.Compose=None, device: str="cuda", dist: bool=False):

        if config is not None:
            self.config = config
            dataset = self.config.get('dataset')
            real_data_path = self.config.get('real_data_path')
            ipc = self.config.get('ipc')
            model_name = self.config.get('model_name')
            data_aug_func = self.config.get('data_aug_func')
            aug_params = self.config.get('aug_params')
            label_type = self.config.get('label_type')
            soft_label_mode = self.config.get('soft_label_mode')
            soft_label_criterion = self.config.get('soft_label_criterion')
            optimizer = self.config.get('optimizer')
            lr_scheduler = self.config.get('lr_scheduler')
            loss_fn_kwargs = self.config.get('loss_fn_kwargs')
            step_size = self.config.get('step_size')
            weight_decay = self.config.get('weight_decay')
            momentum = self.config.get('momentum')
            num_eval = self.config.get('num_eval')
            im_size = self.config.get('im_size')
            num_epochs = self.config.get('num_epochs')
            use_zca = self.config.get('use_zca')
            batch_size = self.config.get('batch_size')
            save_path = self.config.get('save_path')
            num_workers = self.config.get('num_workers')
            stu_use_torchvision = self.config.get('stu_use_torchvision')
            tea_use_torchvision = self.config.get('tea_use_torchvision')
            custom_train_trans = self.config.get('custom_train_trans')
            custom_val_trans = self.config.get('custom_val_trans')
            teacher_dir = self.config.get('teacher_dir')
            teacher_model_names = self.config.get('teacher_model_names')
            random_data_path = self.config.get('random_data_path')
            random_data_format = self.config.get('random_data_format')
            device = self.config.get('device')
            dist = self.config.get('dist')
        
        self.use_dist = dist
        # setup dist
        if self.use_dist:
            setup_dist(self)
        else:
            self.rank = 0
            self.gpu = 0
            self.world_size = 1

        batch_size = batch_size // self.world_size

        self.device = device

        channel, im_size, mean, std, num_classes, dst_train, dst_test_real, dst_test_syn, class_map, class_map_inv = get_dataset(
            dataset, 
            real_data_path, 
            im_size, 
            use_zca,
            custom_val_trans,
            device
        )
        del dst_test_real

        self.default_transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.dataset = dataset
        self.class_map = class_map
        self.class_to_indices = self._get_class_to_indices(dst_train, class_map, num_classes)
        if dataset not in ['CIFAR10', 'CIFAR100']:
            self.class_to_samples = self._get_class_to_samples(dst_train, class_map, num_classes)
        self.dst_train = dst_train

        if self.use_dist:
            test_sampler_syn = torch.utils.data.distributed.DistributedSampler(dst_test_syn)
        else:
            test_sampler_syn = torch.utils.data.RandomSampler(dst_test_syn)

        self.test_loader_syn = DataLoader(dst_test_syn, batch_size=batch_size, num_workers=num_workers, sampler=test_sampler_syn)

        self.label_type = label_type
        if label_type == 'soft':
            assert soft_label_mode is not None and soft_label_criterion is not None and loss_fn_kwargs is not None, "You must specify soft_label_mode, soft_label_criterion, and loss_fn_kwargs when label_type is 'soft'"
            self.soft_label_mode = soft_label_mode
            self.soft_label_criterion = soft_label_criterion
            self.loss_fn_kwargs = loss_fn_kwargs

        # data info
        self.im_size = im_size
        self.num_classes = num_classes
        self.ipc = ipc
        self.custom_train_trans = custom_train_trans

        self.random_data_path = random_data_path
        self.random_data_format = random_data_format

        # training info
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay

        self.momentum = momentum
        self.num_eval = num_eval
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.test_interval = 20
        self.num_workers = num_workers
        self.device = device

        # data augmentation
        if data_aug_func == 'dsa':
            self.aug_func = DSA(aug_params)
        elif data_aug_func == 'mixup':
            self.aug_func = Mixup(aug_params)
        elif data_aug_func == 'cutmix':
            self.aug_func = Cutmix(aug_params)
        else:
            self.aug_func = None

        # save path
        if not save_path:
            save_path = f"./results/{dataset}/{model_name}/ipc{ipc}/aug_scores.csv"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.save_path = save_path

        # teacher model
        self.stu_use_torchvision = stu_use_torchvision

        if label_type == 'soft':
            pretrained_model_paths = get_pretrained_model_path(teacher_dir, teacher_model_names, dataset)
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
    
    def _get_class_to_indices(self, dataset, class_map, num_classes):
        class_to_indices = [[] for c in range(num_classes)]
        for idx, label in enumerate(dataset.targets):
            if torch.is_tensor(label):
                label = label.item()
            true_label = class_map[label]
            class_to_indices[true_label].append(idx)
        return class_to_indices
    
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
    
    def _hyper_param_search_for_hard_label(self, image_tensor, image_path, hard_labels, use_aug):
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
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
            acc = self._compute_hard_label_metrics(
                model=model, 
                image_tensor=image_tensor,
                image_path=image_path,
                lr=lr, 
                hard_labels=hard_labels,
                use_aug=use_aug
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr

    def _hyper_param_search_for_soft_label(self, image_tensor, image_path, soft_labels, use_aug):
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
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
            acc = self._compute_soft_label_metrics(
                model=model, 
                image_tensor=image_tensor,
                image_path=image_path,
                lr=lr, 
                soft_labels=soft_labels,
                use_aug=use_aug
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
            del model
        return best_acc, best_lr
    
    def _compute_hard_label_metrics(self, model, image_tensor, image_path, lr, hard_labels, use_aug):

        if image_tensor is None:
            hard_label_dataset = datasets.ImageFolder(root=image_path, transform=self.custom_train_trans if use_aug else self.default_transform)
        else:
            hard_label_dataset = TensorDataset(image_tensor, hard_labels)

        if self.use_dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(hard_label_dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(hard_label_dataset)
        train_loader = DataLoader(hard_label_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=train_sampler)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = get_optimizer(self.optimizer, model, lr, self.weight_decay, self.momentum)
        lr_scheduler = get_lr_scheduler(self.lr_scheduler, optimizer, self.num_epochs)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs, desc="Training with hard labels", disable=self.rank != 0):
            train_one_epoch(
                epoch=epoch, 
                stu_model=model, 
                loader=train_loader,
                loss_fn=loss_fn, 
                optimizer=optimizer,
                aug_func=self.aug_func if use_aug else None,
                lr_scheduler=lr_scheduler, 
                tea_models=self.teacher_models,
                device=self.device
            )
            if (epoch + 1) % self.test_interval == 0:
                acc1 = validate(
                    model=model, 
                    loader=self.test_loader_real,
                    class_map=self.class_map,
                    device=self.device
                )
                if acc1 > best_acc1:
                    best_acc1 = acc1

        return best_acc1
        
    def _compute_soft_label_metrics(self, model, image_tensor, image_path, lr, soft_labels, use_aug):
        if soft_labels is None:
            labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), dtype=torch.long, requires_grad=False).view(-1)
        else:
            labels = soft_labels
            
        if image_tensor is None:
            soft_label_dataset = datasets.ImageFolder(root=image_path, transform=self.custom_train_trans if use_aug else self.default_transform)
        else:
            soft_label_dataset = TensorDataset(image_tensor, labels)

        if self.use_dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(soft_label_dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(soft_label_dataset)
        train_loader = DataLoader(soft_label_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=train_sampler)

        if self.soft_label_criterion == 'sce':
            loss_fn = SoftCrossEntropyLoss(**self.loss_fn_kwargs).to(self.device)
        elif self.soft_label_criterion == 'kl':
            loss_fn = KLDivergenceLoss(**self.loss_fn_kwargs).to(self.device)
        elif self.soft_label_criterion == 'mse_gt':
            loss_fn = MSEGTLoss(**self.loss_fn_kwargs).to(self.device)
        else:
            raise NotImplementedError(f"Soft label criterion {self.soft_label_criterion} not implemented")
        
        optimizer = get_optimizer(self.optimizer, model, lr, self.weight_decay, self.momentum)
        lr_scheduler = get_lr_scheduler(self.lr_scheduler, optimizer, self.num_epochs)

        best_acc1 = 0
        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs, desc="Training with soft labels", disable=self.rank != 0):
            train_one_epoch(
                epoch=epoch, 
                stu_model=model,
                loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                aug_func=self.aug_func if use_aug else None,
                soft_label_mode=self.soft_label_mode,
                lr_scheduler=lr_scheduler,
                tea_models=self.teacher_models,
                device=self.device
            )
            if (epoch + 1) % self.test_interval == 0:
                acc1 = validate(
                    model=model, 
                    loader=self.test_loader_syn,
                    class_map=self.class_map,
                    device=self.device
                )
                if acc1 > best_acc1:
                    best_acc1 = acc1
        
        return best_acc1

    def _generate_soft_labels(self, images):
        images = images.to(self.device)
        batches = torch.split(images, self.batch_size)
        soft_labels = []
        
        with torch.no_grad():
            for image_batch in batches:
                outputs = [teacher_model(image_batch) for teacher_model in self.teacher_models]
                outputs = torch.stack(outputs, dim=0).mean(dim=0)
                soft_labels.append(outputs.detach().cpu())
        
        soft_labels = torch.cat(soft_labels, dim=0)
        soft_labels = soft_labels.detach().cpu()
        return soft_labels

    def _get_random_data_helper(self, eval_iter):
        if self.rank == 0:
            if self.random_data_format == 'tensor':
                random_data_tensors, random_data_hard_labels = get_random_data_tensors(self.dataset, self.dst_train, self.class_to_indices, self.ipc, eval_iter, self.random_data_path)
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
                                                    device='cpu')
            elif self.random_data_format == 'image':
                random_data_path = ""
            else:
                raise ValueError(f"Random data format {self.random_data_format} not supported")

        if self.use_dist:
            if self.random_data_format == 'tensor':
                torch.distributed.broadcast(random_data_tensors, src=0)
            elif self.random_data_format == 'image':
                random_data_path = broadcast_string(random_data_path, device=self.device, src=0)

        if self.label_type == 'soft' and self.soft_label_mode == 'S':
            random_data_soft_labels = self._generate_soft_labels(random_data_tensors)
        else:
            random_data_soft_labels = None

        if self.random_data_format == 'tensor':
            return None, random_data_tensors, random_data_hard_labels, random_data_soft_labels
        else:
            return random_data_path, None, None, None
    
    def _compute_with_aug_metrics_helper(self, image_tensor, image_path, hard_labels, soft_labels, lr, hyper_param_search=False):
        if hyper_param_search:
            if self.label_type == 'hard':
                with_aug_acc, best_lr = self._hyper_param_search_for_hard_label(
                    image_tensor=image_tensor,
                    image_path=image_path,
                    hard_labels=hard_labels,
                    use_aug=True
                )
            else:
                with_aug_acc, best_lr = self._hyper_param_search_for_soft_label(
                    image_tensor=image_tensor,
                    image_path=image_path,
                    soft_labels=soft_labels,
                    use_aug=True
                )
        else:
            best_lr = lr
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
            if self.label_type == 'hard':
                with_aug_acc = self._compute_hard_label_metrics(
                    model=model,
                    image_tensor=image_tensor,
                    image_path=image_path,
                    lr=lr,
                    hard_labels=hard_labels,
                    use_aug=True
                )
            else:
                with_aug_acc = self._compute_soft_label_metrics(
                    model=model,
                    image_tensor=image_tensor,
                    image_path=image_path,
                    lr=lr,
                    soft_labels=soft_labels,
                    use_aug=True
                )
        return with_aug_acc, best_lr
    
    def _compute_without_aug_metrics_helper(self, image_tensor, image_path, hard_labels, soft_labels, lr, hyper_param_search=False):
        if hyper_param_search:
            if self.label_type == 'hard':
                without_aug_acc, best_lr = self._hyper_param_search_for_hard_label(
                    image_tensor=image_tensor,
                    image_path=image_path,
                    hard_labels=hard_labels,
                    use_aug=False
                )
            else:
                without_aug_acc, best_lr = self._hyper_param_search_for_soft_label(
                    image_tensor=image_tensor,
                    image_path=image_path,
                    soft_labels=soft_labels,
                    use_aug=False
                )
        else:
            best_lr = lr
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
            if self.label_type == 'hard':
                without_aug_acc = self._compute_hard_label_metrics(
                    model=model,
                    image_tensor=image_tensor,
                    image_path=image_path,
                    lr=lr,
                    hard_labels=hard_labels,
                    use_aug=False
                )
            else:
                without_aug_acc = self._compute_soft_label_metrics(
                    model=model,
                    image_tensor=image_tensor,
                    image_path=image_path,
                    lr=lr,
                    soft_labels=soft_labels,
                    use_aug=False
                )
        return without_aug_acc, best_lr

    def compute_metrics(self, image_tensor: Tensor=None, image_path: str=None, soft_labels: Tensor=None, syn_lr: float=None, ars_lambda: float=0.5):
        if image_tensor is None and image_path is None:
            raise ValueError("Either image_tensor or image_path must be provided")

        if self.label_type == 'soft' and self.soft_label_mode == 'S' and soft_labels is None:
            raise ValueError("Soft label mode 'S' requires soft labels")
        
        if torch.is_tensor(syn_lr):
            syn_lr = syn_lr.item()

        hard_labels = torch.tensor(np.array([np.ones(self.ipc) * i for i in range(self.num_classes)]), 
                                   dtype=torch.long, requires_grad=False).view(-1)

        with_aug_scores = []
        without_aug_scores = []
        ars_list = []

        for i in range(self.num_eval):
            set_seed()
            logging(f"================ EVALUATION RUN {i+1}/{self.num_eval} ================")

            syn_data_with_aug_acc, best_lr = self._compute_with_aug_metrics_helper(
                image_tensor=image_tensor,
                image_path=image_path,
                hard_labels=hard_labels,
                soft_labels=soft_labels,
                lr=syn_lr,
                hyper_param_search=True if syn_lr is None else False
            )
            logging(f"With augmentation acc: {syn_data_with_aug_acc:.2f}%")

            random_data_path, random_data_tensors, random_data_hard_labels, random_data_soft_labels = self._get_random_data_helper(eval_iter=i)
            random_data_with_aug_acc, best_lr = self._compute_with_aug_metrics_helper(
                image_tensor=random_data_tensors,
                image_path=random_data_path,
                hard_labels=random_data_hard_labels,
                soft_labels=random_data_soft_labels,
                lr=None,
                hyper_param_search=True
            )
            logging(f"Without augmentation acc: {random_data_with_aug_acc:.2f}%")

            syn_data_without_aug_acc, best_lr = self._compute_without_aug_metrics_helper(
                image_tensor=image_tensor,
                image_path=image_path,
                hard_labels=hard_labels,
                soft_labels=soft_labels,
                lr=syn_lr,
                hyper_param_search=True if syn_lr is None else False
            )
            logging(f"Without augmentation acc: {syn_data_without_aug_acc:.2f}%")
            
            random_data_without_aug_acc, best_lr = self._compute_without_aug_metrics_helper(
                image_tensor=random_data_tensors,
                image_path=random_data_path,
                hard_labels=random_data_hard_labels,
                soft_labels=random_data_soft_labels,
                lr=None,
                hyper_param_search=True
            )
            logging(f"Without augmentation acc: {random_data_without_aug_acc:.2f}%")

            with_aug_score = 1.00 * (syn_data_with_aug_acc - random_data_with_aug_acc)
            without_aug_score = 1.00 * (syn_data_without_aug_acc - random_data_without_aug_acc)
            beta = ars_lambda * with_aug_score + (1 - ars_lambda) * without_aug_score
            ars = (np.exp(beta) - np.exp(-1)) / (np.exp(1) - np.exp(-1))

            with_aug_scores.append(with_aug_score)
            without_aug_scores.append(without_aug_score)
            ars_list.append(ars)
        
        if self.use_dist:
            with_aug_scores_tensor = torch.tensor(with_aug_scores, device=self.device)
            without_aug_scores_tensor = torch.tensor(without_aug_scores, device=self.device)
            ars_tensor = torch.tensor(ars_list, device=self.device)
            
            torch.distributed.all_reduce(with_aug_scores_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(without_aug_scores_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(ars_tensor, op=torch.distributed.ReduceOp.SUM)

            with_aug_scores = (with_aug_scores_tensor / self.world_size).cpu().tolist()
            without_aug_scores = (without_aug_scores_tensor / self.world_size).cpu().tolist()
            ars_list = (ars_tensor / self.world_size).cpu().tolist()

        if self.rank == 0:
            results_to_save = {
                "with_aug_scores": with_aug_scores,
                "without_aug_scores": without_aug_scores,
                "ars_list": ars_list
            }
            save_results(results_to_save, self.save_path)

            with_aug_scores_mean = np.mean(with_aug_scores)
            with_aug_scores_std = np.std(with_aug_scores)
            without_aug_scores_mean = np.mean(without_aug_scores)
            without_aug_scores_std = np.std(without_aug_scores)
            ars_mean = np.mean(ars_list)
            ars_std = np.std(ars_list)

            logging(f"With Augmentation Mean: {with_aug_scores_mean:.2f}%  Std: {with_aug_scores_std:.2f}")
            logging(f"Without Augmentation Mean: {without_aug_scores_mean:.2f}%  Std: {without_aug_scores_std:.2f}")
            logging(f"Augmentation-Robust Score Mean: {ars_mean:.2f}%  Std: {ars_std:.2f}")

        if self.use_dist:
            torch.distributed.destroy_process_group()

        return {
            "with_aug_scores_mean": with_aug_scores_mean,
            "with_aug_scores_std": with_aug_scores_std,
            "without_aug_scores_mean": without_aug_scores_mean,
            "without_aug_scores_std": without_aug_scores_std,
            "augmentation_robust_score_mean": ars_mean,
            "augmentation_robust_score_std": ars_std
        }