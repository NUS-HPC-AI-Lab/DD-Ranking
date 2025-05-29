import time
import torch
import timm
import math
import warnings
import datetime
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
from collections import OrderedDict
from .meter import MetricLogger, SmoothedValue, accuracy
from .misc import reduce_across_processes, is_dist_avail_and_initialized
from ..loss import MSEGTLoss


REAL_DATA_ACC_CACHE = {
    "ImageNet1K-ResNet-18-BN": 56.5,
    "TinyImageNet-ResNet-18-BN": 46.4
}

REAL_DATA_TRAINING_CONFIG = {
    "ImageNet1K-ResNet-18-BN": {
        "optimizer": "sgd",
        "lr_scheduler": "step",
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "num_epochs": 90,
        "batch_size": 512,
        "lr": 0.1,
        "step_size": 30,
        "gamma": 0.1
    },
    "TinyImageNet-ResNet-18-BN": {
        "optimizer": "adamw",
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "lr": 0.01,
        "num_epochs": 100,
        "batch_size": 512,
        "step_size": 0,
        "gamma": 0,
        "momentum": (0.9, 0.999)
    },
    "TinyImageNet-ConvNet-4-BN": {
        "optimizer": "sgd",
        "lr_scheduler": "step",
        "weight_decay": 0.0005,
        "momentum": 0.9,
        "num_epochs": 100,
        "batch_size": 512,
        "lr": 0.01,
        "step_size": 50,
        "gamma": 0.1
    },
    "CIFAR10-ConvNet-3": {
        "optimizer": "sgd",
        "lr_scheduler": "step",
        "weight_decay": 0.0005,
        "momentum": 0.9,
        "num_epochs": 200,
        "batch_size": 512,
        "lr": 0.01,
        "step_size": 100,
        "gamma": 0.1
    },
    "CIFAR10-ConvNet-3-BN": {
        "optimizer": "sgd",
        "lr_scheduler": "step",
        "weight_decay": 0.0005,
        "momentum": 0.9,
        "num_epochs": 200,
        "batch_size": 512,
        "lr": 0.01,
        "step_size": 100,
        "gamma": 0.1
    },
    "CIFAR100-ConvNet-3": {
        "optimizer": "sgd",
        "lr_scheduler": "step",
        "weight_decay": 0.0005,
        "momentum": 0.9,
        "num_epochs": 200,
        "batch_size": 512,
        "lr": 0.01,
        "step_size": 100,
        "gamma": 0.1
    },
    "CIFAR100-ConvNet-3-BN": {
        "optimizer": "sgd",
        "lr_scheduler": "step",
        "weight_decay": 0.0005,
        "momentum": 0.9,
        "num_epochs": 200,
        "batch_size": 512,
        "lr": 0.01,
        "step_size": 100,
        "gamma": 0.1
    }
}


def default_augmentation(images):    
    return images

def get_optimizer(optimizer_name, model, lr, weight_decay=0.0005, momentum=0.9):
    if optimizer_name == 'sgd':
        return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=momentum if isinstance(momentum, tuple) else (0.9, 0.999))
    elif optimizer_name == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=momentum if isinstance(momentum, tuple) else (0.9, 0.999))
    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} not implemented")

def get_lr_scheduler(lr_scheduler_name, optimizer, num_epochs=None, step_size=None, gamma=None):
    if lr_scheduler_name == 'step':
        assert step_size is not None, "step_size must be provided for step scheduler"
        return StepLR(optimizer, step_size=step_size, gamma=gamma if gamma is not None else 0.1)
    elif lr_scheduler_name == 'cosineannealing':
        assert num_epochs is not None, "num_epochs must be provided for cosine scheduler"
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif lr_scheduler_name == 'cosine':
        assert num_epochs is not None, "num_epochs must be provided for lambda cosine scheduler"
        return LambdaLR(optimizer, lambda step: 0.5 * (1.0 + math.cos(math.pi * step / num_epochs / 2))
            if step <= num_epochs
            else 0,
            last_epoch=-1,
        )
    else:
        raise NotImplementedError(f"LR Scheduler {lr_scheduler_name} not implemented")

def train_one_epoch(
    epoch,
    stu_model,
    loader,
    loss_fn,
    optimizer,
    soft_label_mode='S',
    aug_func=None,
    tea_models=None,
    lr_scheduler=None,
    class_map=None,
    grad_accum_steps=1,
    log_interval=500,
    device='cuda'
):

    stu_model.train()
    if tea_models is not None:
        for tea_model in tea_models:
            tea_model.eval()

    if is_dist_avail_and_initialized():
        loader.sampler.set_epoch(epoch)
    
    if aug_func is None:
        aug_func = default_augmentation

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    
    accumulated_loss = 0.0
    accum_step = 0
    
    for i, (images, targets) in enumerate(metric_logger.log_every(loader, log_interval, header)):
        start_time = time.time()

        if class_map is not None:
            targets = torch.tensor([class_map[targets[i].item()] for i in range(len(targets))], dtype=targets.dtype, device=targets.device)

        images, targets = images.to(device), targets.to(device)
        images = aug_func(images)

        raw_targets = targets.clone()
        if soft_label_mode == 'M':
            tea_outputs = [tea_model(images) for tea_model in tea_models]
            tea_output = torch.stack(tea_outputs, dim=0).mean(dim=0)
            targets = tea_output

        output = stu_model(images)

        if isinstance(loss_fn, MSEGTLoss):
            loss = loss_fn(output, targets, raw_targets)
        else:
            loss = loss_fn(output, targets)
        
        loss = loss / grad_accum_steps
        accumulated_loss += loss.item()
        
        loss.backward()
        
        accum_step += 1
        if accum_step == grad_accum_steps:
            optimizer.step()
            optimizer.zero_grad()
            accum_step = 0
            
            metric_logger.update(loss=accumulated_loss, lr=round(optimizer.param_groups[0]["lr"], 8))
            accumulated_loss = 0.0

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        batch_size = images.shape[0]
        
        if accum_step == 0:
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(round(batch_size / (time.time() - start_time), 2))
    
    if accum_step > 0:
        optimizer.step()
        optimizer.zero_grad()
    
    if lr_scheduler is not None:
        lr_scheduler.step()


def validate(
    model,
    loader,
    device='cuda',
    class_map=None,
    log_interval=100,
    topk=(1, 5)
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Test"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(loader, log_interval, header):
            if class_map is not None:
                target = torch.tensor([class_map[target[i].item()] for i in range(len(target))], dtype=target.dtype, device=target.device)
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    num_processed_samples = reduce_across_processes(num_processed_samples)
    if (
        hasattr(loader.dataset, "__len__")
        and len(loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        warnings.warn(
            f"It looks like the dataset has {len(loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()
    
    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg