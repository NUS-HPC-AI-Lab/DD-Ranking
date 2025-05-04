import time
import torch
import timm
import math
import datetime
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
from collections import OrderedDict
from .meter import MetricLogger, accuracy
from .misc import reduce_across_processes


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
    },
    "TinyImageNet-ConvNet-4-BN": {
        "optimizer": "sgd",
        "lr_scheduler": "step",
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "num_epochs": 100,
        "batch_size": 512,
        "lr": 0.01,
        "step_size": 50,
        "gamma": 0.1
    }
}


def default_augmentation(images):    
    return images

def get_optimizer(optimizer_name, model, lr, weight_decay=0.0005, momentum=0.9):
    if optimizer_name == 'sgd':
        return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} not implemented")

def get_lr_scheduler(lr_scheduler_name, optimizer, num_epochs=None, step_size=None):
    if lr_scheduler_name == 'step':
        assert step_size is not None, "step_size must be provided for step scheduler"
        return StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif lr_scheduler_name == 'cosine':
        assert num_epochs is not None, "num_epochs must be provided for cosine scheduler"
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif lr_scheduler_name == 'lambda_cos':
        assert num_epochs is not None, "num_epochs must be provided for lambda cosine scheduler"
        return LambdaLR(optimizer, lambda step: 0.5 * (1.0 + math.cos(math.pi * step / num_epochs / 2))
            if step <= num_epochs
            else 0,
            last_epoch=-1,
        )
    elif lr_scheduler_name == 'lambda_step':
        assert num_epochs is not None, "num_epochs must be provided for lambda step scheduler"
        return LambdaLR(optimizer, lambda step: (1.0 - step / num_epochs) if step <= num_epochs else 0, last_epoch=-1)
    else:
        raise NotImplementedError(f"LR Scheduler {lr_scheduler_name} not implemented")

# modified from pytorch-image-models/train.py
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
    logging=False,
    log_interval=10,
    device='cuda',
):

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    update_time_m = timm.utils.AverageMeter()
    data_time_m = timm.utils.AverageMeter()
    losses_m = timm.utils.AverageMeter()

    stu_model.train()
    if tea_models is not None:
        for tea_model in tea_models:
            tea_model.eval()

    if torch.distributed.is_initialized():
        loader.sampler.set_epoch(epoch)

    accum_steps = grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    if aug_func is None:
        aug_func = default_augmentation

    data_start_time = update_start_time = time.time()
    update_sample_count = 0
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        if class_map is not None:
            target = torch.tensor([class_map[target[i].item()] for i in range(len(target))], dtype=target.dtype, device=target.device)
        
        input, target = input.to(device), target.to(device)
        input = aug_func(input)
        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            stu_output = stu_model(input)
            if soft_label_mode == 'M':
                with torch.no_grad():
                    tea_outputs = [tea_model(input) for tea_model in tea_models]
                    tea_output = torch.stack(tea_outputs, dim=0).mean(dim=0)
                loss = loss_fn(stu_output, tea_output)
            else:
                loss = loss_fn(stu_output, target)
            if accum_steps > 1:
                loss /= accum_steps
            return loss

        def _backward(_loss):
            _loss.backward()
            if need_update:
                optimizer.step()
        
        optimizer.zero_grad()
        loss = _forward()
        _backward(loss)

        losses_m.update(loss.item() * accum_steps, input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        if logging and rank == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            loss_avg, loss_now = losses_m.avg, losses_m.val
            print(
                f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  '
                f'Loss: {loss_now:#.3g} ({loss_avg:#.3g})  '
                f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                f'LR: {lr:.3e}  '
                f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
            )

        update_sample_count = 0
        data_start_time = time.time()
    
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
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
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
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()
    
    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg