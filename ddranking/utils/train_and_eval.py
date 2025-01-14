import time
import torch
import timm
import math
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
from collections import OrderedDict


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

def get_lr_scheduler(lr_scheduler_name, optimizer, num_epochs):
    if lr_scheduler_name == 'step':
        return StepLR(optimizer, step_size=num_epochs // 2 + 1, gamma=0.1)
    elif lr_scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif lr_scheduler_name == 'lambda_cos':
        return LambdaLR(optimizer, lambda step: 0.5 * (1.0 + math.cos(math.pi * step / num_epochs / 2))
            if step <= num_epochs
            else 0,
            last_epoch=-1,
        )
    elif lr_scheduler_name == 'lambda_step':
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
    tea_model=None,
    lr_scheduler=None,
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
    if tea_model is not None:
        tea_model.eval()

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

        input, target = input.to(device), target.to(device)
        input = aug_func(input)
        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            stu_output = stu_model(input)
            if soft_label_mode == 'M':
                with torch.no_grad():
                    tea_output = tea_model(input)
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

        if logging:
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

        if lr_scheduler is not None:
            lr_scheduler.step(epoch)

        update_sample_count = 0
        data_start_time = time.time()

    loss_avg = losses_m.avg
    return loss_avg


def validate(
    model,
    loader,
    device='cuda',
    logging=False,
    log_interval=10
):
    batch_time_m = timm.utils.AverageMeter()
    top1_m = timm.utils.AverageMeter()
    top5_m = timm.utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

                # augmentation reduction
                reduce_factor = 1
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]
    
            acc1, acc5 = timm.utils.accuracy(output, target, topk=(1, 5))

            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if logging and (last_batch or batch_idx % log_interval == 0):
                print(
                    f'Test: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                    f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})'
                )

    metrics = OrderedDict([('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics