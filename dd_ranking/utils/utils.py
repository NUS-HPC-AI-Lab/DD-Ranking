import os
import torch
import torch.nn as nn
import time
import timm
import numpy as np
import random
from tqdm import tqdm
from collections import OrderedDict
from torch import Tensor
from torchvision import transforms, datasets
from .networks import MLP, ConvNet, LeNet, AlexNet, VGG, ResNet, BasicBlock, Bottleneck
from .networks import VGG11, VGG11_Tiny, VGG11BN, ResNet18, ResNet18_Tiny, ResNet18BN, ResNet18BN_Tiny, ResNet18BN_AP, ResNet18_AP


def set_seed():
    seed = int(time.time() * 1000) % 1000000
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

################################################################################ dataset utils ################################################################################
class Config:
    imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]

    imagemeow = [281, 282, 283, 284, 285, 291, 292, 290, 289, 287]

    imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]

    imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]

    imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]

    dict = {
        "ImageNette": imagenette,
        "ImageWoof": imagewoof,
        "ImageFruit": imagefruit,
        "ImageYellow": imageyellow,
        "ImageMeow": imagemeow,
        "ImageSquawk": imagesquawk
    }

config = Config()


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, images: Tensor, labels: Tensor):
        self.images = images
        self.labels = labels

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.images)


def get_dataset(dataset, data_path, im_size):
    class_map_inv = None

    if dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32) if not im_size else im_size
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_map = {x: x for x in range(num_classes)}

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32) if not im_size else im_size
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_map = {x: x for x in range(num_classes)}

    elif dataset == 'Tiny':
        channel = 3
        im_size = (64, 64) if not im_size else im_size
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)  # no augmentation
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)
        class_map = {x: x for x in range(num_classes)}

    elif dataset in ['ImageNette', 'ImageWoof', 'ImageMeow', 'ImageSquawk', 'ImageFruit', 'ImageYellow']:
        channel = 3
        im_size = (128, 128) if not im_size else im_size
        num_classes = 10

        config.img_net_classes = config.dict[dataset]

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.Resize(im_size),
            transforms.CenterCrop(im_size)
        ])

        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)
        dst_train = torch.utils.data.Subset(dst_train, np.squeeze(np.argwhere(np.isin(dst_train.targets, config.img_net_classes))))
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)
        dst_test = torch.utils.data.Subset(dst_test, np.squeeze(np.argwhere(np.isin(dst_test.targets, config.img_net_classes))))
        for c in range(len(config.img_net_classes)):
            dst_test.dataset.targets[dst_test.dataset.targets == config.img_net_classes[c]] = c
            dst_train.dataset.targets[dst_train.dataset.targets == config.img_net_classes[c]] = c
        
        class_map = {x: i for i, x in enumerate(config.img_net_classes)}
        class_map_inv = {i: x for i, x in enumerate(config.img_net_classes)}

    elif dataset == 'ImageNet1K':
        channel = 3
        im_size = (64, 64)
        num_classes = 1000
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.Resize(im_size),
            transforms.CenterCrop(im_size)
        ])

        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)

        class_map = {x: i for i, x in enumerate(range(num_classes))}
        class_map_inv = {i: x for i, x in enumerate(range(num_classes))}
    
    return channel, im_size, num_classes, dst_train, dst_test, class_map, class_map_inv


def get_random_images(images_all, labels_all, class_indices, n_images_per_class):
    all_selected_indices = []
    num_classes = len(class_indices)
    for c in range(num_classes):
        idx_shuffle = np.random.permutation(class_indices[c])[:n_images_per_class]
        all_selected_indices.extend(idx_shuffle)
    selected_images = images_all[all_selected_indices]
    selected_labels = labels_all[all_selected_indices]
    assert len(selected_images) == num_classes * n_images_per_class
    return selected_images, selected_labels

################################################################################ model utils ################################################################################

def parse_model_name(model_name):
    try:
        depth = int(model_name.split("-")[1])
        if "BN" in model_name and len(model_name.split("-")) > 2 and model_name.split("-")[2] == "BN":
            batchnorm = True
        else:
            batchnorm = False
    except:
        raise ValueError("Model name must be in the format of <model_name>-<depth>-[<batchnorm>]")
    return depth, batchnorm
        

def get_convnet(model_name, im_size, channel, num_classes, net_depth, net_norm, pretrained=False, model_path=None):
    print(f"Creating {model_name} with depth={net_depth}, norm={net_norm}")
    model = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth,
                    net_act='relu', net_norm=net_norm, net_pooling='avgpooling', im_size=im_size)
    if pretrained:
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    return model

def get_mlp(model_name, im_size, channel, num_classes, pretrained=False, model_path=None):
    print(f"Creating {model_name} with channel={channel}, num_classes={num_classes}")
    model = MLP(channel=channel, num_classes=num_classes, res=im_size[0])
    if pretrained:
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    return model

def get_lenet(model_name, im_size, channel, num_classes, pretrained=False, model_path=None):
    print(f"Creating {model_name} with channel={channel}, num_classes={num_classes}")
    model = LeNet(channel=channel, num_classes=num_classes, res=im_size[0])
    if pretrained:
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    return model

def get_alexnet(model_name, im_size, channel, num_classes, use_torchvision=False, pretrained=False, model_path=None):
    print(f"Creating {model_name} with channel={channel}, num_classes={num_classes}")
    if use_torchvision:
        return torchvision.models.alexnet(num_classes=num_classes, pretrained=pretrained)
    else:
        model = AlexNet(channel=channel, num_classes=num_classes, res=im_size[0])
        if pretrained:
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        return model

def get_vgg(model_name, im_size, channel, num_classes, depth=11, batchnorm=False, use_torchvision=False, pretrained=False, model_path=None):
    print(f"Creating {model_name} with channel={channel}, num_classes={num_classes}")
    if use_torchvision:
        if depth == 11:
            if batchnorm:
                return torchvision.models.vgg11_bn(num_classes=num_classes, pretrained=pretrained)
            else:
                return torchvision.models.vgg11(num_classes=num_classes, pretrained=pretrained)
        elif depth == 13:
            if batchnorm:
                return torchvision.models.vgg13_bn(num_classes=num_classes, pretrained=pretrained)
            else:
                return torchvision.models.vgg13(num_classes=num_classes, pretrained=pretrained)
        elif depth == 16:
            if batchnorm:
                return torchvision.models.vgg16_bn(num_classes=num_classes, pretrained=pretrained)
            else:
                return torchvision.models.vgg16(num_classes=num_classes, pretrained=pretrained)
        elif depth == 19:
            if batchnorm:
                return torchvision.models.vgg19_bn(num_classes=num_classes, pretrained=pretrained)
            else:
                return torchvision.models.vgg19(num_classes=num_classes, pretrained=pretrained)
    else:
        model = VGG(f'VGG{depth}', channel, num_classes, norm='batchnorm' if batchnorm else 'instancenorm', res=im_size[0])
        if pretrained:
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        
        return model
    

def get_resnet(model_name, im_size, channel, num_classes, depth=18, batchnorm=False, use_torchvision=False, pretrained=False, model_path=None):
    print(f"Creating {model_name} with channel={channel}, num_classes={num_classes}")
    if use_torchvision:
        if depth == 18:
            if batchnorm:
                return torchvision.models.resnet18_bn(num_classes=num_classes, pretrained=pretrained)
            else:
                return torchvision.models.resnet18(num_classes=num_classes, pretrained=pretrained)
        elif depth == 34:
            if batchnorm:
                return torchvision.models.resnet34_bn(num_classes=num_classes, pretrained=pretrained)
            else:
                return torchvision.models.resnet34(num_classes=num_classes, pretrained=pretrained)
        elif depth == 50:
            if batchnorm:
                return torchvision.models.resnet50_bn(num_classes=num_classes, pretrained=pretrained)
            else:
                return torchvision.models.resnet50(num_classes=num_classes, pretrained=pretrained)
    else:
        if depth == 18:
            model = ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm' if batchnorm else 'instancenorm', res=im_size[0])
        elif depth == 34:
            model = ResNet(BasicBlock, [3,4,6,3], channel=channel, num_classes=num_classes, norm='batchnorm' if batchnorm else 'instancenorm', res=im_size[0])
        elif depth == 50:
            model = ResNet(Bottleneck, [3,4,6,3], channel=channel, num_classes=num_classes, norm='batchnorm' if batchnorm else 'instancenorm', res=im_size[0])
        if pretrained:
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        
        return model


def get_other_models(model_name, channel, num_classes, im_size=(32, 32), pretrained=False, model_path=None):
    try:
        model = torchvision.models.get_model(model_name, pretrained=pretrained)
    except:
        model = timm.create_model(model_name, pretrained=pretrained)
    finally:
        raise ValueError(f"Model {model_name} not found")
    return model


def build_model(model_name: str, num_classes: int, im_size: tuple, pretrained: bool=False, model_path: str=None, device: str="cuda"):
    assert model_name is not None, "model name must be provided"
    depth, batchnorm = parse_model_name(model_name)
    if model_name.startswith("ConvNet"):
        model = get_convnet(model_name, channel=3, num_classes=num_classes, im_size=im_size, net_depth=depth, 
                            net_norm="instancenorm" if not batchnorm else "batchnorm", pretrained=pretrained, model_path=model_path)
    elif model_name.startswith("AlexNet"):
        model = get_alexnet(model_name, im_size=im_size, channel=3, num_classes=num_classes, pretrained=pretrained, model_path=model_path)
    elif model_name.startswith("ResNet"):
        model = get_resnet(model_name, im_size=im_size, channel=3, num_classes=num_classes, depth=depth, 
                            batchnorm=batchnorm, pretrained=pretrained, model_path=model_path)
    elif model_name.startswith("LeNet"):
        model = get_lenet(model_name, im_size=im_size, channel=3, num_classes=num_classes, pretrained=pretrained, model_path=model_path)
    elif model_name.startswith("MLP"):
        model = get_mlp(model_name, im_size=im_size, channel=3, num_classes=num_classes, pretrained=pretrained, model_path=model_path)
    elif model_name.startswith("VGG"):
        model = get_vgg(model_name, im_size=im_size, channel=3, num_classes=num_classes, depth=depth, batchnorm=batchnorm, pretrained=pretrained, model_path=model_path)
    else:
        model = get_other_models(model_name, num_classes=num_classes, im_size=im_size, pretrained=pretrained, model_path=model_path)
    
    model = model.to(device)
    return model


def get_pretrained_model_path(model_name, dataset, ipc):
    if dataset == 'CIFAR10':
        if ipc <= 10:
            return os.path.join(f"/home/wangkai/DD-Ranking/teacher_models/{dataset}", f"{model_name}", "ckpt_40.pt")
        elif ipc <= 100:
            return os.path.join(f"/home/wangkai/DD-Ranking/teacher_models/{dataset}", f"{model_name}", "ckpt_60.pt")
        elif ipc <= 1000:
            return os.path.join(f"/home/wangkai/DD-Ranking/teacher_models/{dataset}", f"{model_name}", "ckpt_80.pt")
    elif dataset == 'CIFAR100':
        if ipc <= 10:
            return os.path.join(f"./teacher_models/{dataset}", f"{model_name}", "ckpt_40.pt")
        elif ipc <= 100:
            return os.path.join(f"./teacher_models/{dataset}", f"{model_name}", "ckpt_80.pt")
    elif dataset == 'Tiny':
        if ipc <= 1:
            return os.path.join(f"./teacher_models/{dataset}", f"{model_name}", "ckpt_40.pt")
        elif ipc <= 10:
            return os.path.join(f"./teacher_models/{dataset}", f"{model_name}", "ckpt_60.pt")
        elif ipc <= 100:
            return os.path.join(f"./teacher_models/{dataset}", f"{model_name}", "ckpt_80.pt")
################################################################################ train and validate ################################################################################
def default_augmentation(images):
    # you can also add your own implementation here
    # img_size = images.shape[2]
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomRotation(degrees=15)
    # ])
    # images = transform(images)
    return images


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
    temperature=1.2,
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

        input = aug_func(input)
        input, target = input.to(device), target.to(device)

        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            stu_output = stu_model(input)
            if soft_label_mode == 'M':
                tea_output = tea_model(input)
                loss = loss_fn(stu_output, tea_output, temperature=temperature)
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

        if update_idx % log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            loss_avg, loss_now = losses_m.avg, losses_m.val

            if logging:
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
    aug_func=None,
    device=torch.device('cuda'),
    logging=False,
    log_interval=10
):
    batch_time_m = timm.utils.AverageMeter()
    top1_m = timm.utils.AverageMeter()
    top5_m = timm.utils.AverageMeter()

    model.eval()

    if aug_func is None:
        aug_func = default_augmentation

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = aug_func(input)
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


def train_one_epoch_dc(epoch, model, dataloader, loss_fn, optimizer, lr_scheduler=None, device='cuda'):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    model.train()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(device)
        lab = datum[1].long().to(device)
        n_b = lab.shape[0]

        output = model(img)
        loss = loss_fn(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item() * n_b
        acc_avg += acc
        num_exp += n_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step(epoch)

    loss_avg /= num_exp
    acc_avg /= num_exp
    print(f'Epoch {epoch}\tTrain Loss: {loss_avg:>7.3f}\tTrain Acc: {acc_avg:>7.3f}')

    return loss_avg, acc_avg

def validate_dc(
    stu_model,
    loader,
    loss_fn,
    device='cuda'
):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    stu_model = stu_model.to(device)

    stu_model.eval()

    for i_batch, datum in enumerate(loader):
        img = datum[0].float().to(device)
        lab = datum[1].long().to(device)
        n_b = lab.shape[0]

        output = stu_model(img)
        loss = loss_fn(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

    loss_avg /= num_exp
    acc_avg /= num_exp
    print(f'Test Loss: {loss_avg:>7.3f}\tTest Acc: {acc_avg:>7.3f}')
    return loss_avg, acc_avg