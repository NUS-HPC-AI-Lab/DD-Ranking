import torch
import torchvision
import timm
import os
from .networks import ConvNet, MLP, LeNet, AlexNet, VGG, ResNet, BasicBlock, Bottleneck


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
                model = torchvision.models.vgg11_bn(num_classes=num_classes, pretrained=False)
            else:
                model = torchvision.models.vgg11(num_classes=num_classes, pretrained=False)
        elif depth == 13:
            if batchnorm:
                model = torchvision.models.vgg13_bn(num_classes=num_classes, pretrained=False)
            else:
                model = torchvision.models.vgg13(num_classes=num_classes, pretrained=False)
        elif depth == 16:
            if batchnorm:
                model = torchvision.models.vgg16_bn(num_classes=num_classes, pretrained=False)
            else:
                model = torchvision.models.vgg16(num_classes=num_classes, pretrained=False)
        elif depth == 19:
            if batchnorm:
                model = torchvision.models.vgg19_bn(num_classes=num_classes, pretrained=False)
            else:
                model = torchvision.models.vgg19(num_classes=num_classes, pretrained=False)
    else:
        model = VGG(f'VGG{depth}', channel, num_classes, norm='batchnorm' if batchnorm else 'instancenorm', res=im_size[0])
    
    if pretrained:
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))    
    return model
    

def get_resnet(model_name, im_size, channel, num_classes, depth=18, batchnorm=False, use_torchvision=False, pretrained=False, model_path=None):
    print(f"Creating {model_name} with channel={channel}, num_classes={num_classes}")
    if use_torchvision:
        print(f"ResNet in torchvision uses batchnorm by default.")
        if depth == 18:
            model = torchvision.models.resnet18(num_classes=num_classes, pretrained=False)
        elif depth == 34:
            model = torchvision.models.resnet34(num_classes=num_classes, pretrained=False)
        elif depth == 50:
            model = torchvision.models.resnet50(num_classes=num_classes, pretrained=False)
        if im_size == (64, 64) or im_size == (32, 32):
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
            model.maxpool = torch.nn.Identity()
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


def build_model(model_name: str, num_classes: int, im_size: tuple, pretrained: bool=False, model_path: str=None, use_torchvision: bool=False, device: str="cuda"):
    assert model_name is not None, "model name must be provided"
    depth, batchnorm = parse_model_name(model_name)
    if model_name.startswith("ConvNet"):
        model = get_convnet(model_name, channel=3, num_classes=num_classes, im_size=im_size, net_depth=depth, 
                            net_norm="instancenorm" if not batchnorm else "batchnorm", pretrained=pretrained, model_path=model_path)
    elif model_name.startswith("AlexNet"):
        model = get_alexnet(model_name, im_size=im_size, channel=3, num_classes=num_classes, pretrained=pretrained, 
                            use_torchvision=use_torchvision, model_path=model_path)
    elif model_name.startswith("ResNet"):
        model = get_resnet(model_name, im_size=im_size, channel=3, num_classes=num_classes, depth=depth, use_torchvision=use_torchvision,
                            batchnorm=batchnorm, pretrained=pretrained, model_path=model_path)
    elif model_name.startswith("LeNet"):
        model = get_lenet(model_name, im_size=im_size, channel=3, num_classes=num_classes, pretrained=pretrained, model_path=model_path)
    elif model_name.startswith("MLP"):
        model = get_mlp(model_name, im_size=im_size, channel=3, num_classes=num_classes, pretrained=pretrained, model_path=model_path)
    elif model_name.startswith("VGG"):
        model = get_vgg(model_name, im_size=im_size, channel=3, num_classes=num_classes, depth=depth, batchnorm=batchnorm, 
                        use_torchvision=use_torchvision, pretrained=pretrained, model_path=model_path)
    else:
        model = get_other_models(model_name, num_classes=num_classes, im_size=im_size, pretrained=pretrained, model_path=model_path)
    
    model = model.to(device)
    return model


def get_pretrained_model_path(teacher_dir, model_name, dataset, ipc):
    # if dataset == 'CIFAR10':
    #     if ipc <= 10:
    #         return os.path.join(teacher_dir, f"{dataset}", f"{model_name}", "ckpt_20.pt")
    #     elif ipc <= 100:
    #         return os.path.join(teacher_dir, f"{dataset}", f"{model_name}", "ckpt_40.pt")
    #     elif ipc <= 1000:
    #         return os.path.join(teacher_dir, f"{dataset}", f"{model_name}", "ckpt_80.pt")
    # elif dataset == 'CIFAR100':
    #     if ipc <= 1:
    #         return os.path.join(teacher_dir, f"{dataset}", f"{model_name}", "ckpt_20.pt")
    #     elif ipc <= 10:
    #         return os.path.join(teacher_dir, f"{dataset}", f"{model_name}", "ckpt_60.pt")
    #     elif ipc <= 100:
    #         return os.path.join(teacher_dir, f"{dataset}", f"{model_name}", "ckpt_100.pt")
    # elif dataset == 'TinyImageNet':
    #     if ipc <= 1:
    #         return os.path.join(teacher_dir, f"{dataset}", f"{model_name}", "ckpt_20.pt")
    #     elif ipc <= 10:
    #         return os.path.join(teacher_dir, f"{dataset}", f"{model_name}", "ckpt_60.pt")
    #     elif ipc <= 100:
    #         return os.path.join(teacher_dir, f"{dataset}", f"{model_name}", "ckpt_80.pt")
    return os.path.join(os.path.join(teacher_dir, f"{dataset}", f"{model_name}", "ckpt_best.pt"))