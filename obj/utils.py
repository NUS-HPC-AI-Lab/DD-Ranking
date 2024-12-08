import torch
import torch.nn as nn
import time
import torchvision
from networks import MLP, ConvNet, LeNet, AlexNet, VGG, ResNet, BasicBlock, Bottleneck


def get_convnet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling):
    print(f"Creating ConvNet with width={net_width}, depth={net_depth}, act={net_act}, norm={net_norm}, pooling={net_pooling}")
    return ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                   net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

def get_mlp(channel, num_classes):
    print(f"Creating MLP with channel={channel}, num_classes={num_classes}")
    return MLP(channel=channel, num_classes=num_classes)

def get_lenet(channel, num_classes):
    print(f"Creating LeNet with channel={channel}, num_classes={num_classes}")
    return LeNet(channel=channel, num_classes=num_classes)

def get_alexnet(channel, num_classes, use_torchvision=False):
    print(f"Creating AlexNet with channel={channel}, num_classes={num_classes}")
    if use_torchvision:
        return torchvision.models.alexnet(num_classes=num_classes, pretrained=False)
    else:
        return AlexNet(channel=channel, num_classes=num_classes)

def get_vgg(channel, num_classes, depth=11, batchnorm=False, use_torchvision=False):
    print(f"Creating VGG{depth} with channel={channel}, num_classes={num_classes}")
    if use_torchvision:
        if depth == 11:
            if batchnorm:
                return torchvision.models.vgg11_bn(num_classes=num_classes, pretrained=False)
            else:
                return torchvision.models.vgg11(num_classes=num_classes, pretrained=False)
        elif depth == 13:
            if batchnorm:
                return torchvision.models.vgg13_bn(num_classes=num_classes, pretrained=False)
            else:
                return torchvision.models.vgg13(num_classes=num_classes, pretrained=False)
        elif depth == 16:
            if batchnorm:
                return torchvision.models.vgg16_bn(num_classes=num_classes, pretrained=False)
            else:
                return torchvision.models.vgg16(num_classes=num_classes, pretrained=False)
        elif depth == 19:
            if batchnorm:
                return torchvision.models.vgg19_bn(num_classes=num_classes, pretrained=False)
            else:
                return torchvision.models.vgg19(num_classes=num_classes, pretrained=False)
    else:
        return VGG(f'VGG{depth}', channel, num_classes, norm='batchnorm' if batchnorm else 'instancenorm')
    

def get_resnet(channel, num_classes, depth=18, batchnorm=False, use_torchvision=False):
    print(f"Creating ResNet{depth} with channel={channel}, num_classes={num_classes}")
    if use_torchvision:
        if depth == 18:
            if batchnorm:
                return torchvision.models.resnet18_bn(num_classes=num_classes, pretrained=False)
            else:
                return torchvision.models.resnet18(num_classes=num_classes, pretrained=False)
        elif depth == 34:
            if batchnorm:
                return torchvision.models.resnet34_bn(num_classes=num_classes, pretrained=False)
            else:
                return torchvision.models.resnet34(num_classes=num_classes, pretrained=False)
        elif depth == 50:
            if batchnorm:
                return torchvision.models.resnet50_bn(num_classes=num_classes, pretrained=False)
            else:
                return torchvision.models.resnet50(num_classes=num_classes, pretrained=False)
    else:
        if depth == 18:
            return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm' if batchnorm else 'instancenorm')
        elif depth == 34:
            return ResNet(BasicBlock, [3,4,6,3], channel=channel, num_classes=num_classes, norm='batchnorm' if batchnorm else 'instancenorm')
        elif depth == 50:
            return ResNet(Bottleneck, [3,4,6,3], channel=channel, num_classes=num_classes, norm='batchnorm' if batchnorm else 'instancenorm')

def get_other_models(model_name, channel, num_classes, im_size=(32, 32), dist=True):
    pass

def get_network(model_name, channel, num_classes, im_size=(32, 32), dist=True):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    """ MLP """
    if model_name == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    if model_name == 'MLP_Tiny':
        net = MLP(channel=channel, num_classes=num_classes, res=64)
    
    """ LeNet """
    if model_name == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    if model_name == 'LeNet_Tiny':
        net = LeNet(channel=channel, num_classes=num_classes, res=64)
    if model_name == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    if model_name == 'AlexNet_Tiny':
        net = AlexNet(channel=channel, num_classes=num_classes, res=64)

    """ VGG """
    if model_name == 'VGG11':
        net = VGG11(channel=channel, num_classes=num_classes)
    if model_name == 'VGG11_Tiny':
        net = VGG11_Tiny(channel=channel, num_classes=num_classes)
    if model_name == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)

    """ ResNet """
    if model_name == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    if model_name == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    if model_name == 'ResNet18_AP':
        net = ResNet18_AP(channel=channel, num_classes=num_classes)
    if model_name == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)
    if model_name == 'ResNet18_Tiny':
        net = ResNet18_Tiny(channel=channel, num_classes=num_classes)
    if model_name == 'ResNet18BN_Tiny':
        net = ResNet18BN_Tiny(channel=channel, num_classes=num_classes)
    
    """ ConvNet """
    if model_name == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    if model_name == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    if model_name == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    if model_name == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    if model_name == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    if model_name == 'ConvNetD4BN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act,
                      net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)
    if model_name == 'ConvNetD5':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=5, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    if model_name == 'ConvNetD6':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=6, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    if model_name == 'ConvNetD7':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=7, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    if model_name == 'ConvNetD8':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=8, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    if model_name == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling)
    if model_name == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling)
    if model_name == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling)
    if model_name == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling)
    if model_name == 'ConvNetW512':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=512, net_depth=net_depth, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling)
    if model_name == 'ConvNetW1024':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=1024, net_depth=net_depth, net_act=net_act,
                      net_norm=net_norm, net_pooling=net_pooling)

    if model_name == "ConvNetKIP":
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=1024, net_depth=net_depth, net_act=net_act,
                      net_norm="none", net_pooling=net_pooling)

    if model_name == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling)
    if model_name == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act='relu', net_norm=net_norm, net_pooling=net_pooling)
    if model_name == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling)

    if model_name == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='none', net_pooling=net_pooling)
    if model_name == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling)
    if model_name == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='layernorm', net_pooling=net_pooling)
    if model_name == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling)
    if model_name == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling)

    if model_name == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling='none')
    if model_name == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling='maxpooling')
    if model_name == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                      net_act=net_act, net_norm=net_norm, net_pooling='avgpooling')


    else:
        net = None
        exit('DC error: unknown model')

    if dist:
        gpu_num = torch.cuda.device_count()
        if gpu_num > 0:
            device = 'cuda'
            if gpu_num > 1:
                net = nn.DataParallel(net)
        else:
            device = 'cpu'
        net = net.to(device)

    return net