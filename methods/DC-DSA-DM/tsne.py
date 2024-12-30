from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from models import ConvNet
import torchvision as tv
from torchvision.models.resnet import BasicBlock
from resnet18 import MyResNet


def vis_tsne(datasets, net, args):
    # for batch_idx, (image, target) in enumerate(datasets):
    #     image, target = image.cuda(), target.cuda()
    #     logit = net(image).detach()
    #     if batch_idx == 0:
    #         feature_bank = image
    #         label_bank = target
    #         logit_bank = logit
    #     else:
    #         feature_bank = torch.cat((feature_bank, image))
    #         label_bank = torch.cat((label_bank, target))
    #         logit_bank = torch.cat((logit_bank, logit))
    
    for c in range(args.nclass):
        data = datasets[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, 3, 32, 32))
        
            
    # print(feature_bank.shape, label_bank.shape)  
    feature_bank = feature_bank.reshape(feature_bank.shape[0], -1) 
    logit_bank = logit_bank.reshape(logit_bank.shape[0], -1)
    feature_bank = feature_bank.cpu().numpy()
    label_bank = label_bank.cpu().numpy()
    logit_bank = logit_bank.cpu().numpy()
    

    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(logit_bank)
    # X_pca = PCA(n_components=2).fit_transform(digits.data)

    ckpt_dir="tsne"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    for i in range(10):	# 对每类的数据画上特定颜色的点
        index = (label_bank==i)
        plt.scatter(X_tsne[index, 0], X_tsne[index, 1],s=5, cmap=plt.cm.Spectral)
    plt.legend(["0", "1", "2", "3", "4", "5", "6","7", "8", "9"])
    # plt.subplot(122)
    #  plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target,label="PCA")
    # plt.legend()
    plt.savefig(os.path.join(ckpt_dir, 'tsne.png'), dpi=120)
    # plt.show()
    
def vis_tsne_syn(image, teacher, it, args):
    # ConvNet-3  
    # teacher = ConvNet(
    #         num_classes=args.nclass,
    #         net_norm="batch",
    #         net_act="relu",
    #         net_pooling="avgpooling",
    #         net_depth=3,
    #         net_width=128,
    #         channel=3,
    #         im_size=(32, 32),
    #     )

    # checkpoint = torch.load(
    #     f"./cifar10_conv3.pth", map_location="cpu"
    #     )
    # teacher.load_state_dict(checkpoint["model"])
    # teacher = teacher.to(args.device)
    teacher.eval()
    embed_teacher = teacher.module.embed if torch.cuda.device_count() > 1 else teacher.embed
    
    model = MyResNet(BasicBlock, [2, 2, 2, 2]) 
    resnet18 = tv.models.resnet18(pretrained=True)
    pretrained_dict = resnet18.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.to(args.device)
    model.eval()
    
    for c in range(args.nclass):
        img = image[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, 3, 32, 32)).cuda()
        logit = teacher(img)
        feature = embed_teacher(img)
        # resnet-18 logit
        # resnet-18 feature
        # logit2 = net(img).detach()
        # feature = model(img).detach()
        if c == 0:
            logit_bank = logit.detach()
            feature_bank = feature.detach()
            # logit_bank2 = logit.detach()
            # print(logit.shape)
        else:
            logit_bank = torch.cat((logit_bank, logit.detach()))    
            feature_bank = torch.cat((feature_bank, feature.detach()))    
            # logit_bank2 = torch.cat((logit_bank2, logit2.detach()))  
            
    # print(feature_bank.shape, label_bank.shape)
    logit_bank = logit_bank.reshape(logit_bank.shape[0], -1)
    logit_bank = logit_bank.cpu().numpy()
    # logit_bank2 = logit_bank2.reshape(logit_bank2.shape[0], -1)
    # logit_bank2 = logit_bank2.cpu().numpy()
    feature_bank = feature_bank.reshape(feature_bank.shape[0], -1) 
    feature_bank = feature_bank.cpu().numpy()
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(args.nclass)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)
    label_syn = label_syn.cpu().numpy()

    logit_tsne = TSNE(n_components=2,random_state=33).fit_transform(logit_bank)
    # logit_tsne2 = TSNE(n_components=2,random_state=33).fit_transform(logit_bank2)
    feature_tsne = TSNE(n_components=2,random_state=33).fit_transform(feature_bank)

    # ckpt_dir="tsne_CMI_conv"
    ckpt_dir="tsne_result_DSA_CMI_cifar10"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    for i in range(args.nclass):	# 对每类的数据画上特定颜色的点
        index = (label_syn==i)
        plt.scatter(feature_tsne[index, 0], feature_tsne[index, 1], s=5, cmap=plt.cm.Spectral)
    # plt.legend(["0", "1", "2", "3", "4", "5", "6","7", "8", "9"])
    plt.savefig(os.path.join(ckpt_dir, 'tsne_syn_DSA_feature_conv_%s.png'%(it)), dpi=120)
    plt.close()
    
    # for i in range(10):	# 对每类的数据画上特定颜色的点
    #     index = (label_syn==i)
    #     plt.scatter(logit_tsne2[index, 0], logit_tsne2[index, 1], s=5, cmap=plt.cm.Spectral)
    # plt.legend(["0", "1", "2", "3", "4", "5", "6","7", "8", "9"])
    # plt.savefig(os.path.join(ckpt_dir, 'tsne_syn_DC_logit_res_%s.png'%(it)), dpi=120)
    # plt.close()
    
    for i in range(args.nclass):	# 对每类的数据画上特定颜色的点
        index = (label_syn==i)
        plt.scatter(logit_tsne[index, 0], logit_tsne[index, 1], s=5, cmap=plt.cm.Spectral)
    # plt.legend(["0", "1", "2", "3", "4", "5", "6","7", "8", "9"])
    plt.savefig(os.path.join(ckpt_dir, 'tsne_syn_DSA_logit_conv_%s.png'%(it)), dpi=120)
    plt.close()