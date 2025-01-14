import os
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import kornia as K
from tqdm import tqdm
from torch import Tensor


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
        image = self.images[index]
        label = self.labels[index]
        return image, label
    
    def __len__(self):
        return len(self.images)


def get_dataset(dataset, data_path, im_size, use_zca, custom_train_trans, custom_val_trans, device):
    class_map_inv = None

    if dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32) if not im_size else im_size
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        if not use_zca:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform if custom_train_trans is None else custom_train_trans)
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform if custom_val_trans is None else custom_val_trans)
        class_map = {x: x for x in range(num_classes)}

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32) if not im_size else im_size
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        if not use_zca:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform if custom_train_trans is None else custom_train_trans)
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform if custom_val_trans is None else custom_val_trans)
        class_map = {x: x for x in range(num_classes)}

    elif dataset == 'TinyImageNet':
        channel = 3
        im_size = (64, 64) if not im_size else im_size
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if not use_zca:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform if custom_train_trans is None else custom_train_trans)
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform if custom_val_trans is None else custom_val_trans)
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

        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform if custom_train_trans is None else custom_train_trans)
        dst_train = torch.utils.data.Subset(dst_train, np.squeeze(np.argwhere(np.isin(dst_train.targets, config.img_net_classes))))
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform if custom_val_trans is None else custom_val_trans)
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

        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform if custom_train_trans is None else custom_train_trans)
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform if custom_val_trans is None else custom_val_trans)

        class_map = {x: i for i, x in enumerate(range(num_classes))}
        class_map_inv = {i: x for i, x in enumerate(range(num_classes))}
    
    if use_zca:
        images, labels = [], []
        for i in tqdm(range(len(dst_train))):
            im, lab = dst_train[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device='cpu')
        zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
        zca.fit(images)
        zca_images = zca(images).to("cpu")
        dst_train = TensorDataset(zca_images, labels)

        images, labels = [], []
        for i in tqdm(range(len(dst_test))):
            im, lab = dst_test[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device='cpu')

        zca_images = zca(images).to("cpu")
        dst_test = TensorDataset(zca_images, labels)

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