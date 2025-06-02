import os
import shutil
import torch
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import kornia as K
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Subset


class ImageNetSubsets:
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

imagenet_subsets = ImageNetSubsets()

class TensorDataset(torch.utils.data.Dataset):
    
    def __init__(self, images: Tensor, labels: Tensor):
        super(TensorDataset, self).__init__()
        self.images = images
        self.targets = labels
        self.imgs = [(image, label) for image, label in zip(images, labels)]

    def __getitem__(self, index: int):
        image = self.images[index]
        label = self.targets[index]
        return image, label
    
    def __len__(self):
        return len(self.images)


def get_dataset(dataset, data_path, im_size, use_zca, custom_val_trans, device):
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

        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        dst_test_real = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        dst_test_syn = datasets.CIFAR10(data_path, train=False, download=True, transform=transform if custom_val_trans is None else custom_val_trans)
        class_map = {x: x for x in range(num_classes)}
        class_map_inv = {x: x for x in range(num_classes)}
        
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

        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
        dst_test_real = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        dst_test_syn = datasets.CIFAR100(data_path, train=False, download=True, transform=transform if custom_val_trans is None else custom_val_trans)
        class_map = {x: x for x in range(num_classes)}
        class_map_inv = {x: x for x in range(num_classes)}

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
        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)
        dst_test_real = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)
        dst_test_syn = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform if custom_val_trans is None else custom_val_trans)
        class_map = {x: x for x in range(num_classes)}
        class_map_inv = {x: x for x in range(num_classes)}

    elif dataset in ['ImageNette', 'ImageWoof', 'ImageMeow', 'ImageSquawk', 'ImageFruit', 'ImageYellow']:
        channel = 3
        im_size = (128, 128) if not im_size else im_size
        num_classes = 10

        imagenet_subsets.img_net_classes = imagenet_subsets.dict[dataset]

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.Resize(im_size),
            transforms.CenterCrop(im_size)
        ])

        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)
        dst_train = torch.utils.data.Subset(dst_train, np.squeeze(np.argwhere(np.isin(dst_train.targets, imagenet_subsets.img_net_classes))))

        dst_test_real = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)
        dst_test_syn = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform if custom_val_trans is None else custom_val_trans)

        dst_test_real = torch.utils.data.Subset(dst_test_real, np.squeeze(np.argwhere(np.isin(dst_test_real.targets, imagenet_subsets.img_net_classes))))
        dst_test_syn = torch.utils.data.Subset(dst_test_syn, np.squeeze(np.argwhere(np.isin(dst_test_syn.targets, imagenet_subsets.img_net_classes))))

        for c in range(len(imagenet_subsets.img_net_classes)):
            dst_test_real.dataset.targets[dst_test_real.dataset.targets == imagenet_subsets.img_net_classes[c]] = c
            dst_test_syn.dataset.targets[dst_test_syn.dataset.targets == imagenet_subsets.img_net_classes[c]] = c
            dst_train.dataset.targets[dst_train.dataset.targets == imagenet_subsets.img_net_classes[c]] = c
        
        class_map = {x: i for i, x in enumerate(imagenet_subsets.img_net_classes)}
        class_map_inv = {i: x for i, x in enumerate(imagenet_subsets.img_net_classes)}

    elif dataset == 'ImageNet1K':
        channel = 3
        im_size = (224, 224) if not im_size else im_size
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
        dst_test_real = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)
        dst_test_syn = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform if custom_val_trans is None else custom_val_trans)

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
        for i in tqdm(range(len(dst_test_real))):
            im, lab = dst_test_real[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device='cpu')
        zca_images = zca(images).to("cpu")
        dst_test_real = TensorDataset(zca_images, labels)

        images, labels = [], []
        for i in tqdm(range(len(dst_test_syn))):
            im, lab = dst_test_syn[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device='cpu')
        zca_images = zca(images).to("cpu")
        dst_test_syn = TensorDataset(zca_images, labels)

    return channel, im_size, mean, std,num_classes, dst_train, dst_test_real, dst_test_syn, class_map, class_map_inv


def get_random_data_tensors(dataset_name, dataset, class_to_indices, n_images_per_class, class_map, eval_iter, saved_path=None):
    if saved_path is None:
        saved_path = f"./random_data/{dataset_name}_IPC{n_images_per_class}/iter{eval_iter}"
    os.makedirs(saved_path, exist_ok=True)

    subset_indices = []
    for indices in class_to_indices:
        subset_indices.extend(random.sample(indices, n_images_per_class))
    subset_dataset = Subset(dataset, subset_indices)
    
    selected_images, selected_labels = [], []
    for i, (image, label) in enumerate(subset_dataset):
        selected_images.append(image)
        selected_labels.append(class_map[label])
    selected_images = torch.stack(selected_images, dim=0)
    selected_labels = torch.tensor(selected_labels, dtype=torch.long)
    if saved_path is not None:
        torch.save(selected_images, os.path.join(saved_path, f'rdm_data_iter{eval_iter}.pt'))
        torch.save(selected_labels, os.path.join(saved_path, f'rdm_labels_iter{eval_iter}.pt'))
    return selected_images.detach().cpu(), selected_labels.detach().cpu()


def get_random_data_path_from_cifar(dataset_name, dataset, class_to_indices, n_images_per_class, eval_iter, saved_path=None):

    if saved_path is None:
        saved_path = f"./random_data/{dataset_name}_IPC{n_images_per_class}/iter{eval_iter}"
    os.makedirs(saved_path, exist_ok=True)

    def denormalize(image_tensor):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        image_tensor = image_tensor * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        return image_tensor

    to_pil = transforms.ToPILImage()
    for class_idx, indices in enumerate(class_to_indices):
        os.makedirs(os.path.join(saved_path, f"{class_idx:05d}"), exist_ok=True)
        selected_indices = random.sample(indices, n_images_per_class)
        for i, index in enumerate(selected_indices):
            image_tensor, label = dataset[index]
            image_tensor = denormalize(image_tensor)
            image_tensor = torch.clamp(image_tensor, 0, 1)
            image_pil = to_pil(image_tensor)
            image_pil.save(os.path.join(saved_path, f"{class_idx:05d}", f"{i:05d}.jpg"))

    return saved_path   

def get_random_data_path(dataset_name, class_to_samples, n_images_per_class, eval_iter, saved_path=None):

    if saved_path is None:
        saved_path = f"./random_data/{dataset_name}_IPC{n_images_per_class}/iter{eval_iter}"
    os.makedirs(saved_path, exist_ok=True)

    for class_idx, samples in enumerate(class_to_samples):
        os.makedirs(os.path.join(saved_path, f"{class_idx:05d}"), exist_ok=True)
        selected_samples = random.sample(samples, n_images_per_class)
        for i, sample in enumerate(selected_samples):
            shutil.copy(sample, os.path.join(saved_path, f"{class_idx:05d}", f"{i:05d}.jpg"))

    return saved_path

        