import os
import torch
import torch.nn.functional as F
from typing import List
from torch import Tensor
from torchvision import transforms, datasets
from utils import parse_model_name, get_convnet, get_alexnet, get_resnet, get_lenet, get_mlp, get_vgg, get_other_model
from utils import TensorDataset, get_random_images, get_dataset
from hard_label import compute_hard_label_metrics


class DD_Ranking_Objective:
    def __init__(self, dataset: str="CIFAR10", real_data_path: str=None, syn_data_path: str=None, images: Tensor=None, num_classes: int=10, 
                 ipc: int=1, model_name: str=None, use_default_transform: bool=True, 
                 custom_transform: transforms.Compose=None, device=torch.device('cuda')):
        channel, im_size, num_classes, dst_train, dst_test, class_map, class_map_inv = get_dataset(dataset, real_data_path)
        self.images_train, self.labels_train, self.class_indices_train = self.load_real_data(dst_train, class_map, num_classes)
        self.images_test, self.labels_test, self.class_indices_test = self.load_real_data(dst_test, class_map, num_classes)

        if syn_data_path is not None:
            image_path = os.path.join(syn_data_path, 'images.pt')
            assert os.path.exists(image_path), "Image file not found in {}".format(data_path)
            self.syn_images = self.load_syn_data(image_path)

        elif images is not None:
            self.syn_images = images
        else:
            raise ValueError("Either data_path or images must be provided")
        
        assert len(self.syn_images) == ipc * num_classes, "Number of images must be equal to ipc * num_classes"
        self.im_size = self.syn_images[0].shape
        self.num_classes = num_classes
        self.ipc = ipc
        
        self.model = self.build_model(model_name)
        self.device = device
    
    def load_real_data(self, dataset, class_map, num_classes):
        images_all = []
        labels_all = []
        class_indices = [[] for c in range(num_classes)]
        for i, (image, label) in enumerate(dataset):
            images_all.append(torch.unsqueeze(image, 0))
            labels_all.append(class_map[label].item())
        images_all = torch.cat(images_all, dim=0)
        labels_all = torch.tensor(labels_all)
        for i, label in enumerate(labels_all):
            class_indices.append(i)
        
        return images_all, labels_all, class_indices
    
    def load_syn_data(self, image_path: str):
        return torch.load(image_path, map_location='cpu')
    
    def build_model(self, model_name: str):
        assert model_name is not None, "model name must be provided"
        depth, batchnorm = parse_model_name(model_name)
        if model_name.startswith("ConvNet"):
            return get_convnet(channel=3, num_classes=self.num_classes, im_size=self.im_size, 
                               net_width=128, net_depth=depth, net_act="relu", net_norm="instancenorm" if not batchnorm else "batchnorm",
                               net_pooling="avgpooling")
        elif model_name.startswith("AlexNet"):
            return get_alexnet(im_size=self.im_size, channel=3, num_classes=self.num_classes)
        elif model_name.startswith("ResNet"):
            return get_resnet(im_size=self.im_size, channel=3, num_classes=self.num_classes, depth=depth, batchnorm=batchnorm)
        elif model_name.startswith("LeNet"):
            return get_lenet(im_size=self.im_size, channel=3, num_classes=self.num_classes)
        elif model_name.startswith("MLP"):
            return get_mlp(im_size=self.im_size, channel=3, num_classes=self.num_classes)
        elif model_name.startswith("VGG"):
            return get_vgg(im_size=self.im_size, channel=3, num_classes=self.num_classes, depth=depth, batchnorm=batchnorm)
        else:
            return get_other_model(model_name, num_classes=self.num_classes, im_size=self.im_size)
        
    def compute_metrics(self):
        pass


class Soft_Label_Objective(DD_Ranking_Objective):
    def __init__(self, soft_labels: Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_labels = soft_labels
        self.soft_label_dataset = TensorDataset(self.syn_images.detach().clone(), self.soft_labels.detach().clone())
        self.num_epochs = 100
        self.lr = 0.01
        self.batch_size = 256

    def compute_syn_data_hard_label_metrics(self):
        return compute_hard_label_metrics(self.syn_images, self.model, self.ipc, self.num_classes, device=self.device)

    @staticmethod
    def SoftCrossEntropy(inputs, target):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch_size = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch_size
        return loss

    def compute_syn_data_soft_label_metrics(self):
        train_loader = DataLoader(self.soft_label_dataset, batch_size=self.batch_size, shuffle=True)
        loss_fn = self.SoftCrossEntropy
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))

        best_acc1 = 0
        best_epoch = 0
        for epoch in range(self.num_epochs):
            train_one_epoch(self.model, train_loader, loss_fn, optimizer, lr_scheduler=lr_scheduler, device=self.device)
            metric = validate(self.model, self.ipc, self.num_classes, device=self.device)
            if metric['acc1'] > best_acc1:
                best_acc1 = metric['acc1']
                best_epoch = epoch
        
        self.syn_data_soft_label = best_acc1

    def compute_random_data_soft_label_metrics(self):
        random_images = get_random_images(self.images_train, self.class_indices_train, target_class, target_num)

    def compute_full_data_hard_label_metrics(self):
        pass




if __name__ == "__main__":
    images = torch.randn(10, 3, 32, 32)
    obj = DD_Ranking_Objective(images=images, model_name="ConvNet")
    print(obj.syn_images.shape)