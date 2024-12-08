import os
import torch
from typing import List
from torch import Tensor
from torchvision import transforms, datasets
from utils import get_convnet, get_alexnet, get_resnet, get_lenet, get_mlp, get_vgg
from utils import TensorDataset


class DD_Ranking_Objective:
    def __init__(self, data_path: str=None, images: Tensor=None, num_classes: int=10, ipc: int=1, model_name: str=None, use_default_transform: bool=True, 
                 custom_transform: transforms.Compose=None):

        if data_path is not None:
            image_path = os.path.join(data_path, 'images.pt')

            assert os.path.exists(image_path), "Image file not found in {}".format(data_path)
            self.images = self.load_data(image_path)

        elif images is not None:
            self.images = images
        else:
            raise ValueError("Either data_path or images must be provided")
        
        assert len(self.images) == ipc * num_classes, "Number of images must be equal to ipc * num_classes"
        self.im_size = self.images[0].shape
        self.num_classes = num_classes
        self.ipc = ipc
        
        self.model = self.build_model(model_name)

    def load_data(self, image_path: str):
        return torch.load(image_path, map_location='cpu')
    
    def build_model(self, model_name: str):
        assert model_name is not None, "model name must be provided"
        if model_name.startswith("ConvNet"):
            return get_convnet(channel=3, num_classes=self.num_classes, im_size=self.im_size, 
                               net_width=128, net_depth=3, net_act="relu", net_norm="instancenorm", net_pooling="avgpooling") 
        
    def compute_metrics(self):
        pass


class Soft_Label_Objective(DD_Ranking_Objective):
    def __init__(self, soft_labels: Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_labels = soft_labels

    def compute_hard_label_metrics(self):
        hard_labels = torch.tensor([np.ones(self.ipc) * i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False).view(-1)
        hard_label_dataset = TensorDataset(self.images.detach().clone(), hard_labels)

    def compute_metrics(self):
        pass  # Add implementation here



if __name__ == "__main__":
    images = torch.randn(10, 3, 32, 32)
    labels = torch.randint(0, 10, (10,))
    obj = DD_Ranking_Objective(images=images, labels=labels, model_name="ConvNet")
    print(obj.dataset.images.shape)