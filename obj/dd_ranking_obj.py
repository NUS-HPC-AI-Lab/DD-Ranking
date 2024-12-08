import os
import torch
from typing import List, Tensor
from torchvision import transforms, datasets


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, images: Tensor, labels: Tensor):
        self.images = images
        self.labels = labels

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.images)
    

class DD_Ranking_Objective:
    def __init__(self, data_path: str=None, images: Tensor=None, labels: Tensor=None, model_name: str=None, use_default_transform: bool=True, 
                 custom_transform: transforms.Compose=None):

        if data_path is not None:
            image_path = os.path.join(data_path, 'images.pt')
            label_path = os.path.join(data_path, 'labels.pt')
            if os.path.exists(image_path) and os.path.exists(label_path):
                dataset = self.load_data(image_path, label_path)
                self.dataset = dataset
            else:
                try:
                    dataset = datasets.ImageFolder(
                        root=data_path,
                        transform=self.default_transform() if use_default_transform else custom_transform
                    )
                    all_images = torch.stack([img for img, _ in dataset])
                    all_labels = torch.tensor([label for _, label in dataset])
                    self.dataset = TensorDataset(all_images, all_labels)
                except Exception as e:
                    raise FileNotFoundError(f"Image or label file not found in {data_path}")

        elif images is not None and labels is not None:
            self.dataset = TensorDataset(images, labels)
        else:
            raise ValueError("Either data_path or images and labels must be provided")
        
        self.model = self.build_model(model_name)

    def load_data(self, image_path: str, label_path: str):
        self.images = torch.load(image_path)
        self.labels = torch.load(label_path)
        dataset = TensorDataset(self.images, self.labels)
        return dataset

    def default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def build_model(self, model_name: str):
        if model_name is None:
            return None
        from .utils import get_network
        return get_network(model_name, channel=3, num_classes=10)  # Adjust channel and num_classes as needed

    def compute_metrics(self):
        pass


class Hard_Label_Objective(DD_Ranking_Objective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_metrics(self):
        pass  # Add implementation here
        