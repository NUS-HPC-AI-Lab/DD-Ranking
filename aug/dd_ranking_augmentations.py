import os
import torch
from typing import Dict
from torch import Tensor
from torchvision import transforms
from utils import set_seed, Default_Augmentation, calculate_acc
import random
import utils
import kornia


class Augmentation:
    def __init__(self, images:Tensor=None, transform_params:Dict=None, model:torch.nn.Module=None):
        self.images=images
        self.transform_params=transform_params
        self.model=model
        self.transform=None
        
        
    def __call__(self):
        p = self.transform_params["transform"]["prob"]
        for i,img in enumerate(self.images):
            for t in self.transform.transforms:
                if p > torch.rand(1):
                    self.images[i]=t(img)
        return self.images
    
    def compute_metric(self,labels,distilled_dataset):
        
        set_seed(self.transform_params["seed"])
        
        q_d = calculate_acc(self.model, distilled_dataset, labels, Default_Augmentation)
        q_e = calculate_acc(self.model, distilled_dataset, labels, self.transform)
        p_f = calculate_acc(self.model, self.images, labels, Default_Augmentation)
        
        select = random.sample(range(self.images.shape[0]), self.transform_params["subset_size"])
        subset = self.images[select]
        r_e = calculate_acc(self.model, subset, labels, self.transform)
        
        return abs((p_f-r_e)/(q_d-q_e))
        
        
    
class DSA_Augmentation(Augmentation):
    def __init__(self, images = None, transform_type = None, transform_params = None):
        super().__init__(images, transform_type, transform_params)
        self.transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=transform_params["flip"]["prob"]),
        transforms.RandomAffine(
            degrees=transform_params["affine"]["degrees"],
            shear=transform_params["affine"]["shear"],
            interpolation=transform_params["affine"]["interpolation"]),
        transforms.RandomGrayscale(p=transform_params["gray"]["prob"]),
        transforms.ColorJitter(
        brightness=transform_params["colorjitter"]["brightness"],
        contrast=transform_params["colorjitter"]["contrast"],
        saturation=transform_params["colorjitter"]["saturation"],
        hue=transform_params["colorjitter"]["hue"])]
        )
    def __call__(self):
        return self.transform(images)
        
    
class ZCA_Whitening_Augmentation(Augmentation):
    def __init__(self, images = None, transform_params = None, model = None):
        super().__init__(images, transform_params, model)
        self.transform = kornia.enhance.ZCAWhitening()
        
    def __call__(self):
        return self.transform(self.images,include_fit=True)
        
        
class Mixup_Augmentation(Augmentation):
    def __init__(self, labels, images = None, transform_params = None, model = None):
        super().__init__(images, transform_params, model)
        self.labels=labels
        self.transform = kornia.augmentation.RandomMixUpV2(
        lambda_val = transform_params["mixup"]["lambda_range"],
        same_on_batch = transform_params["mixup"]["same_on_batch"],
        keepdim = transform_params["mixup"]["keepdim"],
        p = transform_params["transform"]["prob"])
        
    def __call__(self):
        return self.transform(self.images, self.labels)


class Cutmix_Augmentation(Augmentation):
    def __init__(self, labels, images = None, transform_params = None, model = None):
        super().__init__(images, transform_params, model)
        self.labels=labels
        self.transform = kornia.augmentation.RandomCutMixV2(
        num_mix = transform_params["cutmix"]["times"],
        cut_size = transform_params["cutmix"]["size"],
        same_on_batch = transform_params["cutmix"]["same_on_batch"],
        beta = transform_params["cutmix"]["beta"],
        keepdim = transform_params["cutmix"]["keep_dim"],
        p = transform_params["transform"]["prob"])
        
    def __call__(self):
        return self.transform(self.images, self.labels)
    
    
class Erasing_Augmentation(Augmentation):
    def __init__(self, images = None, transform_params = None, model = None):
        super().__init__(images, transform_params, model)
        self.transform = transforms.Compose([transforms.RandomErasing(
        p=self.transform_params["transform"]["prob"],     
        scale=self.transform_params["erasing"]["scale"],   
        ratio=self.transform_params["erasing"]["ratio"], 
        value=self.transform_params["erasing"]["value"],      
        inplace=False)])
        
        
        
class Guassian_Blur_Augmentation(Augmentation):
    def __init__(self, images = None, transform_params = None, model = None):
        super().__init__(images, transform_params, model)
        self.transform = transforms.Compose([transforms.GaussianBlur
        (kernel_size=self.transform_params["gb"]["kernel_size"],sigma=self.transform_params["gb"]["sigma"])])
        
        
class Crop_Augmentation(Augmentation):
    def __init__(self, images = None, transform_params = None, model = None):
        super().__init__(images, transform_params, model)
        self.transform = transforms.Compose([transforms.RandomCrop(self.transform_params["crop"]["size"])])
        
    def __call__(self):
        p = self.transform_params["transform"]["prob"]
        
        images = torch.zeros(self.images.shape[0], self.images.shape[1],
        self.transform_params["crop"]["size"], self.transform_params["crop"]["size"])
        
        for i,img in enumerate(self.images):
            for t in self.transform.transforms:
                if p > torch.rand(1):
                    images[i]=t(img)
        return images
        
        
if __name__ == "__main__":
    images = torch.randn(10, 3, 32, 32)
    labels = torch.randn(10)
    torch.nn.init.constant_(images, 1.0)
    # images[0] = torch.zeros_like(images[0]); images[4] = torch.zeros_like(images[0]); images[9] = torch.zeros_like(images[0]); 
    torch.nn.init.constant_(labels, 1.0)
    transform_params={
        "flip":{"prob":0.5},
        
        "affine":{"degrees":15,
                  "shear":20,
                  "interpolation":transforms.InterpolationMode.BILINEAR},
        
        "gray":{"prob":0.5},
        
        "colorjitter":{"brightness":0.03,
                       "contrast":0.01,
                       "saturation":0.03,
                       "hue":0.0},
        
        "transform":{"prob":0.5},
        
        "mixup":{"lambda_range":[0,1],
        "same_on_batch":True,
        "keepdim":True},
        
        "erasing":{"scale":(0.02, 0.33),   
        "ratio":(0.3, 3.3),
        "value":0},
        
        "cutmix":{"times":1,
        "size":[0.2,0.5],
        "same_on_batch":False,
        "beta":None,
        "keep_dim":False},
        
        "gb":{"kernel_size":9,
        "sigma":(0.1, 2.0)},
        
        "crop":{"size":9}
        
        
        
    }
    # aug = Cutmix_Augmentation(labels=labels,images=images,transform_params=transform_params)
    # aug = Erasing_Augmentation(images,transform_params)
    # aug = Mixup_Augmentation(labels=labels,images=images,transform_params=transform_params)
    # aug = Guassian_Blur_Augmentation(images,transform_params)
    # aug = Crop_Augmentation(images,transform_params)
    aug = ZCA_Whitening_Augmentation(images,transform_params)
    images= aug()
    
    for img in images:
        print(img)
        print((img==images[0]).all())
    
    
        
    
    

        