import torch
import numpy as np
import kornia


class Mixup_Augmentation:
    def __init__(self, params: dict):
        self.transform = kornia.augmentation.RandomMixUpV2(
            lambda_val = params["lambda_range"],
            same_on_batch = params["same_on_batch"],
            keepdim = params["keepdim"],
            p = params["prob"]
        )
        
    def __call__(self, images, labels):
        return self.transform(images, labels)