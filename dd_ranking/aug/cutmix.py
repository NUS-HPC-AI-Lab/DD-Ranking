import torch
import numpy as np
import kornia


class Cutmix_Augmentation:
    def __init__(self, params: dict):
        self.transform = kornia.augmentation.RandomCutMixV2(
            num_mix = params["times"],
            cut_size = params["size"],
            same_on_batch = params["same_on_batch"],
            beta = params["beta"],
            keepdim = params["keep_dim"],
            p = params["prob"]
        )
    
    def __call__(self, images, labels):
        return self.transform(images, labels)