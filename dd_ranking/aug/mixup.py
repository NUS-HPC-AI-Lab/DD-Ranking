import torch
import numpy as np
import kornia


class Mixup_Augmentation:
    def __init__(self, device: str="cuda"):
        # self.transform = kornia.augmentation.RandomMixUpV2(
        #     lambda_val = params["lambda_range"],
        #     same_on_batch = params["same_on_batch"],
        #     keepdim = params["keepdim"],
        #     p = params["prob"]
        # )

        self.mixup_p = 0.8
        self.device = device

    def mixup(self, images):
        rand_index = torch.randperm(images.size()[0]).to(self.device)
        lam = np.random.beta(self.mixup_p, self.mixup_p)

        mixed_images = lam * images + (1 - lam) * images[rand_index]
        return mixed_images
        
    def __call__(self, images):
        return self.mixup(images)
