import torch
import numpy as np
import kornia


class Cutmix_Augmentation:
    def __init__(self, device: str="cuda"):
        # self.transform = kornia.augmentation.RandomCutMixV2(
        #     num_mix = params["times"],
        #     cut_size = params["size"],
        #     same_on_batch = params["same_on_batch"],
        #     beta = params["beta"],
        #     keepdim = params["keep_dim"],
        #     p = params["prob"]
        # )

        self.device = device
        self.cutmix_p = 1.0

    def cutmix(self, images):
        rand_index = torch.randperm(images.size()[0]).to(self.device)
        lam = np.random.beta(self.cutmix_p, self.cutmix_p)
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        return images

    def __call__(self, images):
        return self.cutmix(images)




