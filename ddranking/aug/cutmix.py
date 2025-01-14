import torch
import numpy as np
import kornia


class Cutmix:
    def __init__(self, params: dict):
        self.beta = params["beta"]

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix(self, images):
        rand_index = torch.randperm(images.size()[0]).to(images.device)
        lam = np.random.beta(self.beta, self.beta)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)

        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        return images

    def __call__(self, images):
        return self.cutmix(images)




