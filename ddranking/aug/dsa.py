import torch
import numpy as np
import torch.nn.functional as F


class DSA:

    def __init__(self, params: dict, seed: int=-1, aug_mode: str='S'):        
        self.params = params
        self.seed = seed
        self.aug_mode = aug_mode

        default_funcs = ['scale', 'rotate', 'flip', 'color', 'crop', 'cutout']
        self.transform_funcs = self.create_transform_funcs(default_funcs)

    def create_transform_funcs(self, func_names):
        funcs = []
        for func_name in func_names:
            funcs.append(getattr(self, 'rand_' + func_name))
        return funcs
    
    def set_seed_DiffAug(self):
        if self.params["latestseed"] == -1:
            return
        else:
            torch.random.manual_seed(self.params["latestseed"])
            self.params["latestseed"] += 1
    
    # The following differentiable augmentation strategies are adapted from https://github.com/VICO-UoE/DatasetCondensation
    def rand_scale(self, x):
        # x>1, max scale
        # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
        ratio = self.params["scale"]
        self.set_seed_DiffAug()
        sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
        self.set_seed_DiffAug()
        sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
        theta = [[[sx[i], 0,  0],
                [0,  sy[i], 0],] for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        if self.params["siamese"]: # Siamese augmentation:
            theta[:] = theta[0]
        grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def rand_rotate(self, x): # [-180, 180], 90: anticlockwise 90 degree
        ratio = self.params["rotate"]
        self.set_seed_DiffAug()
        theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
        theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
            [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        if self.params["siamese"]: # Siamese augmentation:
            theta[:] = theta[0]
        grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def rand_flip(self, x):
        prob = self.params["flip"]
        self.set_seed_DiffAug()
        randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
        if self.params["siamese"]: # Siamese augmentation:
            randf[:] = randf[0]
        return torch.where(randf < prob, x.flip(3), x)

    def rand_brightness(self, x):
        ratio = self.params["brightness"]
        self.set_seed_DiffAug()
        randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        if self.params["siamese"]: # Siamese augmentation:
            randb[:] = randb[0]
        x = x + (randb - 0.5)*ratio
        return x

    def rand_saturation(self, x):
        ratio = self.params["saturation"]
        x_mean = x.mean(dim=1, keepdim=True)
        self.set_seed_DiffAug()
        rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        if self.params["siamese"]: # Siamese augmentation:
            rands[:] = rands[0]
        x = (x - x_mean) * (rands * ratio) + x_mean
        return x

    def rand_contrast(self, x):
        ratio = self.params["contrast"]
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        self.set_seed_DiffAug()
        randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        if self.params["siamese"]: # Siamese augmentation:
            randc[:] = randc[0]
        x = (x - x_mean) * (randc + ratio) + x_mean
        return x
    
    def rand_color(self, x):
        return self.rand_contrast(self.rand_saturation(self.rand_brightness(x)))

    def rand_crop(self, x):
        # The image is padded on its surrounding and then cropped.
        ratio = self.params["crop"]
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        self.set_seed_DiffAug()
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        self.set_seed_DiffAug()
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        if self.params["siamese"]: # Siamese augmentation:
            translation_x[:] = translation_x[0]
            translation_y[:] = translation_y[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

    def rand_cutout(self, x):
        ratio = self.params["cutout"]
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        self.set_seed_DiffAug()
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        self.set_seed_DiffAug()
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        if self.params["siamese"]: # Siamese augmentation:
            offset_x[:] = offset_x[0]
            offset_y[:] = offset_y[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x
        
    def __call__(self, images):
        
        if not self.transform_funcs:
            return images

        if self.seed == -1:  
            self.params["siamese"] = False
        else: 
            self.params["siamese"] = True
            
        self.params["latestseed"] = self.seed
        
        if self.aug_mode == 'M': # original
            for f in self.transform_funcs:
                images = f(images)
                
        elif self.aug_mode == 'S':
            self.set_seed_DiffAug()
            p = self.transform_funcs[torch.randint(0, len(self.transform_funcs), size=(1,)).item()]
            images = p(images)
                
        images = images.contiguous()
            
        return images