import os
import random
import torch
from torchvision import transforms
from ddranking.metrics import SoftLabelEvaluator


class ShufflePatches(torch.nn.Module):
    def shuffle_weight(self, img, factor):
        h, w = img.shape[1:]
        th, tw = h // factor, w // factor
        patches = []
        for i in range(factor):
            i = i * tw
            if i != factor - 1:
                patches.append(img[..., i : i + tw])
            else:
                patches.append(img[..., i:])
        random.shuffle(patches)
        img = torch.cat(patches, -1)
        return img

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        return img


root = "./"
device = "cuda:5"
method_name = "RDED"
dataset = "TinyImageNet"
im_size = (64, 64)
data_dir = os.path.join(root, "datasets/tiny-imagenet-200")
model_name = "ConvNet-4-BN"
cutmix_params = {"beta": 1.0}
ipc = 1

custom_train_transform = []
custom_train_transform.append(transforms.ToTensor())
custom_train_transform.append(ShufflePatches(1))
custom_train_transform.append(
    transforms.RandomResizedCrop(
        size=im_size[0],
        scale=(1, 1),
        antialias=True,
    )
)
custom_train_transform.append(transforms.RandomHorizontalFlip())
custom_train_transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
custom_train_transform = transforms.Compose(custom_train_transform)

custom_val_transform = transforms.Compose([
    transforms.Resize(im_size[0] // 7 * 8, antialias=True),
    transforms.CenterCrop(im_size[0]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f"Evaluating {method_name} on {dataset} with ipc{ipc}")
syn_image_dir = os.path.join(root, f"baselines/{method_name}/{dataset}/IPC{ipc}/")
save_path_soft = f"./results/{dataset}/{model_name}/IPC{ipc}/rded_tiny_ipc1.csv"
convd4_soft_obj = SoftLabelEvaluator(
    dataset=dataset, 
    real_data_path=data_dir, 
    ipc=ipc,
    soft_label_criterion='kl',
    soft_label_mode='M',
    default_lr=0.001,
    optimizer='adamw',
    lr_scheduler='lambda_cos',
    weight_decay=0.01,
    temperature=20,
    num_epochs=300,
    model_name=model_name,
    data_aug_func='cutmix',
    aug_params=cutmix_params,
    im_size=im_size,
    real_batch_size=256,
    syn_batch_size=50,
    num_workers=4,
    custom_train_trans=custom_train_transform,
    custom_val_trans=custom_val_transform,
    device=device,
    save_path=save_path_soft
)
print(convd4_soft_obj.compute_metrics(image_path=syn_image_dir, syn_lr=0.001))