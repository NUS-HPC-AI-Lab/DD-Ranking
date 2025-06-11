import os
import random
import torch
from torchvision import transforms
from ddranking.metrics import LabelRobustScoreSoft  


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


root = "/path/to/your/root"
device = "cuda"
method_name = "RDED"    
dataset = "ImageNet1K"
im_size = (224, 224)
data_dir = os.path.join(root, "datasets/imagenet1k")
model_name = "ResNet-18-BN"
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
syn_image_dir = os.path.join(root, f"DD-Ranking/baselines/{method_name}/{dataset}/IPC{ipc}/")
save_path_soft = f"./results/{dataset}/{model_name}/IPC{ipc}/rded_in1k_ipc1.csv"
convd4_soft_obj = LabelRobustScoreSoft(
    dataset=dataset, 
    real_data_path=data_dir, 
    ipc=ipc,
    loss_fn_kwargs={'temperature': 20, 'scale_loss': False},
    soft_label_mode='M',
    optimizer='adamw',
    lr_scheduler='cosine',
    weight_decay=0.01,
    num_epochs=300,
    model_name=model_name,
    stu_use_torchvision=False,
    tea_use_torchvision=False,
    teacher_dir="./teacher_models",
    teacher_model_names=["ResNet-18-BN"],
    data_aug_func='cutmix',
    aug_params=cutmix_params,
    use_zca=False,
    use_aug_for_hard=False,
    im_size=im_size,
    real_batch_size=256,
    syn_batch_size=20,
    num_workers=4,
    eval_full_data=False,
    custom_train_trans=custom_train_transform,
    custom_val_trans=custom_val_transform,
    device=device,
    dist=True,
    save_path=save_path_soft,
    random_data_format='image',
    random_data_path="./random_data/RDED/ImageNet1K/IPC1/"
)
convd4_soft_obj.compute_metrics(image_path=syn_image_dir, syn_lr=0.001)