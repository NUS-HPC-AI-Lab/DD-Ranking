# Tiny-ImageNet training in Torchvision

This project expends `torchvision` to support training on Tiny-ImageNet.

Code is based on the official implementation for image classification in torchvision: <https://github.com/pytorch/vision/tree/main/references/classification>

## Model Zoo

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/zeyuanyin/tiny-imagenet)
|   name    | epochs | acc@1 (last) |                                              url                                               |
| :-------: | :----: | :----------: | :--------------------------------------------------------------------------------------------: |
| ResNet-18 |   50   |    59.57     | [model](https://huggingface.co/zeyuanyin/tiny-imagenet/resolve/main/rn18_50ep/checkpoint.pth)  |
| ResNet-18 |  100   |    60.23     | [model](https://huggingface.co/zeyuanyin/tiny-imagenet/resolve/main/rn18_100ep/checkpoint.pth) |
| ResNet-18 |  200   |    60.50     | [model](https://huggingface.co/zeyuanyin/tiny-imagenet/resolve/main/rn18_200ep/checkpoint.pth) |
| ResNet-50 |   50   |    62.77     | [model](https://huggingface.co/zeyuanyin/tiny-imagenet/resolve/main/rn50_50ep/checkpoint.pth)  |
| ResNet-50 |  100   |    63.19     | [model](https://huggingface.co/zeyuanyin/tiny-imagenet/resolve/main/rn50_100ep/checkpoint.pth) |
| ResNet-50 |  200   |    63.45     | [model](https://huggingface.co/zeyuanyin/tiny-imagenet/resolve/main/rn50_200ep/checkpoint.pth) |

## Training

All models have been trained on 1x A100 GPU with the following parameters with different `epochs`.

| Parameter                | value               |
| ------------------------ | ------------------- |
| `--batch_size`           | `256`               |
| `--epochs`               | `50`                |
| `--lr`                   | `0.2`               |
| `--momentum`             | `0.9`               |
| `--wd`, `--weight-decay` | `1e-4`              |
| `--lr-scheduler`         | `cosineannealinglr` |
| `--lr-warmup-epochs`     | `5`                 |
| `--lr-warmup-method`     | `linear`            |
| `--lr-warmup-decay`      | `0.01`              |

```bash
torchrun --nproc_per_node=1  classification/train.py \
    --model 'resnet18' \
    --batch-size 256 \
    --epochs 50 \
    --opt 'sgd' \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler 'cosineannealinglr' \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --output-dir 'save/rn18_50ep'
```

or

```bash
cd script
bash train_rn18.sh
```

## Downstream Use

- **Dataset Distillation** - [SRe<sup>2</sup>L](https://github.com/VILA-Lab/SRe2L)
  - distill Tiny-ImageNet images from these pre-trained models
  - post-train the validation model on the distilled dataset using `classification/train_kd.py`

## Switch to Tiny ImageNet from ImageNet

### Dataset Transform

> <https://github.com/Westlake-AI/openmixup/blob/084a8f113df34997d041c323a2ea1c9342f5400d/configs/classification/_base_/datasets/tiny_imagenet/sz64_bs100.py#L10-L20>

### Modified ResNet

> replace the 7 × 7 convolution and MaxPooling by a 3 × 3 convolution on ResNet models

```python
model = torchvision.models.get_model('resnet18', num_classes=200)
model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
model.maxpool = nn.Identity()
```

### LR optimizations

> <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#lr-optimizations>

### mixup and cutmix (optional)

> <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#mixup-and-cutmix>

## Reference

Code Base (Official TorchVision):
<https://github.com/pytorch/vision/tree/main/references/classification>

Blog V1 -> V2:
<https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>

ImageNet Evaluation Table:
<https://pytorch.org/vision/stable/models.html>

AutoMixup Paper:
<https://arxiv.org/abs/2103.13027>
