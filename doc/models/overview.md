# Models

DD-Ranking provides the implementation of a set of commonly used model architectures in existing dataset distillation methods. Users can flexibly use these models for main evaluation or cross-architecture evaluation. We will keep updating this section with more models.

- [ConvNet](convnet.md)
- [ResNet](resnet.md)
- [VGG](vgg.md)
- [LeNet](lenet.md)
- [AlexNet](alexnet.md)
- [MLP](mlp.md)

Users can also define any model with `torchvision`.

## Naming Convention

We use the following naming conventions for models in DD-Ranking:

- `model name - model depth - norm type` (for DD-Ranking implemented models)
- torchvision model names, e.g. `vgg11` and `vit_b_16`

Model name and depth are required when **not using tochvision**. When norm type is not specified, we use default normalization for the model. For example, `ResNet-18-BN` means ResNet18 with batch normalization. `ConvNet-4` means ConvNet with depth 4 and default instance normalization.

## Pretrained Model Weights

For users' convenience, we provide pretrained model weights on CIFAR10, CIFAR100, and TinyImageNet for the following models:
- ConvNet-3 (CIFAR10, CIFAR100)
- ConvNet-3-BN (CIFAR10, CIFAR100)
- ConvNet-4 (TinyImageNet)
- ConvNet-4-BN (TinyImageNet)
- ResNet-18-BN (CIFAR10, CIFAR100, TinyImageNet, ImageNet1K)

Users can download the weights from the following links: [Pretrained Model Weights](https://drive.google.com/drive/folders/19OnR85PRs3TZk8xS8XNr9hiokfsML4m2?usp=sharing).

Users can also feel free to use `torchvision` pretrained models.
