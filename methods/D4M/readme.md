# D4M

We provide all the commands to reproduce the D4M results. For instructions on how to setup the code and details, please see [instructions](instructions.md). Due to the lack of evaluation code for the low-resolution dataset, we have chosen to use the relevant code from SRe2L for the evaluation.

# CIFAR10
For CIFAR10, we provide the code to train a teacher model.
``````bash
bash ./scripts/train_cifar10.sh
``````
With the pre-trained teacher, we can evaluate the distilled datasets.
``````bash
bash ./scripts/eval_cifar10.sh
``````
# CIFAR100
For CIFAR100, we provide the code to train a teacher model.
``````bash
bash ./scripts/train_cifar100.sh
``````
With the pre-trained teacher, we can evaluate the distilled datasets.
``````bash
bash ./scripts/eval_cifar100.sh
``````
# TinyImageNet
For TinyImageNet, You can find the training code and checkpoints at [TinyImageNet repo](https://github.com/zeyuanyin/tiny-imagenet). With the pre-trained teacher, we can evaluate the distilled datasets.
``````bash
bash ./scripts/eval_tiny_imagenet.sh
``````

The original evaluation code can be found in [instructions](instructions.md).

