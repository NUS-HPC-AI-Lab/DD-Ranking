# SRe$^2$L

We provide all the commands to reproduce the SRe$^2$L results. For instructions on how to setup the code and details, please see [instructions](instructions.md). 

# CIFAR10
``````bash
bash ./scripts/eval_cifar10.sh
``````
# CIFAR100
``````bash
bash ./scripts/eval_cifar100.sh
``````
# TinyImageNet
For TinyImageNet, you can find the training code and checkpoints at [TinyImageNet repo](https://github.com/zeyuanyin/tiny-imagenet). With the pre-trained teacher, we can evaluate the distilled datasets.
``````bash
bash ./scripts/eval_tiny_imagenet.sh
``````

More details and the original commands can be found in [instructions](instructions.md).

