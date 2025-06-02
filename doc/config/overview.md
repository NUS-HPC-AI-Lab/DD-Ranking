# Config

To ease the usage of DD-Ranking, we allow users to specify the parameters of the evaluator in a config file. The config file is a YAML file that contains the parameters of the evaluator. We illustrate the config file with the following examples.

## LRS

```yaml
dataset: CIFAR100                 # dataset name
real_data_path: ./dataset/        # path to the real dataset
ipc: 10                           # image per class
im_size: [32, 32]                 # image size
model_name: ResNet-18-BN          # model name
stu_use_torchvision: true         # whether to use torchvision to load student model

tea_use_torchvision: true         # whether to use torchvision to load teacher model

teacher_dir: ./teacher_models     # path to the pretrained teacher model
teacher_model_names: [ResNet-18-BN]     # the list of teacher models being used for evaluation

data_aug_func: mixup              # data augmentation function
aug_params:
    lambda: 0.8                   # data augmentation parameter; please follow this format for other parameters

use_zca: false                    # whether to use ZCA whitening
use_aug_for_hard: false             # whether to use data augmentation for hard label evaluation

custom_train_trans:               # custom torchvision-based transformations to process training data; please follow this format for your own transformations
  - name: RandomCrop
    args:
      size: 32
      padding: 4
  - name: RandomHorizontalFlip
    args:
      p: 0.5
  - name: ToTensor
  - name: Normalize
    args:
      mean: [0.4914, 0.4822, 0.4465]
      std: [0.2023, 0.1994, 0.2010]

custom_val_trans: null              # custom torchvision-based transformations to process validation data; please follow the format above for your own transformations

soft_label_mode: M                  # soft label mode
soft_label_criterion: kl            # soft label criterion
loss_fn_kwargs:
    temperature: 30.0               # temperature for soft label
    scale_loss: false               # whether to scale the loss

optimizer: adamw                    # optimizer
lr_scheduler: cosine                # learning rate scheduler
weight_decay: 0.01                  # weight decay
momentum: 0.9                       # momentum
num_eval: 5                         # number of evaluations
eval_full_data: false               # whether to compute the test accuracy on the full dataset
num_epochs: 400                     # number of training epochs
num_workers: 4                      # number of workers
device: cuda                        # device
dist: true                          # whether to use distributed training
syn_batch_size: 256                 # batch size for synthetic data
real_batch_size: 256                # batch size for real data
save_path: ./results.csv            # path to save the results

random_data_format: tensor          # format of the random data, tensor or image
random_data_path: ./random_data     # path to the save the random data

```

To use config file, you can follow the example below.

```python
from dd_ranking.metrics import LabelRobustScoreSoft

config = Config(config_path='./config.yaml')
evaluator = LabelRobustScoreSoft(config)
```


## ARS

```yaml

```

