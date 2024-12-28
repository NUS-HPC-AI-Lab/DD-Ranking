## Soft Label Evaluator

<div style="background-color:#F7F7F7; padding:15px; border:1px solid #E0E0E0; border-top:3px solid #FF0000; font-family:monospace; font-size:14px;">

<span style="color:#FF6B00;">CLASS</span> 
dd_ranking.metrics.Soft_Label_Evaluator(config: Optional[Config] = None,
    dataset: str = 'CIFAR10',
    real_data_path: str = './dataset/',
    ipc: int = 10,
    model_name: str = 'ConvNet-3',
    soft_label_mode: str='S',
    soft_label_criterion: str='kl', 
    temperature: float=1.0,
    data_aug_func: str='cutmix', 
    aug_params: dict={'cutmix_p': 1.0}, 
    optimizer: str='sgd', 
    lr_scheduler: str='step', 
    weight_decay: float=0.0005, 
    momentum: float=0.9, 
    num_eval: int=5, 
    im_size: tuple=(32, 32), 
    num_epochs: int=300, 
    use_zca: bool=False,
    real_batch_size: int=256, 
    syn_batch_size: int=256, 
    default_lr: float=0.01, 
    save_path: str=None, 
    stu_use_torchvision: bool=False, 
    tea_use_torchvision: bool=False, 
    num_workers: int=4, 
    teacher_dir: str='./teacher_models', 
    custom_train_trans: Optional[Callable]=None, 
    custom_val_trans: Optional[Callable]=None, 
    device: str="cuda"
)
[**[SOURCE]**](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/blob/main/dd_ranking/metrics/soft_label.py)
</div>

A class for evaluating the performance of a dataset distillation method with soft labels. User is able to modify the attributes as needed.

### Parameters

- **config**(<span style="color:#FF6B00;">Optional[Config]</span>): Config object for specifying all attributes. See [config](../config/overview.md) for more details.
- **dataset**(<span style="color:#FF6B00;">str</span>): Name of the real dataset.
- **real_data_path**(<span style="color:#FF6B00;">str</span>): Path to the real dataset.
- **ipc**(<span style="color:#FF6B00;">int</span>): Images per class.
- **model_name**(<span style="color:#FF6B00;">str</span>): Name of the surrogate model. See [models](../models/overview.md) for more details.
- **soft_label_mode**(<span style="color:#FF6B00;">str</span>): Number of soft labels per image. `S` for single soft label, `M` for multiple soft labels.
- **soft_label_criterion**(<span style="color:#FF6B00;">str</span>): Loss function for using soft labels. Currently supports `kl` for KL divergence, `sce` for soft cross-entropy.
- **temperature**(<span style="color:#FF6B00;">float</span>): Temperature for knowledge distillation.
- **data_aug_func**(<span style="color:#FF6B00;">str</span>): Data augmentation function used during training. Currently supports `dsa`, `cutmix`, `mixup`. See [augmentations](../augmentations/overview.md) for more details.
- **aug_params**(<span style="color:#FF6B00;">dict</span>): Parameters for the data augmentation function.
- **optimizer**(<span style="color:#FF6B00;">str</span>): Name of the optimizer. Currently supports torch-based optimizers - `sgd`, `adam`, and `adamw`.
- **lr_scheduler**(<span style="color:#FF6B00;">str</span>): Name of the learning rate scheduler. Currently supports torch-based schedulers - `step`, `cosine`, `lambda_step`, and `lambda_cos`.
- **weight_decay**(<span style="color:#FF6B00;">float</span>): Weight decay for the optimizer.
- **momentum**(<span style="color:#FF6B00;">float</span>): Momentum for the optimizer.
- **use_zca**(<span style="color:#FF6B00;">bool</span>): Whether to use ZCA whitening.
- **num_eval**(<span style="color:#FF6B00;">int</span>): Number of evaluations to perform.
- **im_size**(<span style="color:#FF6B00;">tuple</span>): Size of the images.
- **num_epochs**(<span style="color:#FF6B00;">int</span>): Number of epochs to train.
- **real_batch_size**(<span style="color:#FF6B00;">int</span>): Batch size for the real dataset.
- **syn_batch_size**(<span style="color:#FF6B00;">int</span>): Batch size for the synthetic dataset.
- **stu_use_torchvision**(<span style="color:#FF6B00;">bool</span>): Whether to use torchvision to initialize the student model.
- **tea_use_torchvision**(<span style="color:#FF6B00;">bool</span>): Whether to use torchvision to initialize the teacher model.
- **teacher_dir**(<span style="color:#FF6B00;">str</span>): Path to the teacher model.
- **default_lr**(<span style="color:#FF6B00;">float</span>): Default learning rate for the optimizer, typically used for training on the real dataset.
- **num_workers**(<span style="color:#FF6B00;">int</span>): Number of workers for data loading.
- **save_path**(<span style="color:#FF6B00;">Optional[str]</span>): Path to save the results.
- **custom_train_trans**(<span style="color:#FF6B00;">Optional[Callable]</span>): Custom transformation function when loading synthetic data. Only support torchvision transformations.
- **custom_val_trans**(<span style="color:#FF6B00;">Optional[Callable]</span>): Custom transformation function when loading test dataset. Only support torchvision transformations.
- **device**(<span style="color:#FF6B00;">str</span>): Device to use for evaluation, `cuda` or `cpu`.

### Methods
<div style="background-color:#F7F7F7; padding:15px; border:1px solid #E0E0E0; border-top:3px solid #FF0000; font-family:monospace; font-size:14px; margin-left:15px; margin-right:15px;">

compute_metrics(image_tensor: Tensor = None, image_path: str = None, soft_labels: Tensor = None, syn_lr: float = None)
</div>

<div style="margin-left:15px; margin-right:15px;">
This method computes the HLR, IOR, and DD-Ranking scores for the given image and soft labels (if provided). In each evaluation round, we set a different random seed and perform the following steps:

1. Compute the test accuracy of the surrogate model on the synthetic dataset under hard labels. We perform learning rate tuning for the best performance.
2. Compute the test accuracy of the surrogate model on the real dataset under the same setting as step 1.
3. Compute the test accuracy of the surrogate model on the synthetic dataset under soft labels.
4. Compute the test accuracy of the surrogate model on the randomly selected dataset under the same setting as step 3.
5. Compute the HLR, IOR, and DD-Ranking scores.

The final scores are the average of the scores from `num_eval` rounds.

#### Parameters

- **image_tensor**(<span style="color:#FF6B00;">Tensor</span>): Image tensor. Must specify when `image_path` is not provided. We require the shape to be `(N x IPC, C, H, W)` where `N` is the number of classes.
- **image_path**(<span style="color:#FF6B00;">str</span>): Path to the image. Must specify when `image_tensor` is not provided.
- **soft_labels**(<span style="color:#FF6B00;">Tensor</span>): Soft label tensor. Must specify when `soft_label_mode` is `S`. The first dimension must be the same as `image_tensor`.
- **syn_lr**(<span style="color:#FF6B00;">float</span>): Learning rate for the synthetic dataset. If not specified, the learning rate will be tuned automatically.

#### Returns

A dictionary with the following keys:

- **hard_label_recovery_mean**: Mean of HLR scores from `num_eval` rounds.
- **hard_label_recovery_std**: Standard deviation of HLR scores from `num_eval` rounds.
- **improvement_over_random_mean**: Mean of improvement over random scores from `num_eval` rounds.
- **improvement_over_random_std**: Standard deviation of improvement over random scores from `num_eval` rounds.
- **dd_ranking_mean**: Mean of DD-Ranking scores from `num_eval` rounds.
- **dd_ranking_std**: Standard deviation of DD-Ranking scores from `num_eval` rounds.

</div>

### Examples

with config file:
```python
>>> config = Config('/path/to/config.yaml')
>>> evaluator = Soft_Label_Evaluator(config=config)
# load image and soft labels
>>> image_tensor, soft_labels = ... 
# compute metrics
>>> evaluator.compute_metrics(image_tensor=image_tensor, soft_labels=soft_labels)
# alternatively, provide image path
>>> evaluator.compute_metrics(image_path='path/to/image.jpg', soft_labels=soft_labels) 
```

with keyword arguments:
```python
>>> evaluator = Soft_Label_Evaluator(
...     dataset='TinyImageNet',
...     model_name='ResNet-18-BN',
...     soft_label_mode='M',
...     soft_label_criterion='kl',
...     temperature=10.0,
...     data_aug_func='mixup',
...     aug_params={
...         "mixup_p": 0.8,
...     },
...     optimizer='sgd',
...     lr_scheduler='step',
...     weight_decay=0.0005,
...     momentum=0.9,
...     stu_use_torchvision=True,
...     tea_use_torchvision=True,
...     num_eval=5,
...     device='cuda'
... )
# load image and soft labels
>>> image_tensor, soft_labels = ... 
# compute metrics
>>> evaluator.compute_metrics(image_tensor=image_tensor, soft_labels=soft_labels)
# alternatively, provide image path
>>> evaluator.compute_metrics(image_path='path/to/image.jpg', soft_labels=soft_labels) 
```