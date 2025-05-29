## AugmentationRobustScore

<div style="background-color:#F7F7F7; padding:15px; border:1px solid #E0E0E0; border-top:3px solid #FF0000; font-family:monospace; font-size:14px;">

<span style="color:#FF6B00;">CLASS</span> 
dd_ranking.metrics.AugmentationRobustScore(config: Optional[Config] = None,
    dataset: str = 'ImageNet1K',
    real_data_path: str = './dataset/',
    ipc: int = 10,
    model_name: str = 'ResNet-18-BN',
    label_type: str = 'soft',
    soft_label_mode: str='S',
    soft_label_criterion: str='kl',
    loss_fn_kwargs: dict=None,
    data_aug_func: str='cutmix', 
    aug_params: dict={'cutmix_p': 1.0}, 
    optimizer: str='sgd', 
    lr_scheduler: str='step', 
    weight_decay: float=0.0005, 
    momentum: float=0.9,
    step_size: int=None,
    num_eval: int=5, 
    im_size: tuple=(224, 224), 
    num_epochs: int=300, 
    use_zca: bool=False,
    random_data_format: str='image',
    random_data_path: str=None,
    batch_size: int=256,
    save_path: str=None, 
    stu_use_torchvision: bool=False, 
    tea_use_torchvision: bool=False, 
    num_workers: int=4, 
    teacher_dir: str='./teacher_models',
    teacher_model_names: list=None,
    custom_train_trans: Optional[Callable]=None, 
    custom_val_trans: Optional[Callable]=None, 
    device: str="cuda",
    dist: bool=False
)
[**[SOURCE]**](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/blob/main/ddranking/metrics/aug_robust.py)
</div>

A class for evaluating the performance of a dataset distillation method with soft labels. User is able to modify the attributes as needed.

### Parameters

- **config**(<span style="color:#FF6B00;">Optional[Config]</span>): Config object for specifying all attributes. See [config](../config/overview.md) for more details.
- **dataset**(<span style="color:#FF6B00;">str</span>): Name of the real dataset.
- **real_data_path**(<span style="color:#FF6B00;">str</span>): Path to the real dataset.
- **ipc**(<span style="color:#FF6B00;">int</span>): Images per class.
- **model_name**(<span style="color:#FF6B00;">str</span>): Name of the surrogate model. See [models](../models/overview.md) for more details.
- **label_type**(<span style="color:#FF6B00;">str</span>): Type of label representation. `soft` for soft labels, `hard` for hard labels.
- **soft_label_mode**(<span style="color:#FF6B00;">str</span>): Number of soft labels per image. `S` for single soft label, `M` for multiple soft labels.
- **soft_label_criterion**(<span style="color:#FF6B00;">str</span>): Loss function for using soft labels. Currently supports `kl` for KL divergence, `sce` for soft cross-entropy, and `mse_gt` for MSEGT loss introduced in EDC.
- **loss_fn_kwargs**(<span style="color:#FF6B00;">dict</span>): Keyword arguments for the loss function, e.g. `temperature` and `scale_loss` for KL and SCE loss, and `mse_weight` and `ce_weight` for MSEGT loss.
- **data_aug_func**(<span style="color:#FF6B00;">str</span>): Data augmentation function used during training. Currently supports `dsa`, `cutmix`, `mixup`. See [augmentations](../augmentations/overview.md) for more details.
- **aug_params**(<span style="color:#FF6B00;">dict</span>): Parameters for the data augmentation function.
- **optimizer**(<span style="color:#FF6B00;">str</span>): Name of the optimizer. Currently supports torch-based optimizers - `sgd`, `adam`, and `adamw`.
- **lr_scheduler**(<span style="color:#FF6B00;">str</span>): Name of the learning rate scheduler. Currently supports torch-based schedulers - `step`, `cosine`, `lambda_step`, and `cosineannealing`.
- **weight_decay**(<span style="color:#FF6B00;">float</span>): Weight decay for the optimizer.
- **momentum**(<span style="color:#FF6B00;">float</span>): Momentum for the optimizer.
- **step_size**(<span style="color:#FF6B00;">int</span>): Step size for the learning rate scheduler.
- **use_zca**(<span style="color:#FF6B00;">bool</span>): Whether to use ZCA whitening.
- **num_eval**(<span style="color:#FF6B00;">int</span>): Number of evaluations to perform.
- **im_size**(<span style="color:#FF6B00;">tuple</span>): Size of the images.
- **num_epochs**(<span style="color:#FF6B00;">int</span>): Number of epochs to train.
- **batch_size**(<span style="color:#FF6B00;">int</span>): Batch size for the model training.
- **stu_use_torchvision**(<span style="color:#FF6B00;">bool</span>): Whether to use torchvision to initialize the student model.
- **tea_use_torchvision**(<span style="color:#FF6B00;">bool</span>): Whether to use torchvision to initialize the teacher model.
- **teacher_dir**(<span style="color:#FF6B00;">str</span>): Path to the teacher model.
- **teacher_model_names**(<span style="color:#FF6B00;">list</span>): List of teacher model names.
- **random_data_format**(<span style="color:#FF6B00;">str</span>): Format of the random data, `tensor` or `image`.
- **random_data_path**(<span style="color:#FF6B00;">str</span>): Path to save the random data.
- **num_workers**(<span style="color:#FF6B00;">int</span>): Number of workers for data loading.
- **save_path**(<span style="color:#FF6B00;">Optional[str]</span>): Path to save the results.
- **custom_train_trans**(<span style="color:#FF6B00;">Optional[Callable]</span>): Custom transformation function when loading synthetic data. Only support torchvision transformations. See [torchvision-based transformations](../augmentations/torchvision.md) for more details.
- **custom_val_trans**(<span style="color:#FF6B00;">Optional[Callable]</span>): Custom transformation function when loading test dataset. Only support torchvision transformations. See [torchvision-based transformations](../augmentations/torchvision.md) for more details.
- **device**(<span style="color:#FF6B00;">str</span>): Device to use for evaluation, `cuda` or `cpu`.
- **dist**(<span style="color:#FF6B00;">bool</span>): Whether to use distributed training.

### Methods
<div style="background-color:#F7F7F7; padding:15px; border:1px solid #E0E0E0; border-top:3px solid #FF0000; font-family:monospace; font-size:14px; margin-left:15px; margin-right:15px;">

compute_metrics(image_tensor: Tensor = None, image_path: str = None, soft_labels: Tensor = None, syn_lr: float = None, ars_lambda: float = 0.5)
</div>

<div style="margin-left:15px; margin-right:15px;">
This method computes the ARS score for the given image and soft labels (if provided). In each evaluation round, we set a different random seed and perform the following steps:

1. Compute the test accuracy of the surrogate model on the synthetic dataset without data augmentation.
2. Compute the test accuracy of the surrogate model on the synthetic dataset with data augmentation.
3. Compute the test accuracy of the surrogate model on the randomly selected dataset without data augmentation. We perform learning rate tuning for the best performance.
4. Compute the test accuracy of the surrogate model on the randomly selected dataset with data augmentation. We perform learning rate tuning for the best performance.
5. Compute the ARS score.

The final scores are the average of the scores from `num_eval` rounds.

#### Parameters

- **image_tensor**(<span style="color:#FF6B00;">Tensor</span>): Image tensor. Must specify when `image_path` is not provided. We require the shape to be `(N x IPC, C, H, W)` where `N` is the number of classes.
- **image_path**(<span style="color:#FF6B00;">str</span>): Path to the image. Must specify when `image_tensor` is not provided.
- **soft_labels**(<span style="color:#FF6B00;">Tensor</span>): Soft label tensor. Must specify when `soft_label_mode` is `S`. The first dimension must be the same as `image_tensor`.
- **syn_lr**(<span style="color:#FF6B00;">float</span>): Learning rate for the synthetic dataset. If not specified, the learning rate will be tuned automatically.
- **ars_lambda**(<span style="color:#FF6B00;">float</span>): Weighting parameter for the ARS.

#### Returns

A dictionary with the following keys:

- **with_aug_mean**: Mean of test accuracy scores with data augmentation from `num_eval` rounds.
- **with_aug_std**: Standard deviation of test accuracy scores with data augmentation from `num_eval` rounds.
- **without_aug_mean**: Mean of test accuracy scores without data augmentation from `num_eval` rounds.
- **without_aug_std**: Standard deviation of test accuracy scores without data augmentation from `num_eval` rounds.
- **augmentation_robust_score_mean**: Mean of ARS scores from `num_eval` rounds.
- **augmentation_robust_score_std**: Standard deviation of ARS scores from `num_eval` rounds.

</div>

### Examples

with config file:
```python
>>> config = Config('/path/to/config.yaml')
>>> evaluator = AugmentationRobustScore(config=config)
# load image and soft labels
>>> image_tensor, soft_labels = ... 
# compute metrics
>>> evaluator.compute_metrics(image_tensor=image_tensor, soft_labels=soft_labels)
# alternatively, provide image path
>>> evaluator.compute_metrics(image_path='path/to/image/folder/', soft_labels=soft_labels) 
```

with keyword arguments:
```python
>>> evaluator = AugmentationRobustScore(
...     dataset='ImageNet1K',
...     real_data_path='./dataset/',
...     ipc=10,
...     model_name='ResNet-18-BN',
...     label_type='soft',
...     soft_label_mode='M',
...     soft_label_criterion='kl',
...     loss_fn_kwargs={
...         "temperature": 30.0,
...         "scale_loss": False,
...     },
...     data_aug_func='mixup',
...     aug_params={
...         "mixup_p": 0.8,
...     },
...     optimizer='adamw',
...     lr_scheduler='cosine',
...     num_epochs=300,
...     weight_decay=0.0005,
...     momentum=0.9,
...     use_zca=False,
...     stu_use_torchvision=True,
...     tea_use_torchvision=True,
...     num_workers=4,
...     save_path='./results',
...     random_data_format='image',
...     random_data_path='./random_data',
...     teacher_dir='./teacher_models',
...     teacher_model_names=['ResNet-18-BN'],
...     num_eval=5,
...     device='cuda'
... )
# load image and soft labels
>>> image_tensor, soft_labels = ... 
# compute metrics
>>> evaluator.compute_metrics(image_tensor=image_tensor, soft_labels=soft_labels)
# alternatively, provide image path
>>> evaluator.compute_metrics(image_path='path/to/image/folder/', soft_labels=soft_labels) 
```