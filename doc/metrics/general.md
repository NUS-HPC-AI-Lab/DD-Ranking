## GeneralEvaluator

<div style="background-color:#F7F7F7; padding:15px; border:1px solid #E0E0E0; border-top:3px solid #FF0000; font-family:monospace; font-size:14px;">

<span style="color:#FF6B00;">CLASS</span> 
dd_ranking.metrics.GeneralEvaluator(config: Optional[Config] = None,
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
[**[SOURCE]**](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/blob/main/ddranking/metrics/general.py)
</div>

A class for evaluating the traditional test accuracy of a surrogate model on the synthetic dataset under various settings (label type, data augmentation, etc.).

### Parameters
Same as [Soft Label Evaluator](soft-label.md).

### Methods
<div style="background-color:#F7F7F7; padding:15px; border:1px solid #E0E0E0; border-top:3px solid #FF0000; font-family:monospace; font-size:14px; margin-left:15px; margin-right:15px;">

compute_metrics(image_tensor: Tensor = None, image_path: str = None, labels: Tensor = None, syn_lr: float = None)
</div>

<div style="margin-left:15px; margin-right:15px;">
This method computes the test accuracy of the surrogate model on the synthetic dataset under various settings (label type, data augmentation, etc.).

#### Parameters

- **image_tensor**(<span style="color:#FF6B00;">Tensor</span>): Image tensor. Must specify when `image_path` is not provided. We require the shape to be `(N x IPC, C, H, W)` where `N` is the number of classes.
- **image_path**(<span style="color:#FF6B00;">str</span>): Path to the image. Must specify when `image_tensor` is not provided.
- **labels**(<span style="color:#FF6B00;">Tensor</span>): Label tensor. It can be either hard labels or soft labels. When `soft_label_mode=S`, the label tensor must be provided.
- **syn_lr**(<span style="color:#FF6B00;">float</span>): Learning rate for the synthetic dataset. If not specified, the learning rate will be tuned automatically.

#### Returns
A dictionary with the following keys:
- **acc_mean**: Mean of test accuracy from `num_eval` rounds.
- **acc_std**: Standard deviation of test accuracy from `num_eval` rounds.
</div>

### Examples

with config file:
```python
>>> config = Config('/path/to/config.yaml')
>>> evaluator = GeneralEvaluator(config=config)
# load image and labels
>>> image_tensor, labels = ... 
# compute metrics
>>> evaluator.compute_metrics(image_tensor=image_tensor, labels=labels)
# alternatively, provide image path
>>> evaluator.compute_metrics(image_path='path/to/image.jpg', labels=labels) 
```

with keyword arguments:
```python
>>> evaluator = GeneralEvaluator(
...     dataset='CIFAR10',
...     model_name='ConvNet-3',
...     soft_label_mode='S',
...     soft_label_criterion='sce',
...     temperature=1.0,
...     data_aug_func='cutmix',
...     aug_params={
...         "cutmix_p": 1.0,
...     },
...     optimizer='sgd',
...     lr_scheduler='step',
...     weight_decay=0.0005,
...     momentum=0.9,
...     stu_use_torchvision=False,
...     tea_use_torchvision=False,
...     num_eval=5,
...     device='cuda'
... )
# load image and labels
>>> image_tensor, labels = ... 
# compute metrics
>>> evaluator.compute_metrics(image_tensor=image_tensor, labels=labels)
# alternatively, provide image path
>>> evaluator.compute_metrics(image_path='path/to/image.jpg', labels=labels) 
```