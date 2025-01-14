## Quick Start

Below is a step-by-step guide on how to use our `dd_ranking`. This demo is based on soft labels (source code can be found in `demo_soft.py`). You can find hard label demo in `demo_hard.py`.

**Step1**: Intialize a soft-label metric evaluator object. Config files are recommended for users to specify hyper-parameters. Sample config files are provided [here](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/tree/main/configs).

```python
from ddranking.metrics import SoftLabelEvaluator
from ddranking.config import Config

>>> config = Config.from_file("./configs/Demo_Soft_Label.yaml")
>>> soft_label_metric_calc = SoftLabelEvaluator(config)
```

<details>
<summary>You can also pass keyword arguments.</summary>

```python
device = "cuda"
method_name = "DATM"                    # Specify your method name
ipc = 10                                # Specify your IPC
dataset = "CIFAR10"                     # Specify your dataset name
syn_data_dir = "./data/CIFAR10/IPC10/"  # Specify your synthetic data path
real_data_dir = "./datasets"            # Specify your dataset path
model_name = "ConvNet-3"                # Specify your model name
teacher_dir = "./teacher_models"		# Specify your path to teacher model chcekpoints
im_size = (32, 32)                      # Specify your image size
dsa_params = {                          # Specify your data augmentation parameters
    "prob_flip": 0.5,
    "ratio_rotate": 15.0,
    "saturation": 2.0,
    "brightness": 1.0,
    "contrast": 0.5,
    "ratio_scale": 1.2,
    "ratio_crop_pad": 0.125,
    "ratio_cutout": 0.5
}
save_path = f"./results/{dataset}/{model_name}/IPC{ipc}/datm_ranking_scores.csv"

""" We only list arguments that usually need specifying"""
soft_label_metric_calc = SoftLabelEvaluator(
    dataset=dataset,
    real_data_path=real_data_dir, 
    ipc=ipc,
    model_name=model_name,
    soft_label_criterion='sce',  # Use Soft Cross Entropy Loss
    soft_label_mode='S',         # Use one-to-one image to soft label mapping
    data_aug_func='dsa',         # Use DSA data augmentation
    aug_params=dsa_params,       # Specify dsa parameters
    im_size=im_size,
    stu_use_torchvision=False,
    tea_use_torchvision=False,
    teacher_dir='./teacher_models',
    device=device,
    save_path=save_path
)
```
</details>

For detailed explanation for hyper-parameters, please refer to our <a href="">documentation</a>.

**Step 2:** Load your synthetic data, labels (if any), and learning rate (if any).

```python
>>> syn_images = torch.load('/your/path/to/syn/images.pt')
# You must specify your soft labels if your soft label mode is 'S'
>>> soft_labels = torch.load('/your/path/to/syn/labels.pt')
>>> syn_lr = torch.load('/your/path/to/syn/lr.pt')
```

**Step 3:** Compute the metric.

```python
>>> metric = soft_label_metric_calc.compute_metrics(image_tensor=syn_images, soft_labels=soft_labels, syn_lr=syn_lr)
# alternatively, you can specify the image folder path to compute the metric
>>> metric = soft_label_metric_calc.compute_metrics(image_path='./your/path/to/syn/images', soft_labels=soft_labels, syn_lr=syn_lr)
```

The following results will be returned to you:
- `hard_label_recovery mean`: The mean of hard label recovery scores.
- `hard_label_recovery std`: The standard deviation of hard label recovery scores.
- `improvement_over_random mean`: The mean of improvement over random scores.
- `improvement_over_random std`: The standard deviation of improvement over random scores.
<!-- - `dd_ranking_score mean`: The mean of dd ranking scores.
- `dd_ranking_score std`: The standard deviation of dd ranking scores. -->
