## Quick Start

Below is a step-by-step guide on how to use our `dd_ranking`. This demo is for label-robust score (LRS) on soft labels (source code can be found in `demo_lrs_soft.py`). You can find the demo for LRS on hard label demo in `demo_lrs_hard.py` and the demo for augmentation-robust score (ARS) in `demo_ars.py`.
DD-Ranking supports multi-GPU Distributed evaluation. You can simply use `torchrun` to launch the evaluation.

**Step1**: Intialize a soft-label metric evaluator object. Config files are recommended for users to specify hyper-parameters. Sample config files are provided [here](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/tree/main/configs).

```python
from ddranking.metrics import LabelRobustScoreSoft
from ddranking.config import Config

>>> config = Config.from_file("./configs/Demo_LRS_Soft_Label.yaml")
>>> lrs_soft_metric = LabelRobustScoreSoft(config)
```

<details>
<summary>You can also pass keyword arguments.</summary>

```python
device = "cuda"
method_name = "DATM"                    # Specify your method name
ipc = 10                                # Specify your IPC
dataset = "CIFAR100"                     # Specify your dataset name
syn_data_dir = "./data/CIFAR100/IPC10/"  # Specify your synthetic data path
real_data_dir = "./datasets"            # Specify your dataset path
model_name = "ConvNet-3"                # Specify your model name
teacher_dir = "./teacher_models"		# Specify your path to teacher model chcekpoints
teacher_model_names = ["ConvNet-3"]      # Specify your teacher model names
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
random_data_format = "tensor"              # Specify your random data format (tensor or image)
random_data_path = "./random_data"          # Specify your random data path
save_path = f"./results/{dataset}/{model_name}/IPC{ipc}/dm_hard_scores.csv"

""" We only list arguments that usually need specifying"""
lrs_soft_metric = LabelRobustScoreSoft(
    dataset=dataset,
    real_data_path=real_data_dir, 
    ipc=ipc,
    model_name=model_name,
    soft_label_criterion='sce',  # Use Soft Cross Entropy Loss
    soft_label_mode='S',         # Use one-to-one image to soft label mapping
    loss_fn_kwargs={'temperature': 1.0, 'scale_loss': False},
    data_aug_func='dsa',         # Use DSA data augmentation
    aug_params=dsa_params,       # Specify dsa parameters
    im_size=im_size,
    random_data_format=random_data_format,
    random_data_path=random_data_path,
    stu_use_torchvision=False,
    tea_use_torchvision=False,
    teacher_dir=teacher_dir,
    teacher_model_names=teacher_model_names,
    num_eval=5,
    device=device,
    dist=True,
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
>>> lrs_soft_metric.compute_metrics(image_tensor=syn_images, soft_labels=soft_labels, syn_lr=syn_lr)
# alternatively, you can specify the image folder path to compute the metric
>>> lrs_soft_metric.compute_metrics(image_path='./your/path/to/syn/images', soft_labels=soft_labels, syn_lr=syn_lr)
```

The following results will be printed and saved to `save_path`:
- `HLR mean`: The mean of hard label recovery over `num_eval` runs.
- `HLR std`: The standard deviation of hard label recovery over `num_eval` runs.
- `IOR mean`: The mean of improvement over random over `num_eval` runs.
- `IOR std`: The standard deviation of improvement over random over `num_eval` runs.
- `LRS mean`: The mean of Label-Robust Score over `num_eval` runs.
- `LRS std`: The standard deviation of Label-Robust Score over `num_eval` runs.
<!-- - `dd_ranking_score mean`: The mean of dd ranking scores.
- `dd_ranking_score std`: The standard deviation of dd ranking scores. -->
