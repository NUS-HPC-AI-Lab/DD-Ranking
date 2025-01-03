# <center>DD-Ranking: Rethinking the Evaluation of Dataset Distillation</center>

<p align="center">
  <picture>
  <!-- Dark theme logo -->
    <source media="(prefers-color-scheme: dark)" srcset="static/logo.png">
    <!-- Light theme logo -->
    <img alt="DD-Ranking" src="static/logo.png"%>
  </picture>
</p>

<!-- <h3 align="center">
Fair and benchmark for dataset distillation.
</h3> -->
<p align="center">
| <a href="https://nus-hpc-ai-lab.github.io/DD-Ranking/"><b>Documentation</b></a> | <a href="https://huggingface.co/spaces/Soptq/DD-Ranking"><b>Leaderboard</b></a> | <a href=""><b>Paper </b> (Coming Soon)</a> | <a href=""><b>Twitter/X</b> (Coming Soon)</a> | <a href=""><b>Developer Slack</b> (Coming Soon)</a> |
</p>


---

*Latest News* 🔥

[Latest] Our PyPI package is officially released! Users can now install DD-Ranking via `pip install dd_ranking`.

<details>
<summary>Unfold to see more details.</summary>
<br>

- [2025/01] Our PyPI package is officially released! Users can now install DD-Ranking via `pip install dd_ranking`.

- [2024/12] We officially released DD-Ranking! DD-Ranking provides us a new benchmark decoupling the impacts from knowledge distillation and data augmentation.
</details>

---

## Motivation: DD Lacks an Evaluation Benchmark

<details>
<summary>Unfold to see more details.</summary>
<br>
Dataset Distillation (DD) aims to condense a large dataset into a much smaller one, which allows a model to achieve comparable performance after training on it. DD has gained extensive attention since it was proposed. With some foundational methods such as DC, DM, and MTT, various works have further pushed this area to a new standard with their novel designs.

![history](./static/history.png)

Notebaly, more and more methods are transitting from "hard label" to "soft label" in dataset distillation, especially during evaluation. **Hard labels** are categorical, having the same format of the real dataset. **Soft labels** are outputs of a pre-trained teacher model. 
Recently, Deng et al., pointed out that "a label is worth a thousand images". They showed analytically that soft labels are exetremely useful for accuracy improvement. 

However, since the essence of soft labels is **knowledge distillation**, we find that when applying the same evaluation method to randomly selected data, the test accuracy also improves significantly (see the figure above).

This makes us wonder: **Can the test accuracy of the model trained on distilled data reflect the real informativeness of the distilled data?**

Additionally, we have discoverd unfairness of using only test accuracy to demonstrate one's performance from the following three aspects:
1. Results of using hard and soft labels are not directly comparable since soft labels introduce teacher knowledge.
2. Strategies of using soft labels are diverse. For instance, different objective functions are used during evaluation, such as soft Cross-Entropy and Kullback–Leibler divergence. Also, one image may be mapped to one or multiple soft labels.
3. Different data augmentations are used during evaluation.

Motivated by this, we propose DD-Ranking, a new benchmark for DD evaluation. DD-Ranking provides a fair evaluation scheme for DD methods that can decouple the impacts from knowledge distillation and data augmentation to reflect the real informativeness of the distilled data.

</details>

## Introduction

<details>
<summary>Unfold to see more details.</summary>
<br>
DD-Ranking (DD, *i.e.*, Dataset Distillation) is an integrated and easy-to-use benchmark for dataset distillation. It aims to provide a fair evaluation scheme for DD methods that can decouple the impacts from knowledge distillation and data augmentation to reflect the real informativeness of the distilled data.

<!-- Hard label is tested -->
<!-- Keep the same compression ratio, comparing with random selection -->
### Benchmark

Revisit the original goal of dataset distillation: 
> The idea is to synthesize a small number of data points that do not need to come from the correct data distribution, but will, when given to the learning algorithm as training data, approximate the model trained on the original data. (Wang et al., 2020)
>

The evaluation method for DD-Ranking is grounded in the essence of dataset distillation, aiming to better reflect the informativeness of the synthesized data by assessing the following two aspects:  
1. The degree to which the original dataset is recovered under hard labels (hard label recovery): $\text{HLR}=\text{Acc.}{\text{full-hard}}-\text{Acc.}{\text{syn-hard}}$.  

2. The improvement over random selection when using personalized evaluation methods (improvement over random): $\text{IOR}=\text{Acc.}{\text{syn-any}}-\text{Acc.}{\text{rdm-any}}$.
$\text{Acc.}$ is the accuracy of models trained on different samples. Samples' marks are as follows:
- $\text{full-hard}$: Full dataset with hard labels;
- $\text{syn-hard}$: Synthetic dataset with hard labels;
- $\text{syn-any}$: Synthetic dataset with personalized evaluation methods (hard or soft labels);
- $\text{rdm-any}$: Randomly selected dataset (under the same compression ratio) with the same personalized evaluation methods.

DD-Ranking uses a weight sum of $-\text{IOR}$ and $\text{HLR}$ to rank different methods:
$$
\text{Rank\_Score} = w_1 (-\text{IOR}) + w_2 \text{HLR}, \quad w_1 + w_2 = 1
$$
By default, we set $w_1 = w_2 = 0.5$ on the leaderboard, meaning that both $\text{IOR}$ and $\text{HLR}$ are equally important. Users can adjust the weights to emphasize one aspect on the leaderboard.

</details>

## Overview

DD-Ranking is integrated with:
<!-- Uniform Fair Labels: loss on soft label -->
<!-- Data Aug. -->
- Multiple [strategies](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/tree/main/dd_ranking/loss) of using soft labels in existing works;
- Commonly used [data augmentation](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/tree/main/dd_ranking/aug) methods in existing works;
- Commonly used [model architectures](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/blob/main/dd_ranking/utils/networks.py) in existing works.

DD-Ranking has the following features:
- **Fair Evaluation**: DD-Ranking provides a fair evaluation scheme for DD methods that can decouple the impacts from knowledge distillation and data augmentation to reflect the real informativeness of the distilled data.
- **Easy-to-use**: DD-Ranking provides a unified interface for dataset distillation evaluation.
- **Extensible**: DD-Ranking supports various datasets and models.
- **Customizable**: DD-Ranking supports various data augmentations and soft label strategies.

DD-Ranking currently includes the following datasets and methods (categorized by hard/soft label). Evaluation results can be found in the [leaderboard](https://huggingface.co/spaces/Soptq/DD-Ranking).
|Supported Dataset|Evaluated Hard Label Methods|Evaluated Soft Label Methods|
|:-|:-|:-|
|CIFAR-10|DC|DATM|
|CIFAR-100|DSA|SRe2L|
|TinyImageNet|DM|RDED|
||MTT|D4M|



## Tutorial

Install DD-Ranking with `pip` or from [source](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/tree/main):

### Installation

From pip

```bash
pip install dd_ranking
```

From source

```bash
python setup.py install
```
### Quickstart

Below is a step-by-step guide on how to use our `dd_ranking`. This demo is based on soft labels (source code can be found in `demo_soft.py`). You can find hard label demo in `demo_hard.py`.

**Step1**: Intialize a soft-label metric evaluator object. Config files are recommended for users to specify hyper-parameters. Sample config files are provided [here](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/tree/main/configs).

```python
from dd_ranking.metrics import SoftLabelEvaluator
from dd_ranking.config import Config

config = Config.from_file("./configs/Demo_Soft_Label.yaml")
soft_obj = SoftLabelEvaluator(config)
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
save_path = f"./results/{dataset}/{model_name}/IPC{ipc}/dm_hard_scores.csv"

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
syn_images = torch.load('/your/path/to/syn/images.pt')
# You must specify your soft labels if your soft label mode is 'S'
soft_labels = torch.load('/your/path/to/syn/labels.pt')
syn_lr = torch.load('/your/path/to/syn/lr.pt')
```

**Step 3:** Compute the xxx metric.

```python
metric = soft_label_metric_calc.compute_metrics(syn_images, soft_labels=soft_labels, syn_lr=syn_lr)
```

The following results will be returned to you:
- `HLR mean`: The mean of hard label recovery over `num_eval` runs.
- `HLR std`: The standard deviation of hard label recovery over `num_eval` runs.
- `IOR mean`: The mean of improvement over random over `num_eval` runs.
- `IOR std`: The standard deviation of improvement over random over `num_eval` runs.
<!-- - `IOR/HLR mean`: The mean of IOR/HLR over `num_eval` runs.
- `IOR/HLR std`: The standard deviation of IOR/HLR over `num_eval` runs. -->

Check out our <span style="color: #ff0000;">[documentation](https://nus-hpc-ai-lab.github.io/DD-Ranking/)</span> to learn more.

## Coming Soon
- [ ] DD-Ranking scores that decouple the impacts from data augmentation.
- [ ] Evaluation results on ImageNet subsets.
- [ ] More baseline methods.

## Contributing

<!-- Only PR for the 1st version of DD-Ranking -->
Feel free to submit grades to update the DD-Ranking list. We welcome and value any contributions and collaborations.
Please check out [CONTRIBUTING.md](./CONTRIBUTING.md) for how to get involved.

## License

DD-Ranking is released under the MIT License. See [LICENSE](./LICENSE) for more details.

<!-- ## Acknowledgement

DD-Ranking is a community project. The compute resources for development and testing are supported by the following organizations. Thanks for your support! -->

<!-- Note: Please sort them in alphabetical order. -->
<!-- Note: Please keep these consistent with docs/source/community/sponsors.md -->

<!-- - First Org.

We also have an official fundraising venue through <span style="color: #ff0000;">[TODO]：</span>[the collection website](). We plan to use the fund to support the development, maintenance, and adoption of DD-Ranking. -->

<!-- Paper to be added -->
<!-- If a pre-print is wanted, a digital asset could be released first. -->

<!-- ## Citation

If you use DD-Ranking for your research, please cite our [paper]():
```bibtex
@inproceedings{,
  title={DD-Ranking: },
  author={},
  booktitle={},
  year={2024}
}
```

<!-- ## Contact Us

**Community Discussions**: Engage with other users on <span style="color: #ff0000;">[TODO]：</span>[Discord]() for discussions.

**Coordination of Contributions and Development**: Use <span style="color: #ff0000;">[TODO]：</span>[Slack]() for coordinating contributions and managing development efforts.

**Collaborations and Partnerships**: For exploring collaborations or partnerships, reach out via <span style="color: #ff0000;">[TODO]：</span>[email]().

**Technical Queries and Feature Requests**: Utilize GitHub issues or discussions for addressing technical questions and proposing new features.

**Security Disclosures**: Report security vulnerabilities through GitHub's security advisory feature. -->
