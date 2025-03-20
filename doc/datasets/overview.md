# Datasets

DD-Ranking provides a set of commonly used datasets in existing dataset distillation methods. Users can flexibly use these datasets for evaluation. The interface to load datasets is as follows:

<div style="background-color:#F7F7F7; padding:15px; border:1px solid #E0E0E0; border-top:3px solid #FF0000; font-family:monospace; font-size:14px;">

ddranking.utils.get_dataset(dataset: str, data_path: str, im_size: tuple, use_zca: bool, custom_val_trans: Optional[Callable], device: str)
[**[SOURCE]**](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/blob/main/ddranking/utils/data.py)
</div>

### Parameters

- **dataset**(<span style="color:#FF6B00;">str</span>): Name of the dataset.
- **data_path**(<span style="color:#FF6B00;">str</span>): Path to the dataset.
- **im_size**(<span style="color:#FF6B00;">tuple</span>): Image size.
- **use_zca**(<span style="color:#FF6B00;">bool</span>): Whether to use ZCA whitening. When set to True, the dataset will **not be** normalized using the mean and standard deviation of the training set.
- **custom_val_trans**(<span style="color:#FF6B00;">Optional[Callable]</span>): Custom transformation on the validation set.
- **device**(<span style="color:#FF6B00;">str</span>): Device for performing ZCA whitening.

Currently, we support the following datasets with default settings. We will keep updating this section with more datasets.

- **CIFAR10**
    - **channels**: `3`
    - **im_size**: `(32, 32)`
    - **num_classes**: `10`
    - **mean**: `[0.4914, 0.4822, 0.4465]`
    - **std**: `[0.2023, 0.1994, 0.2010]`
- **CIFAR100**
    - **channels**: `3`
    - **im_size**: `(32, 32)`
    - **num_classes**: `100`
    - **mean**: `[0.4914, 0.4822, 0.4465]`
    - **std**: `[0.2023, 0.1994, 0.2010]`
- **TinyImageNet**
    - **channels**: `3`
    - **im_size**: `(64, 64)`
    - **num_classes**: `200`
    - **mean**: `[0.485, 0.456, 0.406]`
    - **std**: `[0.229, 0.224, 0.225]`