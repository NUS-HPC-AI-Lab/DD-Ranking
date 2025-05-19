## Differentiable Siamese Augmentation (DSA)

DSA is one of differentiable data augmentations, first used in the dataset distillation task by [DSA](https://github.com/VICO-UoE/DatasetCondensation). 
Our implementation of DSA is adopted from [DSA](https://github.com/VICO-UoE/DatasetCondensation). It supports the following differentiable augmentations:

- Random Flip
- Random Rotation
- Random Saturation
- Random Brightness
- Random Contrast
- Random Scale
- Random Crop
- Random Cutout

<div style="background-color:#F7F7F7; padding:15px; border:1px solid #E0E0E0; border-top:3px solid #FF0000; font-family:monospace; font-size:14px;">

<span style="color:#FF6B00;">CLASS</span> 
ddranking.aug.DSA(params: dict, seed: int, aug_mode: str)
[**[SOURCE]**](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/blob/main/ddranking/aug/dsa.py)

</div>

### Parameters

- **params**(<span style="color:#FF6B00;">dict</span>): Parameters for the DSA augmentations. We require the parameters to be in the format of `{'param_name': param_value}`. For example, `{'flip': 0.5, 'rotate': 15.0, 'scale': 1.2, 'crop': 0.125, 'cutout': 0.5, 'brightness': 1.0, 'contrast': 0.5, 'saturation': 2.0}`.
- **seed**(<span style="color:#FF6B00;">int</span>): Random seed. Default is `-1`.
- **aug_mode**(<span style="color:#FF6B00;">str</span>): `S` for randomly selecting one augmentation for each batch. `M` for applying all augmentations for each batch.

### Example

```python
# When intializing an evaluator with DSA augmentation, and DSA object will be constructed.
>>> self.aug_func = DSA(params={'flip': 0.5, 'rotate': 15.0, 'scale': 1.2, 'crop': 0.125, 'cutout': 0.5, 'brightness': 1.0, 'contrast': 0.5, 'saturation': 2.0}, seed=-1, aug_mode='S')

# During training, the DSA object will be used to augment the data.
>>> images = aug_func(images)
```
