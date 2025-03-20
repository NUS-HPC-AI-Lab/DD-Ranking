## Cutmix

Cutmix is a data augmentation technique that creates new samples by combining patches from two images while blending their labels proportionally to the area of the patches.. We follow the implementation of cutmix in [SRe2L](https://github.com/VILA-Lab/SRe2L/tree/main/SRe2L).

<div style="background-color:#F7F7F7; padding:15px; border:1px solid #E0E0E0; border-top:3px solid #FF0000; font-family:monospace; font-size:14px;">

<span style="color:#FF6B00;">CLASS</span> 
ddranking.aug.Cutmix(params: dict)
[**[SOURCE]**](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/blob/main/ddranking/aug/cutmix.py)

</div>

### Parameters

- **params**(<span style="color:#FF6B00;">dict</span>): Parameters for the cutmix augmentation. We require the parameters to be in the format of `{'param_name': param_value}`. For cutmix, only `beta` (beta distribution parameter) needs to be specified, e.g. `{'beta': 1.0}`.

### Example
    
```python
# When intializing an evaluator with cutmix augmentation, and cutmix object will be constructed.
>>> self.aug_func = Cutmix(params={'beta': 1.0})

# During training, the cutmix object will be used to augment the data.
>>> images = aug_func(images)
```