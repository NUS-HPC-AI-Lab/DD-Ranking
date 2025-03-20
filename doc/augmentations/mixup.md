## Mixup

Mixup is a data augmentation technique that generates new training samples by linearly interpolating pairs of images. We follow the implementation of mixup in [SRe2L](https://github.com/VILA-Lab/SRe2L/tree/main/SRe2L).

<div style="background-color:#F7F7F7; padding:15px; border:1px solid #E0E0E0; border-top:3px solid #FF0000; font-family:monospace; font-size:14px;">

<span style="color:#FF6B00;">CLASS</span> 
ddranking.aug.Mixup(params: dict)
[**[SOURCE]**](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/blob/main/ddranking/aug/mixup.py)

</div>

### Parameters

- **params**(<span style="color:#FF6B00;">dict</span>): Parameters for the mixup augmentation. We require the parameters to be in the format of `{'param_name': param_value}`. For mixup, only `lambda` (mixup strength) needs to be specified, e.g. `{'lambda': 0.8}`.

### Example

```python
# When intializing an evaluator with mixup augmentation, and mixup object will be constructed.
>>> self.aug_func = Mixup(params={'lambda': 0.8})

# During training, the mixup object will be used to augment the data.
>>> images = aug_func(images)
```