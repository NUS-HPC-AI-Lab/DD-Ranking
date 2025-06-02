## LeNet

Our [implementation](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/blob/main/ddranking/utils/networks.py) of LeNet is based on [DC](https://github.com/VICO-UoE/DatasetCondensation). 

We provide the following interface to initialize a LeNet model:

<div style="background-color:#F7F7F7; padding:15px; border:1px solid #E0E0E0; border-top:3px solid #FF0000; font-family:monospace; font-size:14px;">

ddranking.utils.get_lenet(model_name: str, im_size: tuple, channel: int, num_classes: int, pretrained: bool, model_path: str)
[**[SOURCE]**](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/blob/main/ddranking/utils/model.py)
</div>

### Parameters

- **model_name**(<span style="color:#FF6B00;">str</span>): Name of the model. Please navigate to [models](models/overview.md) for the model naming convention in DD-Ranking.
- **im_size**(<span style="color:#FF6B00;">tuple</span>): Image size.
- **channel**(<span style="color:#FF6B00;">int</span>): Number of channels of the input image.
- **num_classes**(<span style="color:#FF6B00;">int</span>): Number of classes.
- **pretrained**(<span style="color:#FF6B00;">bool</span>): Whether to load pretrained weights.
- **model_path**(<span style="color:#FF6B00;">str</span>): Path to the pretrained model weights.
