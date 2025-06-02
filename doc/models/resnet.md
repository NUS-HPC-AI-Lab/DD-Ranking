## ResNet

DD-Ranking supports implementation of ResNet in both [DC](https://github.com/VICO-UoE/DatasetCondensation) and [torchvision](https://pytorch.org/vision/main/models/resnet.html).

We provide the following interface to initialize a ConvNet model:

<div style="background-color:#F7F7F7; padding:15px; border:1px solid #E0E0E0; border-top:3px solid #FF0000; font-family:monospace; font-size:14px;">

ddranking.utils.get_resnet(model_name: str, 
im_size: tuple, channel: int, num_classes: int, depth: int, batchnorm: bool, use_torchvision: bool, pretrained: bool, model_path: str)
[**[SOURCE]**](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/blob/main/ddranking/utils/model.py)
</div>

### Parameters

- **model_name**(<span style="color:#FF6B00;">str</span>): Name of the model. Please navigate to [models](models/overview.md) for the model naming convention in DD-Ranking.
- **im_size**(<span style="color:#FF6B00;">tuple</span>): Image size.
- **channel**(<span style="color:#FF6B00;">int</span>): Number of channels of the input image.
- **num_classes**(<span style="color:#FF6B00;">int</span>): Number of classes.
- **depth**(<span style="color:#FF6B00;">int</span>): Depth of the network.
- **batchnorm**(<span style="color:#FF6B00;">bool</span>): Whether to use batch normalization.
- **use_torchvision**(<span style="color:#FF6B00;">bool</span>): Whether to use torchvision to initialize the model. When using torchvision, the ResNet model uses batch normalization by default.
- **pretrained**(<span style="color:#FF6B00;">bool</span>): Whether to load pretrained weights.
- **model_path**(<span style="color:#FF6B00;">str</span>): Path to the pretrained model weights.

<div style="background-color:#40C4FF;color: #FFFFFF; padding: 5px; font-weight:bold; font-size:14px;">NOTE</div>

<div style="background-color:#F7F7F7; padding:15px; border:1px solid #E0E0E0; font-family:monospace; font-size:14px;">
When using torchvision ResNet on image size smaller than 224 x 224, we make the following modifications:

```python
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
model.maxpool = torch.nn.Identity()
```
</div>
