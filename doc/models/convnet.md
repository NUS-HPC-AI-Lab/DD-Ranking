## ConvNet

Our [implementation](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/blob/main/ddranking/utils/networks.py) of ConvNet is based on [DC](https://github.com/VICO-UoE/DatasetCondensation). 

By default, we use width 128, average pooling, and ReLU activation. We provide the following interface to initialize a ConvNet model:

<div style="background-color:#F7F7F7; padding:15px; border:1px solid #E0E0E0; border-top:3px solid #FF0000; font-family:monospace; font-size:14px;">

dd_ranking.utils.get_convnet(model_name: str, 
im_size: tuple, channel: int, num_classes: int, net_depth: int, net_norm: str, pretrained: bool, model_path: str)
[**[SOURCE]**](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/blob/main/ddranking/utils/model.py)
</div>

### Parameters

- **model_name**(<span style="color:#FF6B00;">str</span>): Name of the model. Please navigate to [models](models/overview.md) for the model naming convention in DD-Ranking.
- **im_size**(<span style="color:#FF6B00;">tuple</span>): Image size.
- **channel**(<span style="color:#FF6B00;">int</span>): Number of channels of the input image.
- **num_classes**(<span style="color:#FF6B00;">int</span>): Number of classes.
- **net_depth**(<span style="color:#FF6B00;">int</span>): Depth of the network.
- **net_norm**(<span style="color:#FF6B00;">str</span>): Normalization method. In ConvNet, we support `instance`, `batch`, and `group` normalization.
- **pretrained**(<span style="color:#FF6B00;">bool</span>): Whether to load pretrained weights.
- **model_path**(<span style="color:#FF6B00;">str</span>): Path to the pretrained model weights.

To load a ConvNet model with different width or activation function or pooling method, you can use the following interface:

<div style="background-color:#F7F7F7; padding:15px; border:1px solid #E0E0E0; border-top:3px solid #FF0000; font-family:monospace; font-size:14px;">

dd_ranking.utils.networks.ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size)
[**[SOURCE]**](https://github.com/NUS-HPC-AI-Lab/DD-Ranking/blob/main/dd_ranking/utils/networks.py)
</div>

### Parameters
We only list the parameters that are not present in `get_convnet`.
- **net_width**(<span style="color:#FF6B00;">int</span>): Width of the network.
- **net_act**(<span style="color:#FF6B00;">str</span>): Activation function. We support `relu`, `leakyrelu`, and `sigmoid`.
- **net_pooling**(<span style="color:#FF6B00;">str</span>): Pooling method. We support `avgpooling`, `maxpooling`, and `none`.

