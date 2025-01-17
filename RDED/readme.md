# RDED

We provide all the commands to reproduce the RDED results. For instructions on how to setup the code and details, please see [instructions](instructions.md). For all the datasets, you can find corresponding teacher model checkpoint at [model zoo](https://drive.google.com/drive/folders/1HmrheO6MgX453a5UPJdxPHK4UTv-4aVt?usp=drive_link).

### Storage Format for Raw Datasets

All the raw datasets, including those like ImageNet-1K and CIFAR10, store their training and validation components in the following format to facilitate uniform reading using a standard dataset class method:

```
/path/to/dataset/
├── 00000/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   └── image5.jpg
├── 00001/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   └── image5.jpg
├── 00002/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   └── image5.jpg
```

This organizational structure ensures compatibility with the unified dataset class, streamlining the process of data handling and accessibility. We provide code to transform the file structure of CIFAR10/100.

```
python transform.py
```

# CIFAR10
# ConvNet
``````bash
bash ./scripts/cifar10_conv3_to_conv3_cr5.sh
``````
# ResNet
``````bash
bash ./scripts/cifar10_resnet-18-modified_to_resnet-18-modified_cr5.sh
``````
# CIFAR100
# ConvNet
``````bash
bash ./scripts/cifar100_conv3_to_conv3_cr5.sh
``````
# ResNet
``````bash
bash ./scripts/cifar10_resnet-18-modified_to_resnet-18-modified_cr5.sh
``````
# TinyImageNet
# ConvNet
``````bash
bash ./scripts/tinyimagenet_conv4_to_conv4_cr5.sh
``````
# ResNet
``````bash
bash ./scripts/tinyimagenet_resnet-18-modified_to_resnet-18-modified_cr5.sh
``````

More details and the original commands can be found in [instructions](instructions.md).

