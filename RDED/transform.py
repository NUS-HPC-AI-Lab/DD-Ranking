import os
import torch.utils
from synthesize.utils import *
from validation.utils import ImageFolder

def create_imagefolder_structure(dataset, split='train', name='cifar100'):
    root_dir = f'./{name}/{split}'
    os.makedirs(root_dir, exist_ok=True)
    
    class_names = dataset.classes  # ['airplane', 'automobile', ...]
    
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(root_dir, 'n'+ str(idx))  
        os.makedirs(class_dir, exist_ok=True)
    
    for i in range(len(dataset)):
        image, label = dataset[i]
        class_name = str(label)  
        img_path = os.path.join(root_dir, 'n'+class_name, f'{i}.jpg') 
        image.save(img_path)


train_data = datasets.CIFAR100(root='./datasets', train=True, download=True)
test_data = datasets.CIFAR100(root='./datasets', train=False, download=True)
create_imagefolder_structure(train_data, split='train', name='cifar100')
create_imagefolder_structure(test_data, split='test', name='cifar100')

train_data = datasets.CIFAR10(root='./datasets', train=True, download=True)
test_data = datasets.CIFAR10(root='./datasets', train=False, download=True)
create_imagefolder_structure(train_data, split='train', name='cifar10')
create_imagefolder_structure(test_data, split='test', name='cifar10')