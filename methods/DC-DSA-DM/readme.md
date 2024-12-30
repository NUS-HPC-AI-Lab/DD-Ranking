# DC-DSA-DM

We provide all the commands to reproduce the results of DC, DSA and DM. For instructions on how to setup the code, please see [instructions](instructions.md). 

# DC
```
python main.py  --dataset CIFAR10  --ipc 10 
# --dataset: CIFAR10, CIFAR100, TinyImageNet
# --ipc (images/class): 1, 10, 50
```
# DSA
```
python main.py  --dataset CIFAR10 --ipc 10  --method DSA  
# --dataset: CIFAR10, CIFAR100, TinyImageNet
# --ipc (images/class): 1, 10, 50
```
# DM
```
python main_DM.py  --dataset CIFAR10 --ipc 10 
# --dataset: CIFAR10, CIFAR100, TinyImageNet
# --ipc (images/class): 1, 10, 50
```

More details and the original commands can be found in [instructions](/instructions.md).
