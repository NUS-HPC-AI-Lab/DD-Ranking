# MTT
Since the original MTT provides different hyperparameters for various settings, we list the commands required to reproduce each corresponding setting. For instructions on how to setup the code, please see [instructions](instructions.md). 

# Generate expert trajectories
MTT requires pre-obtaining and storing the parameter trajectories of the network trained on the full dataset. Therefore, we first need to run the following command. 

It is important to note that not all of the SOTA results in MTT use ZCA. Therefore, we strongly recommend that you refer to the instructions in the distillation data section to decide whether to apply ZCA when generating expert trajectories.

```
python buffer.py --dataset=CIFAR10 --train_epochs=50 --num_experts=100 --zca
# --dataset: CIFAR10, CIFAR100, TinyImageNet
```

# Train synthetic datasets
Some SOTA results are achieved with ZCA, some are not
## CIFAR-10
### IPC 1
```
python distill.py --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=2 --lr_img=100 --lr_lr=1e-07 --lr_teacher=0.01 --zca
```
### IPC 10
```
python distill.py --dataset=CIFAR10 --ipc=10 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.001 --zca
```
### IPC 50
```
python distill.py --dataset=CIFAR10 --ipc=50 --syn_steps=30 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.001
```
## CIFAR-100
### IPC 1
```
python distill.py --dataset=CIFAR100 --ipc=1 --syn_steps=20 --expert_epochs=3 --max_start_epoch=20 --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --zca
```
### IPC 10
```
python distill.py --dataset=CIFAR100 --ipc=10 --syn_steps=20 --expert_epochs=2 --max_start_epoch=20 --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01
```
### IPC 50
```
python distill.py --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --batch_syn=125 --zca
```
### TinyImagenet
### IPC 1
```
python distill.py --dataset=Tiny --ipc=1 --syn_steps=10 --expert_epochs=2 --max_start_epoch=10 --lr_img=10000 --lr_lr=1e-04 --lr_teacher=0.01
```
### IPC 10
```
python distill.py --dataset=Tiny --ipc=10 --syn_steps=20 --expert_epochs=2 --max_start_epoch=40 --lr_img=10000 --lr_lr=1e-04 --lr_teacher=0.01 --batch_syn 200
```
### IPC 50
```
python distill.py --dataset=Tiny --ipc=50 --syn_steps=20 --expert_epochs=2 --max_start_epoch=40 --lr_img=10000 --lr_lr=1e-04 --lr_teacher=0.01 --batch_syn=300 
```
