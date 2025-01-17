# DATM

We provide all the commands to reproduce the DATM results. For instructions on how to setup the code, please see [instructions](instructions.md). 

# Generate expert trajectories
Similar to MTT, DATM requires pre-obtaining and storing the parameter trajectories of the network trained on the full dataset. Therefore, we first need to run the following command. 

While different from MTT, all the reported results are obtained with ZCA.
```
python ./buffer/buffer_FTD.py --dataset=CIFAR10 --train_epochs=100 --num_experts=100 --zca --rho_max=0.01 --rho_min=0.01 --alpha=0.3 --lr_teacher=0.01 --mom=0. --batch_train=256
# --dataset: CIFAR10, CIFAR100, TinyImageNet
```

# Train synthetic datasets
```
python ./distill/DATM.py --cfg ../configs/CIFAR10/ConvIN/IPC10.yaml
```

More details and the original commands can be found in [instructions](instructions.md).