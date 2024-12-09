import torch
import numpy as np
from utils import train_one_epoch, validate
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR


def compute_hard_label_metrics(syn_images, model, ipc, num_classes, device=torch.device('cuda')):
    # default hyperparameters
    num_epochs = 100
    lr = 0.01
    batch_size = 256

    hard_labels = torch.tensor([np.ones(ipc) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False).view(-1)
    hard_label_dataset = TensorDataset(syn_images.detach().clone(), hard_labels)

    model.to(device)
    train_loader = DataLoader(hard_label_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))

    best_acc1 = 0
    best_epoch = 0
    for epoch in range(num_epochs):
        train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, device, lr_scheduler=lr_scheduler, grad_accum_steps=1, log_interval=10)
        metrics = validate(model, train_loader, loss_fn, device)
        if metrics['acc1'] > best_acc1:
            best_acc1 = metrics['acc1']
            best_epoch = epoch
    
    print(f"Best accuracy: {best_acc1} at epoch {best_epoch}")
    
    return best_acc1, best_epoch