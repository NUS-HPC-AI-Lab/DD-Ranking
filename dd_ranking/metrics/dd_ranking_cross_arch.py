import os
import numpy as np
import torch
import optuna
from torch.utils.data import DataLoader
from dd_ranking.utils import get_dataset
from dd_ranking.utils import build_model, set_seed, train_one_epoch, validate


class Cross_Architecture_Evaluator:
    def __init__(self, arch_list: list, dataset: str, ipc: int, num_eval: int=5, device: str="cuda"):

        # default params for training a model
        self.batch_size = 256
        self.num_epochs = 300
        self.lr = 0.01

        self.ipc = ipc
        self.num_eval = num_eval
        self.device = device

        channel, im_size, num_classes, dst_train, dst_test, _, _ = get_dataset(dataset, real_data_path)
        self.num_classes = num_classes
        self.im_size = im_size
        self.test_loader = DataLoader(dst_test, batch_size=self.batch_size, shuffle=False)

        self.arch_list = arch_list

    def compute_metric_on_one_model(self, trial, model, train_loader):
        lr = trial.suggest_float('lr', 1e-3, 1e-1)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * self.num_epochs)

        best_top1 = 0
        for epoch in range(self.num_epochs):
            train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, lr_scheduler=scheduler, device=self.device)
            metrics = validate(model, self.test_loader, device=self.device)
            if metrics['top1'] > best_top1:
                best_top1 = metrics['top1']
            
        return best_top1

    def compute_metrics(self, images, labels=None):
        if labels is None:
            labels = torch.tensor([np.ones(self.ipc) * i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False).view(-1)
        train_dataset = TensorDataset(images, labels)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        model_to_metrics = {}
        for i, model_name in enumerate(self.arch_list):
            print("Evaluating model: ", model_name)
            cross_arch_metrics = []
            for _ in range(self.num_eval):
                set_seed()
                model = build_model(model_name, num_classes=self.num_classes, im_size=self.im_size, pretrained=False, device=self.device)
                study = optuna.create_study(direction='maximize')
                study.optimize(self.compute_metric_on_one_model, n_trials=20, model=model, train_loader=train_loader)
                cross_arch_metrics.append(study.best_trial.value)
                del model
        
            cross_arch_mean = np.mean(cross_arch_metrics, axis=0)
            cross_arch_std = np.std(cross_arch_metrics, axis=0)
            print(f"Cross-architecture mean: {cross_arch_mean}, std: {cross_arch_std}")
            model_to_metrics[model_name] = (cross_arch_mean, cross_arch_std)
        return model_to_metrics
