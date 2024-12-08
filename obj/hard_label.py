import torch
from torch.nn import CrossEntropyLoss


def compute_hard_label_metrics(distilled_dataset, model, ipc, num_classes, device):
    hard_labels = torch.tensor([np.ones(ipc) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False).view(-1)
    hard_label_dataset = TensorDataset(images.detach().clone(), hard_labels)
