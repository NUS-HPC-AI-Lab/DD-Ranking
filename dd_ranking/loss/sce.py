import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, stu_outputs, tea_outputs):
        input_log_likelihood = -F.log_softmax(stu_outputs, dim=1)
        target_log_likelihood = F.softmax(tea_outputs, dim=1)
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch_size
        return loss