import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=1.2, scale_loss=False):
        super(SoftCrossEntropyLoss, self).__init__()
        self.temperature = temperature
        self.scale_loss = scale_loss

    def forward(self, stu_outputs, tea_outputs):
        input_log_likelihood = -F.log_softmax(stu_outputs / self.temperature, dim=1)
        target_log_likelihood = F.softmax(tea_outputs / self.temperature, dim=1)
        batch_size = stu_outputs.size(0)
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch_size
        if self.scale_loss:
            loss = loss * (self.temperature ** 2)
        return loss