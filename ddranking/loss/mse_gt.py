import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEGTLoss(nn.Module):
    def __init__(self, mse_weight=1.0, ce_weight=0.0):
        super(MSEGTLoss, self).__init__()
        self.mse_weight = mse_weight
        self.ce_weight = ce_weight

    def forward(self, stu_outputs, tea_outputs, ground_truth):
        mse_loss = F.mse_loss(stu_outputs, tea_outputs)
        ce_loss = F.cross_entropy(stu_outputs, ground_truth)
        loss = self.mse_weight * mse_loss + self.ce_weight * ce_loss
        return loss