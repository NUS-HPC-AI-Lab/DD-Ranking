import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import KLDivLoss


class KLDivergenceLoss(nn.Module):
    def __init__(self, temperature=1.2):
        super(KLDivergenceLoss, self).__init__()
        self.temperature = temperature
        self.kl = KLDivLoss(reduction='batchmean')

    def forward(self, stu_outputs, tea_outputs):
        stu_probs = F.log_softmax(stu_outputs / self.temperature, dim=1)
        tea_probs = F.softmax(tea_outputs / self.temperature, dim=1)
        loss = self.kl(stu_probs, tea_probs)
        return loss