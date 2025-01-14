import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceLoss(nn.Module):
    def __init__(self, temperature=1.2):
        super(KLDivergenceLoss, self).__init__()
        self.temperature = temperature

    def forward(self, stu_outputs, tea_outputs):
        stu_probs = F.log_softmax(stu_outputs / self.temperature, dim=1)
        with torch.no_grad():
            tea_probs = F.softmax(tea_outputs / self.temperature, dim=1)
        loss = F.kl_div(stu_probs, tea_probs, reduction='batchmean')
        return loss