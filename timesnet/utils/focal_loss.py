import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=1)
        probs = torch.exp(log_probs)
        target_one_hot = F.one_hot(target, num_classes=input.size(1)).float()

        focal_weight = (1 - probs) ** self.gamma
        loss = -target_one_hot * focal_weight * log_probs

        if self.weight is not None:
            weight = self.weight.unsqueeze(0)  # (1, C)
            loss = loss * weight

        loss = loss.sum(dim=1)  # sum over classes
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss