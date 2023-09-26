import torch

class MSELoss:
    def __init__(self, scale=1.0):
        self.scale = 1.0

    def forward(self, preds, targets):
        return (targets - preds).pow(2).mean() * self.scale
