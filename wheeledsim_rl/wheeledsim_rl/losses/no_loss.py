import torch

class NoLoss:
    """
    Loss function that returns nothing. Useful to have input modalities without requiring supervision.
    """
    def __init__(self, scale=1.0):
        self.scale = 1.0

    def forward(self, preds, targets):
        return preds.mean() * 0.
