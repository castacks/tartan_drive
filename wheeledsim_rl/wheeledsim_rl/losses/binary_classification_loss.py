import torch

"""
Note that the labels are floats
"""

class BinaryClassificationLoss:
    def __init__(self, scale=1.0):
        self.scale = 1.0

    def forward(self, preds, targets):
        #Assume preds are logits, squash to 0-1
        logits = preds.sigmoid()
        loss = torch.nn.functional.binary_cross_entropy(logits, targets)
        return loss * self.scale
