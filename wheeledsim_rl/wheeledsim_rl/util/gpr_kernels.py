"""
Collection of kernels to try for Gaussian Processes
"""

import torch

def sq_exp_kernel(x1, x2, k):
    """
    Does the squared exponential kernel, i.e. e^(-k * L2(x1, x2))
    Args:
        x1: batched tensor of xs
        x2: batched tensor of xs
        k: scale param influencing the recpetive field of the kernel
    Returns: The kernel matrix
    """
    l2s = (x1.unsqueeze(0) - x2.unsqueeze(1)).pow(2).sum(dim=2).sqrt()
    return (-k * l2s).exp()
