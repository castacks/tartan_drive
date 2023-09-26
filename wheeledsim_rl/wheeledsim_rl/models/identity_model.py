import torch

from torch import sin, cos, tan

from wheeledsim_rl.util.util import dict_to

class IdentityModel:
    """
    For debugging purposes, returns the input state for dynamics.
    """

    def __init__(self, hyperparams = {}, device='cpu'):
        self.device = device

    def dynamics(self, state, action):
        return torch.zeros_like(state)

    def forward(self, state, action, dt=0.1):
        return state

    def to(self, device):
        self.device=device
        return self

