import torch

class OUPolicy:
    """
    Take actions according to OU Noise
    """
    def __init__(self, noisegen, noise_steps=20):
        """
        Args:
            noisegen: ouNoise class used to generate the noise
            noise_steps: make this many OU steps between noise samplings.
        """
        self.noisegen = noisegen
        self.noise_steps = noise_steps
        self.t = 0
        self.T = 0

    def action(self, obs, deterministic=False):
        return torch.tensor(self.noisegen.multiGenNoise(self.noise_steps)).float()

    def actions(self, obses, deterministic=False):
        """
        Copy the action to keep the Ou-ness of the policy.
        """
        return torch.tensor(self.noisegen.multiGenNoise(self.noise_steps)).repeat(obses.shape[0], 1).float()
