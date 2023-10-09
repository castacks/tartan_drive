import gym
import torch

from wheeledsim_rl.replaybuffers.simple_replaybuffer import SimpleReplayBuffer
from wheeledsim_rl.util.util import dict_to

class NStepReplayBuffer(SimpleReplayBuffer):
    """
    Replay buffer that supports n-step indexing.
    """
    def __init__(self, env, capacity = int(1e7), device='cpu'):
        super(NStepReplayBuffer, self).__init__(env, capacity, device=device)

    def sample(self, nsamples, N):
        """
        Get a batch of samples from the replay buffer.
        Index output as: [batch x time x feats]
        """
        #Don't want to sample placeholders, so min n and capacity.
        sample_idxs = self.compute_sample_idxs(nsamples, N)

        idxs = sample_idxs[torch.randint(len(sample_idxs), size=(nsamples, ))]

        out = {k:torch.stack([self.buffer[k][(idxs + i) % len(self)] for i in range(N)], dim=1) for k in self.buffer.keys()}
        
        out = dict_to(out, self.device)

        return out

    def compute_sample_idxs(self, nsamples, N):
        all_idxs = torch.arange(len(self), device=self.device)
        terminal_idxs = torch.nonzero(self.buffer['terminal'][:len(self)], as_tuple=False)[:, 0]
        non_sample_idxs = torch.tensor([], device=self.device)
        for i in range(N-1):
            non_sample_idxs = torch.cat([non_sample_idxs, terminal_idxs - i])

        #https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
        combined = torch.cat((all_idxs, non_sample_idxs))
        uniques, counts = combined.unique(return_counts=True)
        sample_idxs = uniques[counts == 1]

        return sample_idxs.long()
