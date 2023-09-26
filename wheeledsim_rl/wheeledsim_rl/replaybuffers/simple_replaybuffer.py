import gym
import torch

from wheeledsim_rl.util.util import dict_to

class SimpleReplayBuffer:
    """
    Generic replay buffer with nothing fancy
    """
    def __init__(self, env, capacity = int(1e7), device='cpu'):
        self.capacity = int(capacity)
        self.obs_dim = env.observation_space.shape[0]
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.n = 0 #the index to start insering into
        self.device = device

        #The actual buffer is a dict that stores torch tensors. 
        if self.discrete:
            self.act_dim = env.action_space.n
            self.buffer = {
                        'observation': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, self.obs_dim),
                        'action': torch.tensor([-1], device=self.device).repeat(self.capacity, 1),
                        'reward':torch.tensor([float('inf')], device=self.device).repeat(self.capacity, 1),
                        'next_observation': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, self.obs_dim),
                        'terminal': torch.tensor([True], device=self.device).repeat(self.capacity, 1)
                        }
        else:
            self.act_dim = env.action_space.shape[0]
            self.buffer = {
                        'observation': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, self.obs_dim),
                        'action': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, self.act_dim),
                        'reward':torch.tensor([float('inf')], device=self.device).repeat(self.capacity, 1),
                        'next_observation': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, self.obs_dim),
                        'terminal': torch.tensor([True], device=self.device).repeat(self.capacity, 1)
                        }

        self.to(self.device)

    def insert(self, samples):
        """
        Assuming samples are being passed in as a dict of tensors.
        """
        assert len(samples['observation']) == len(samples['action']) == len(samples['reward']) == len(samples['next_observation']) == len(samples['terminal']), \
        "expected all elements of samples to have same length, got: {} (\'returns\' should be a different length though)".format([(k, len(samples[k])) for k in samples.keys()])

        nsamples = len(samples['observation'])

        for k in self.buffer.keys():
            for i in range(nsamples):
                self.buffer[k][(self.n + i) % self.capacity] = samples[k][i]

        self.n += nsamples

    def __len__(self):
        return min(self.n, self.capacity)

    def sample_idxs(self, idxs):
        out = {k:self.buffer[k][idxs] for k in self.buffer.keys()}
        return out

    def sample(self, nsamples):
        """
        Get a batch of samples from the replay buffer.
        """
        #Don't want to sample placeholders, so min n and capacity.
        idxs = torch.LongTensor(nsamples).random_(0, min(self.n, self.capacity)) 

        out = self.sample_idxs(idxs)

        out = dict_to(out, self.device)        

        return out

    def to(self, device):
        self.device = device
        self.buffer = dict_to(self.buffer, self.device)
        return self

    def __repr__(self):
        return "buffer = {} \nn = {}\ndiscrete = {}".format(self.buffer, self.n, self.discrete)

