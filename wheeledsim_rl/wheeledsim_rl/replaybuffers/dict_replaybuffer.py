import gym
import torch

from wheeledsim_rl.util.util import dict_stack, dict_to

class DictReplayBuffer:
    """
    Replay buffer for envs with dictionary observations. Assumes shallow dicts, though.
    """
    def __init__(self, env, capacity = int(1e7), device='cpu'):
        assert isinstance(env.observation_space, gym.spaces.Dict), 'Expects an env with dictionary observations'
        assert isinstance(env.action_space, gym.spaces.Box), 'Expects an env with continuous actions (not dictionary)'

        self.capacity = int(capacity)
        self.obs_dims  = {k:v.shape for k, v in env.observation_space.spaces.items()}
        self.n = 0 #the index to start insering into
        self.device = device

        #The actual buffer is a dict that stores torch tensors. 
        self.act_dim = env.action_space.shape[0]
        self.buffer = {
                    'observation': {k:torch.tensor([float('inf')], device=self.device).repeat(self.capacity, *space) for k, space in self.obs_dims.items()},
                    'action': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, self.act_dim),
                    'reward':torch.tensor([float('inf')], device=self.device).repeat(self.capacity, 1),
                    'next_observation': {k:torch.tensor([float('inf')], device=self.device).repeat(self.capacity, *space) for k, space in self.obs_dims.items()},
                    'terminal': torch.tensor([True], device=self.device).repeat(self.capacity, 1)
                    }

    def insert(self, samples):
        """
        Assuming samples are being passed in as a dict of tensors.
        """
        assert len(samples['action']) == len(samples['reward']) == len(samples['terminal']), \
        "expected all elements of samples to have same length, got: {} (\'returns\' should be a different length though)".format([(k, len(samples[k])) for k in samples.keys()])

        nsamples = len(samples['action'])

        for k in self.buffer.keys():
            if k == 'observation' or k == 'next_observation':
                for i in range(nsamples):
                    for kk in samples[k].keys():
                        self.buffer[k][kk][(self.n + i) % self.capacity] = samples[k][kk][i]
            else:
                for i in range(nsamples):
                    self.buffer[k][(self.n + i) % self.capacity] = samples[k][i]

        self.n += nsamples

    def __len__(self):
        return min(self.n, self.capacity)

    def sample_idxs(self, idxs):
        idxs = idxs.to(self.device)
        out = {k:{kk:self.buffer[k][kk][idxs] for kk in self.buffer[k].keys()} if (k == 'observation' or k == 'next_observation') else self.buffer[k][idxs] for k in self.buffer.keys()}
        return out

    def sample(self, nsamples, N=2):
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
        return "buffer = {} \nn = {}".format(self.buffer, self.n)

class NStepDictReplayBuffer(DictReplayBuffer):
    """
    Replay buffer that supports n-step indexing.
    """
    def __init__(self, env, capacity = int(1e7), device='cpu'):
        super(NStepDictReplayBuffer, self).__init__(env, capacity, device)
        self.to(self.device)

    def sample(self, nsamples, N):
        """
        Get a batch of samples from the replay buffer.
        Index output as: [batch x time x feats]
        """
        sample_idxs = self.compute_sample_idxs(nsamples, N)

        idxs = sample_idxs[torch.randint(len(sample_idxs), size=(nsamples, ))]

        outs = [self.sample_idxs((idxs + i) % len(self)) for i in range(N)]
        out = dict_stack(outs, dim=1)

        return out

    def compute_sample_idxs(self, nsamples, N):
        all_idxs = torch.arange(min(len(self), self.capacity)).to(self.device)
        terminal_idxs = torch.nonzero(self.buffer['terminal'][:len(self)], as_tuple=False)[:, 0]
        if self.n > self.capacity:
            all_idxs = torch.arange(terminal_idxs[-1]+1).to(self.device)

        non_sample_idxs = torch.tensor([]).long().to(self.device)
        for i in range(N-1):
            non_sample_idxs = torch.cat([non_sample_idxs, terminal_idxs - i])

        #https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
        combined = torch.cat((all_idxs, non_sample_idxs))
        uniques, counts = combined.unique(return_counts=True)
        sample_idxs = uniques[counts == 1]

        return sample_idxs.long()

if __name__ == '__main__':
    from offroad_env.AirsimEnv import SubTCarNavEnvMapObs

    env = SubTCarNavEnvMapObs(heightmap='../../../offroad_env/offroad_env/offroad_height_1111_224119.npy', period=0.5, enableimg=False)

    buf = DictReplayBuffer(env, capacity=1e4)

    print(buf)
    print(buf.obs_dims)
    print({k:v.shape for k,v in buf.buffer['observation'].items()})
