import gym
import torch

from wheeledsim_rl.util.util import dict_stack, dict_to
from wheeledsim_rl.util.preprocess_map import preprocess_map

class MapReplayBuffer:
    """
    Replay buffer for envs with dictionary observations and maps. Assumes shallow dicts, though.
    Unlike other replay buffers, this one also expects to recieve a list of maps and mapidxs with its batch.
    Unfortunately, we lose the ability to allocate all memory ahead of time.
    """
    def __init__(self, env, map_preprocess_dict, capacity = int(1e7), device='cpu'):
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
                    'terminal': torch.tensor([True], device=self.device).repeat(self.capacity, 1),
                    'map_idx': torch.tensor([-1], device=self.device).repeat(self.capacity, 1)
                    }

        self.preprocess_dict = map_preprocess_dict
        self.n_maps = 0
        self.maps = []

    def insert(self, samples):
        """
        Assuming samples are being passed in as a dict of tensors.
        """
        assert len(samples['action']) == len(samples['reward']) == len(samples['terminal']) == len(samples['map_idx']), \
        "expected all elements of samples to have same length, got: {} (\'returns\' should be a different length though)".format([(k, len(samples[k])) for k in samples.keys()])

        nsamples = len(samples['action'])

        for k in self.buffer.keys():
            if k == 'observation' or k == 'next_observation':
                for i in range(nsamples):
                    for kk in samples[k].keys():
                        self.buffer[k][kk][(self.n + i) % self.capacity] = samples[k][kk][i]
            elif k == 'map_idx':
                for i in range(nsamples):
                    self.buffer[k][(self.n + i) % self.capacity] = samples[k][i] + self.n_maps
            else:
                for i in range(nsamples):
                    self.buffer[k][(self.n + i) % self.capacity] = samples[k][i]

        self.maps.extend(samples['map'])
        self.n_maps += len(samples['map'])

        self.n += nsamples

    def __len__(self):
        return min(self.n, self.capacity)

    def sample_idxs(self, idxs, with_maps=True):
        idxs = idxs.to(self.device)
        out = {k:{kk:self.buffer[k][kk][idxs] for kk in self.buffer[k].keys()} if (k == 'observation' or k == 'next_observation') else self.buffer[k][idxs] for k in self.buffer.keys()}

        if with_maps:
            out['map'] = self.process_maps(out)

        return out

    def process_maps(self, batch):
        maps = [self.maps[i] for i in batch['map_idx']]
        preprocessed_maps = [preprocess_map(m, self.preprocess_dict) for m in maps]
        nchannels = [len(m) for m in preprocessed_maps[0]]
        maps = [torch.cat(m, dim=0) for m in preprocessed_maps]
        out = {
            'map': torch.stack(maps, dim=0),
            'metadata':{
                'resolution': torch.tensor(self.preprocess_dict['resolution']),
                'size': torch.tensor(self.preprocess_dict['size']),
                'rgb_channels': torch.tensor(range(0, nchannels[0])),
                'height_channels': torch.tensor(range(nchannels[0], sum(nchannels[:2]))),
                'seg_channels': torch.tensor(range(sum(nchannels[:2]), sum(nchannels[:3]))),
                'mask_channels': torch.tensor(range(sum(nchannels[:3]), sum(nchannels)))
            }
        }
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

class NStepMapReplayBuffer(MapReplayBuffer):
    """
    Replay buffer that supports n-step indexing.
    """
    def __init__(self, env, map_preprocess_dict, capacity = int(1e7), device='cpu'):
        super(NStepMapReplayBuffer, self).__init__(env, map_preprocess_dict, capacity, device)
        self.to(self.device)

    def sample(self, nsamples, N):
        """
        Get a batch of samples from the replay buffer.
        Also, only need to process the map for the first step.
        Index output as: [batch x time x feats]
        """
        sample_idxs = self.compute_sample_idxs(nsamples, N)

        idxs = sample_idxs[torch.randint(len(sample_idxs), size=(nsamples, ))]

        outs = [self.sample_idxs(idxs + i, with_maps=False) for i in range(N)]

        out = dict_stack(outs, dim=1)
        out['map'] = self.process_maps(outs[0])

        return out

    def compute_sample_idxs(self, nsamples, N):
        all_idxs = torch.arange(len(self)).to(self.device)
        terminal_idxs = torch.nonzero(self.buffer['terminal'][:len(self)], as_tuple=False)[:, 0]
        non_sample_idxs = torch.tensor([]).long().to(self.device)
        for i in range(N-1):
            non_sample_idxs = torch.cat([non_sample_idxs, terminal_idxs - i])

        #https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
        combined = torch.cat((all_idxs, non_sample_idxs))
        uniques, counts = combined.unique(return_counts=True)
        sample_idxs = uniques[counts == 1]

        return sample_idxs.long()

if __name__ == '__main__':
    import numpy as np
    import os

    from wheeledsim_rl.util.ouNoise import ouNoise
    from wheeledsim_rl.policies.ou_policy import OUPolicy
    from wheeledsim_rl.collectors.map_collector import MapCollector

    from offroad_env.make_parallel_envs import *
    from offroad_env.NavSuite import NavSuite

    parser = argparse.ArgumentParser()

    parser.add_argument('--env_executable', type=str, required=True, help='Path to the executable for the env')
    parser.add_argument('--env_config_dir', type=str, required=True, help='dir containing env config files')
    parser.add_argument('--heightmap_fp', type=str, required=True, help='path to the heightmap for the env')
    parser.add_argument('--env_T', type=int, required=False, default=150, help='Max number of timesteps per rollout')
    parser.add_argument('--env_dt', type=float, required=True, help='Time for each timestep')

    args = parser.parse_args()

    processes = initialize_simulators(1, args.env_executable, display=True, speed=1.0)
    env = NavSuite(period=args.env_dt, config_dir = args.env_config_dir, heightmap_fp=args.heightmap_fp, enableimg=False, T=args.env_T)
    noise = ouNoise(lowerlimit=env.action_space.low, upperlimit=env.action_space.high, var = np.array([0.01, 0.01]), offset=np.zeros([2, ]), damp=np.ones([2, ])*1e-4)
    policy = OUPolicy(noise)
    collector = MapCollector(env, policy)

    pdict = torch.load(os.path.join(args.env_config_dir, 'preprocess_dict.pt'))
    buf = NStepMapReplayBuffer(env, pdict, capacity=100)

    import pdb;pdb.set_trace()
    print(buf)

    buf.insert(collector.collect_steps(10))

    print(buf)

    buf.insert(collector.collect_trajs(1))

    print(buf)
    batch = buf.sample(5, N=3)
    print(batch)
    import pdb;pdb.set_trace()
