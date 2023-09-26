import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

from os import path
from torch.utils.data import Dataset, DataLoader

from wheeledsim_rl.util.util import dict_to

class ObservationNormalizer:
    """
    Takes a set of trajectories and normalizes its inputs (to unit Gaussian, by default.)
    Basically, needs to take in a traj and produce the mean and standard deviation for each feature.
    Note that elems in an observation will be dicts or tensors.
    Also note that image observations are expected to be channels-last (as we take running avgs over last feature).
    """
    def __init__(self, fp=None, env=None, device='cpu'):
        if fp is not None:
            self.traj_base_dir = fp
            self.traj_fps = [f for f in os.listdir(self.traj_base_dir) if (path.isfile(path.join(self.traj_base_dir, f)) and f[-3:] == '.pt')]

            self.mean, self.std, self.data_types = self.initialize_stats()
            self.var = copy.deepcopy(self.std)
            self.N = 1
        elif env is not None:
            sample_obs = env.reset()
            sample_act = torch.tensor(env.action_space.low).float()
            self.mean, self.std, self.data_types = self.initialize_stats_helper({
                'observation':sample_obs,
                'action':sample_act,
                'next_observation':sample_obs
            })
            self.var = copy.deepcopy(self.std)
            self.N = 1

        self.device = device
        self.to(device)

    def to(self, device):
        self.device = device
        self.mean = dict_to(self.mean, device)
        self.std = dict_to(self.std, device)
        self.var = dict_to(self.std, device)
        return self

    def initialize_stats(self):
        traj = torch.load(path.join(self.traj_base_dir, self.traj_fps[0]))

        return self.initialize_stats_helper(traj)

    def initialize_stats_helper(self, obs):
        """
        Recurse to build the dict.
        """
        if isinstance(obs, dict):
            means = {k:self.initialize_stats_helper(obs[k])[0] for k in obs.keys()}
            stds = {k:self.initialize_stats_helper(obs[k])[1] for k in obs.keys()}
            dtypes = {k:self.initialize_stats_helper(obs[k])[2] for k in obs.keys()}
            return means, stds, dtypes
        else:
            #Bad code: Will check if image by looking if the last tensor is 4d [batch x channel x width x height]
            is_img = (len(obs.shape) == 3)
            if is_img:
                return torch.zeros(obs.shape[-3]), torch.ones(obs.shape[-3]), "image"
            else:
                return torch.zeros(obs.shape[-1]), torch.ones(obs.shape[-1]), "vector"

    def update(self, traj):
        """
        Use Welford's method to update the mean and standard deviation of all observations.
        """
        k = next(iter(traj['observation'].values())).shape[0] if isinstance(traj['observation'], dict) else traj['observation'].shape[0]

        self.mean, self.var, self.std = self.update_helper(traj, self.mean, self.var, self.std, self.data_types)
        self.N += k
        #Garbage code to keep all observations normalized the same way
        self.mean['next_observation'] = self.mean['observation']
        self.std['next_observation'] = self.std['observation']

    def update_helper(self, obs, mean, var, std, data_types):
        """
        Recurse to find all the tensors. Pass pointers to the current stats, too.
        """
        if isinstance(obs, dict):
            keys = mean.keys() & obs.keys()
            out_dict = {k:self.update_helper(obs[k], mean[k], var[k], std[k], data_types[k]) for k in keys}
            mean = {k:out_dict[k][0] for k in out_dict.keys()}
            var = {k:out_dict[k][1] for k in out_dict.keys()}
            std = {k:out_dict[k][2] for k in out_dict.keys()}
            return mean, var, std
        else:
            #Bad code: Will check if image by looking if the last tensor is 4d [batch x channel x width x height]
            if data_types == 'image':
                return self.apply_welford(obs.moveaxis(-3, -1), mean, var, std, self.N)
            else:
                return self.apply_welford(obs, mean, var, std, self.N)

    def apply_welford(self, data, mean, var, std, N):
        """
        Functional version of applying Welford's method.
        Args:
            data: The data to update stats with
            mean, var, std: Current stats of the data
            N: The number of pre-existing datapoints (that were previously used to calculate stats)
        Returns:
            mean, var, std: The new stats
        """
        x_flat = data.reshape(-1, data.shape[-1])
        k = x_flat.shape[0]
        
        #Use Welford's method in batch to update mean, SSD.
        mean_new = (mean + (x_flat.sum(dim=0) / N)) * (N / (N + k))
        x_ssd = (x_flat - mean_new).pow(2).sum(dim=0)
        mean_diff = mean - mean_new
        x_correction = (x_flat - mean_new).sum(dim=0)
        d_ssd = x_ssd + mean_diff * x_correction

        var_new = (var + (d_ssd / N)) * (N / (N + k))
        std_new = var_new.sqrt()
        
        return mean_new, var_new, std_new

    def normalize(self, obs, clip=None):
        return self.normalize_helper(obs, self.mean, self.std, clip=clip, data_types=self.data_types)

    def normalize_helper(self, obs, mean, std, clip, data_types):
        """
        Recurse to find all the tensors. Pass pointers to the current stats, too.
        """
        if isinstance(obs, dict):
            return {k:self.normalize_helper(obs[k], mean[k], std[k], clip, data_types[k]) if k in mean.keys() else obs[k] for k in obs.keys()}
        elif isinstance(obs, torch.distributions.Normal):
            mean = self.unsqueeze_front(mean, obs.loc)
            std = self.unsqueeze_front(std, obs.loc)
#            out = torch.distributions.Normal(loc=obs.mean - mean, scale=obs.scale/std)
            out = torch.distributions.Normal(loc=obs.mean - mean, scale=obs.scale)
            return out
        else:
            if data_types == 'image':
                obs = obs.swapdims(-1, -3).swapdims(-2, -3)

            mean = self.unsqueeze_front(mean, obs)
            std = self.unsqueeze_front(std, obs)

            is_one_hot = torch.all(obs.sum(dim=-1) == 1)
            if is_one_hot:
                return obs

            out = (obs - mean) / std

            if data_types == 'image':
                out = out.swapdims(-1, -2).swapdims(-2, -3)

            if clip is not None:
                out = out.clamp(-clip, clip)

            return out

    def denormalize(self, obs):
        return self.denormalize_helper(obs, self.mean, self.std, self.data_types)

    def denormalize_helper(self, obs, mean, std, data_types):
        """
        Recurse to find all the tensors. Pass pointers to the current stats, too.
        """
        if isinstance(obs, dict):
            return {k:self.denormalize_helper(obs[k], mean[k], std[k], data_types[k]) if k in mean.keys() else obs[k] for k in obs.keys()}
        elif isinstance(obs, torch.distributions.Normal):
            mean = self.unsqueeze_front(mean, obs.loc)
            std = self.unsqueeze_front(std, obs.loc)
#            out = torch.distributions.Normal(loc=obs.mean + mean, scale=obs.scale*std)
            out = torch.distributions.Normal(loc=obs.mean + mean, scale=obs.scale)
            return out
        else:
            if data_types == 'image':
                obs = obs.swapdims(-1, -3).swapdims(-2, -3)

            mean = self.unsqueeze_front(mean, obs)
            std = self.unsqueeze_front(std, obs)

            is_one_hot = torch.all(obs.sum(dim=-1) == 1)
            if is_one_hot:
                return obs

            out = (obs * std) + mean

            if data_types == 'image':
                out = out.swapdims(-1, -2).swapdims(-2, -3)

            return out

    def unsqueeze_front(self, x, obs):
        """
        Re-dimensions the data tensors to be the right shape.
        """
        out = x.clone()
        while len(out.shape) < len(obs.shape):
            out = out.unsqueeze(0)
        return out

    def dump(self):
        print('MEAN:')
        self.dump_helper(self.mean, 1)
        print('STD:')
        self.dump_helper(self.std, 1)

    def dump_helper(self, obs, depth):
        if isinstance(obs, dict):
            for k in obs.keys():
                print('{}{}:'.format('\t'*depth, k))
                self.dump_helper(obs[k], depth + 1)
        else:
            print('\t'*depth,  obs)


if __name__ == '__main__':
    traj_fp = 'blah'
    normalizer = ObservationNormalizer(traj_fp)
    for i, fp in enumerate(normalizer.traj_fps):
        print(i, end='\r')
        normalizer.update(torch.load(path.join(normalizer.traj_base_dir, fp)))
        if i > 10:
            break
    normalizer.dump()

    traj = torch.load(path.join(normalizer.traj_base_dir, fp))

    ntraj = normalizer.normalize(traj) 
    dtraj = normalizer.denormalize(ntraj)

    #Do some checks. Look at distribution of ntraj, compare traj and dtraj
    print('TRAJ =? DTRAJ?')
    def check(o1, o2, d):
        if isinstance(o1, dict):
            for k in o1.keys():
                print('{}{}'.format('\t'*d, k))
                check(o1[k], o2[k], d+1)
        else:
            print('{}{}, err={}'.format('\t'*d, torch.allclose(o1, o2), (o1-o2).pow(2).sum().sqrt()))

    check(traj, dtraj, 0)

    print('NTRAJ UNIT GAUSSIAN?')
    def check2(o1, d):
        if isinstance(o1, dict):
            for k in o1.keys():
                print('{}{}'.format('\t'*d, k))
                check2(o1[k], d+1)
        else:  
            dims = tuple(torch.arange(len(o1.shape)-1))
            print('{}MEAN={}, STD={}'.format('\t'*d, o1.mean(dim=dims), o1.std(dim=dims)))

    check2(ntraj, 0)
