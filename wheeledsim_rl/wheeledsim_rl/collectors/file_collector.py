import torch

from os import path

from wheeledsim_rl.collectors.base_collector import Collector
from wheeledsim_rl.util.os_util import maybe_mkdir
from wheeledsim_rl.util.rl_util import split_trajs

class FileCollector(Collector):
    """
    Same as a normal collector, but in addition to returning the traj, also saves it to a file in a pre-specified dir.
    """
    def __init__(self, env, policy, base_fp, discount=0.99, torch_obs=False, device='cpu'):
        super(FileCollector, self).__init__(env, policy, discount, torch_obs, device)
        self.base_fp = base_fp
        maybe_mkdir(self.base_fp, force=False)
        self.trajcnt = 0

    def collect_steps(self, n, finish_traj=True, deterministic=True, policy=None, recollect_traj=True):
        res = super(FileCollector, self).collect_steps(n, finish_traj, deterministic, policy, recollect_traj)
        for traj in split_trajs(res):
            torch.save(traj, path.join(self.base_fp, 'traj_{}.pt'.format(self.trajcnt)))
            self.trajcnt += 1
        return res

    def collect_trajs(self, n, deterministic=True, policy=None, recollect_traj=True, discard_traj=True):
        res = super(FileCollector, self).collect_trajs(n, deterministic, policy, recollect_traj)
        for traj in split_trajs(res):
            torch.save(traj, path.join(self.base_fp, 'traj_{}.pt'.format(self.trajcnt)))
            self.trajcnt += 1
        return res
