import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import copy

from wheeledsim_rl.util.util import dict_stack, dict_cat, dict_map

class Collector:
    """
    A basic collector class that uses a policy to collect transitions from an environment.
    """
    def __init__(self, env, policy, discount=0.99, torch_obs=False, device='cpu'):
        self.env = env
        self.dict_obs = isinstance(env.observation_space, gym.spaces.Dict)
#        self.obs_dim = env.observation_space.shape[0]
#        self.act_dim = env.action_space.shape[0]
        self.policy = policy
        self.discount = 0.99
        self.keys = [
            'observation',
            'action',
            'reward',
            'terminal',
            'next_observation'
        ]
        self.torch_obs = torch_obs
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def valid_traj(self, traj):
        """
        Do a quick sanity check on trajectories to avoid simulator errors.
        """
        if isinstance(traj['observation'], dict):
            nancheck = not any([torch.any(torch.isnan(traj['next_observation'][k])) for k in traj['next_observation'].keys()])
            widthcheck = torch.all(traj['next_observation']['state'][:, :2].abs() < 15.)
            heightcheck = torch.all(traj['next_observation']['state'][:, 3].abs() < 3.)
            lencheck = True
        else:
            nancheck = not torch.any(torch.isnan(traj['next_observation']))
            widthcheck = torch.all(traj['next_observation'][:, :2].abs() < 15.)
            heightcheck = torch.all(traj['next_observation'][:, 3].abs() < 3.)
            lencheck = True

        print('CHECKS: nan={}, width={}, height={}, len={}'.format(nancheck, widthcheck, heightcheck, lencheck))

        return nancheck and widthcheck and heightcheck and lencheck

    def collect_steps(self, n, finish_traj=True, deterministic=True, policy=None, recollect_traj=True):
        """
        Collect a certain number of steps from the environment.
        TODO: Use dict_stack to look at the traj-level. Then collect a list of trajs and cat them instead.
        """
        nsteps = 0
        buf = []
        ctraj = self.build_results()
        o = self.env.reset()
        t = False
        d = 1.0

        while nsteps < n or (finish_traj and (not t)):
            if policy is None:
                act = self.policy.action(o, deterministic=deterministic)
            else:
                act = policy.action(o, deterministic=deterministic)

            print('STEP {}: ACT = {}'.format(nsteps, act))

            no, r, t, i = self.env.step(act)

            self.add_results_to(ctraj, self.keys, [o, act, r, t, no] if self.torch_obs else self.to_torch([o, act, r, t, no]))

            o = copy.deepcopy(no)

            if t:
                print('RESET')
                traj = self.stack_results(ctraj)
                if self.valid_traj(traj):
                    buf.append(traj)
                else:
                    print('DISCARD TRAJ')
                    if recollect_traj:
                        nsteps -= len(ctraj['action'])

                o = self.env.reset()
                ctraj = self.build_results()

            nsteps += 1

        return dict_cat(buf, dim=0) if len(buf) > 0 else None

    def collect_trajs(self, n, deterministic=True, policy=None, recollect_traj=True, discard_traj=True):
        """
        Collect a certain number of rollouts from the environment.
        """
        ntrajs = 0
        nsteps = 0
        buf = []
        ctraj = self.build_results()
        o = self.env.reset()
        t = False
        d = 1.0

        while ntrajs < n:
            nsteps += 1

            if policy is None:
                act = self.policy.action(o, deterministic=deterministic)
            else:
                act = policy.action(o, deterministic=deterministic)

            print('TRAJ {}: ACT = {}'.format(ntrajs, act))
            no, r, t, i = self.env.step(act.detach().to('cpu'))

            self.add_results_to(ctraj, self.keys, [o, act, r, t, no] if self.torch_obs else self.to_torch([o, act, r, t, no]))

            o = copy.deepcopy(no)

            if t:
                print('RESET')
                traj = self.stack_results(ctraj)
                if self.valid_traj(traj) or not discard_traj:
                    buf.append(traj)
                else:
                    print('DISCARD TRAJ')
                    if recollect_traj:
                        ntrajs -= 1

                o = self.env.reset()
                ctraj = self.build_results()

                ntrajs += 1

        return dict_cat(buf, dim=0)  if len(buf) > 0 else None

    def stack_results(self, buf):
        if isinstance(buf, list):
            return {k:torch.stack([v[k] for v in buf], dim=0) for k in buf[0].keys()} if self.dict_obs and isinstance(buf[0], dict) else torch.stack(buf, dim=0)
        else:
            return {k:self.stack_results(v) for k, v in buf.items()}

    def to_torch(self, x):
        if isinstance(x, dict):
            return {k:self.to_torch(v) for k,v in x.items()}
        elif isinstance(x, list):
            return [self.to_torch(v) for v in x]
        elif isinstance(x, torch.Tensor):
            return x.clone()
        else:
            out = torch.tensor(x, device=self.device)
            if out.dtype == torch.float64:
                out = out.float()
            return out

    def add_results_to(self, buf, keys, data):
        for k, v in zip(keys, data):
            buf[k].append({kk:vv.clone().to(self.device) for kk, vv in v.items()} if self.dict_obs and isinstance(v, dict) else v.clone().to(self.device))

    def build_results(self):
        return {k:[] for k in self.keys}
