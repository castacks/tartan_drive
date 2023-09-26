import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
Implementation of a continuous trajlib that you can take gradients through.
Simple implementation: Have a set of k action sequences and interpolate them.
"""

def trajlib_action_sequences(T=20, throttle_n=5):
    """
    Pick k points in time to start the steer.
    2 throttle directions, 2 steer directions = 2x2xthrottle_n + 1 total trajs.
    """
    throttle_tidxs = torch.linspace(0, 1, throttle_n+1)[:-1]
    throttle_tidxs = (T * throttle_tidxs).long()

    sequences = torch.zeros(2*2*throttle_n, T, 2)

    for i, tidx in enumerate(throttle_tidxs):
        ii = 4 * i

        #Change throttle
        sequences[ii, :, 0] = 1.
        sequences[ii+1, :, 0] = 1.
        sequences[ii+2, :, 0] = -1.
        sequences[ii+3, :, 0] = -1.

        #Change steer
        sequences[ii, tidx:, 1] = 1.
        sequences[ii+1, tidx:, 1] = -1.
        sequences[ii+2, tidx:, 1] = 1.
        sequences[ii+3, tidx:, 1] = -1.

    return torch.cat([sequences, torch.zeros_like(sequences[[0]])], dim=0)

class InterpolationTrajlib:
    """
    Trajlib class that generates continuous trajs by interpolating between a set of reference trajs
    """
    def __init__(self, sequences, device='cpu'):
        self.sequences = sequences
        self.action_dim = self.sequences.shape[0]
        self.device = device

    def get_sequence(self, z):
        if len(z.shape) <= 1:
            return self.get_sequence(z.unsqueeze(0)).squeeze()

        _z = self.normalize(z)

        trajs = (_z.unsqueeze(-1).unsqueeze(-1) * self.sequences.unsqueeze(0)).sum(dim=1)

        return trajs

    def normalize(self, z, softmax=True):
        """
        Normalize z to sum to 1. (I'm just taking softmax, could experiment w/ linear normalization)
        """
        if softmax:
            return (z + 1e-6).softmax(dim=-1)
        else:
            mins = z.min(dim=-1, keepdim=True)[0]
            _z = z - mins
            _z /= (_z + 1e-6).sum(dim=-1, keepdim=True)
            return _z

    def to(self, device):
        self.device = device
        self.sequences = self.sequences.to(device)
        return self

class InterpolationTrajlibExplorationPolicy:
    """
    Given a low-level model ensemble, (i.e. f(s, a), where a sequence is a series of actions), choose the sequence with the highest uncertainty.
    """
    def __init__(self, env, action_sequences, model, itrs=10, opt_class=torch.optim.Adam, opt_kwargs={'lr':0.1}, device='cpu'):
        self.act_dim = env.action_space.shape[-1]
        assert self.act_dim == action_sequences.shape[-1], "Expected action sequences of dim {}, got {}".format(self.act_dim, action_sequences.shape[-1])
#        self.sequences = action_sequences
        self.trajlib = InterpolationTrajlib(action_sequences)
        self.T = action_sequences.shape[1]
        self.t = 0
        self.current_sequence = None
        self.model = model

        self.itrs = itrs
        self.opt_class = opt_class
        self.opt_kwargs = opt_kwargs

        self.device = device

    def to(self, device):
        self.device = device
        self.trajlib = self.trajlib.to(device)
        self.model = self.model.to(device)
        return self

    def action(self, obs, deterministic=False):
        if self.t % self.T == 0:
            self.current_sequence = self.select_sequence(obs)
            print('SEQ = {}'.format(self.current_sequence))
            self.t = 0

        act = self.current_sequence[self.t]
        self.t += 1
        return act.to(self.device)

    def select_sequence(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = dict_to_torch(obs, device=self.device)
            obs = {k:v.unsqueeze(0) for k,v in obs.items()}

        z = torch.rand(1, self.trajlib.action_dim, requires_grad=True)
        opt = self.opt_class([z], **self.opt_kwargs)

        for i in range(self.itrs):
            seq = self.trajlib.get_sequence(z)
            preds = self.rollout(obs, seq) #[seq x time x ensemble x state]
            unc = preds[:, -1].std(dim=-2).sum(dim=-1)
            loss = -unc.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            seq = self.trajlib.get_sequence(z).squeeze()

        return seq

    def rollout(self, state, seqs):
        """
        Roll out seqs from the start state (always return ensemble predictions)
        """
        cstate = state.repeat(seqs.shape[0], self.model.n_models, 1)
        preds = []
        for i in range(seqs.shape[1]):
            act = seqs[:, [i]].repeat(1, self.model.n_models, 1)
            cstate = self.model.predict(cstate.view(-1, state.shape[-1]), act.view(-1, self.act_dim))
#            cstate = cstate.view(self.model.n_models, self.model.n_models, state.shape[-1]).permute(1, 0, 2) #Re-batch to [pdim x edim x edim x statedim]
            cstate = cstate.view(self.model.n_models, seqs.shape[0], self.model.n_models, state.shape[-1]).permute(1, 0, 2, 3) #Re-batch to [pdim x edim x edim x statedim]
            cstate = cstate[:, torch.arange(self.model.n_models), torch.arange(self.model.n_models)]

            preds.append(cstate)

        preds = torch.stack(preds, dim=1) #batched as [seqdim x time x edim x statedim]
        return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--throttle_n', type=int, required=True, help='Number of throttle idxs to consider')
    parser.add_argument('--T', type=int, required=True, help='Number of timesteps per sequence')

    args = parser.parse_args()

    seqs = trajlib_action_sequences(T=args.T, throttle_n=args.throttle_n)
    torch.save(seqs, "interp_trajlib_seqs.pt")
    trajlib = InterpolationTrajlib(seqs)

    z = torch.zeros(trajlib.action_dim)

    for i in range(5):
        print("_" * 30)
        z = torch.rand(trajlib.action_dim)
        seq = trajlib.get_sequence(z)
        print("Z = {}".format(z))
        print("seq = {}".format(seq))

    z = torch.zeros(trajlib.action_dim, requires_grad=True)
    opt = torch.optim.Adam([z], lr=1e0)

    for i in range(10):
        print("_______{}_______".format(i))

        seq = trajlib.get_sequence(z)
        loss = -seq.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            print("Z = {}".format(z))
            print("seq = {}".format(seq))
