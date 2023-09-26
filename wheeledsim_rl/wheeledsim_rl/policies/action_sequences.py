import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

"""
Implement some basic open-loop action sequences and check the performance.
"""

def generate_action_sequences(throttle=(-1, 1), throttle_n=5, steer=(-1, 1), steer_n=5, t=10):
    throttles = torch.linspace(throttle[0], throttle[1], throttle_n)
    steers = torch.linspace(steer[0], steer[1], steer_n)

    acts = torch.meshgrid(throttles, steers)

    acts = torch.stack([acts[0], acts[1]], dim=-1).flatten(end_dim=-2)

    return acts.unsqueeze(1).repeat(1, t, 1)

def generate_action_sequences_2(throttle=(-1, 1), throttle_n=5, steer=(-1, 1), steer_n=5, t=10):
    assert t%2 == 0, "Need even t"
    base_seqs = generate_action_sequences(throttle, throttle_n, steer, steer_n, t//2).reshape(throttle_n, steer_n, -1, 2)
    flip_seqs = base_seqs.flip(dims=(1, )) #Flip the steer value

    same_seqs = torch.cat([base_seqs, base_seqs], dim=2).flatten(end_dim=1)
    diff_seqs = torch.cat([base_seqs, flip_seqs], dim=2).flatten(end_dim=1)

    all_seqs =  torch.cat([same_seqs, diff_seqs], dim=0)
    filter_seqs = all_seqs[(all_seqs[:, 0, 0] != 0) | torch.all(all_seqs[:, 0, :] == 0, dim=-1)] #Remove extra sequences w/zero throttle

    return filter_seqs

class RandomActionSequencePolicy:
    """
    Policy that acts by selecting an action sequence to run for k timesteps.
    To keep consistency with other non-hierarchical methods, provide the switching rate.
    Store current action sequence and step once every time action is called.

    action_sequences expected to be a tensor of size: [choicedim x timedim x actdim]
    """
    def __init__(self, env, action_sequences):
        self.goal = torch.zeros(2)
        self.act_dim = env.action_space.shape[-1]
        assert self.act_dim == action_sequences.shape[-1], "Expected action sequences of dim {}, got {}".format(self.act_dim, action_sequences.shape[-1])
        self.sequences = action_sequences
        self.T = action_sequences.shape[1]
        self.t = 0
        self.current_sequence = None
        self.n_seqs = self.sequences.shape[0]

    def action(self, obs, deterministic=False):
        if self.t % self.T == 0:
            self.seq_idx = torch.randint(self.n_seqs, size=(1,)).item()
            self.current_sequence = self.sequences[self.seq_idx]
            print('IDX = {}'.format(self.seq_idx))
            print('SEQ = {}'.format(self.current_sequence))
            self.t = 0

        act = self.current_sequence[self.t]
        self.t += 1
        return act

class ActionSequenceGoalPolicy:
    """
    Given a model of where each action sqeuence will take you, given an obs, choose the sequence that minimizes your goal distance.
    """
    def __init__(self, env, action_sequences, model, input_normalizer):
        self.goal = torch.zeros(2)
        self.act_dim = env.action_space.shape[-1]
        assert self.act_dim == action_sequences.shape[-1], "Expected action sequences of dim {}, got {}".format(self.act_dim, action_sequences.shape[-1])
        self.sequences = action_sequences
        self.T = action_sequences.shape[1]
        self.t = 0
        self.current_sequence = None
        self.n_seqs = self.sequences.shape[0]
        self.model = model
        self.input_normalizer = normalizer

    def action(self, obs, deterministic=False):
        if self.t % self.T == 0:
            self.seq_idx = self.select_sqeuence(obs)
            self.current_sequence = self.sequences[self.seq_idx]
            print('IDX = {}'.format(self.seq_idx))
            print('SEQ = {}'.format(self.current_sequence))
            self.t = 0

        act = self.current_sequence[self.t]
        self.t += 1
        return act

    def select_sequence(self, obs):
        import pdb;pdb.set_trace()
        if not isinstance(obs, torch.Tensor):
            obs = dict_to_torch(obs, device=self.device)
        norms = self.normalizer.normalize({'observation':x, 'action':acts})
        preds = model(norms['observation'])[0]
        dists = torch.norm(preds - self.goal.unsqueeze(0))
        return torch.argmin(dists)

if __name__ == '__main__':
    from wheeledsim_rl.envs.pybullet_sim import WheeledSimEnv

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_T', type=int, required=False, default=100, help='Max number of steps per episode')
    parser.add_argument('--throttle_n', type=int, required=False, default=5, help='Number of throttle positions to consider')
    parser.add_argument('--steer_n', type=int, required=False, default=5, help='Number of steer positions to consider')
    args = parser.parse_args()

    env = WheeledSimEnv(T=args.env_T, terrainParamsIn={'cellPerlinScale':0., 'perlinScale':0.}, cliffordParams={'maxThrottle':20})

    seqs = generate_action_sequences_2(t=args.env_T, steer_n=args.steer_n, throttle_n=args.throttle_n)

    print(seqs[:, 0])
    torch.save(seqs, 'sequences9x9.pt')

    for k, seq in enumerate(seqs):
        print('Seq {}/{}'.format(k+1, len(seqs)))
        o = env.reset()
        buf = [o]
        for j, act in enumerate(seq):
            print('Step {}/{}'.format(j+1, len(seq)), end='\r')
            o, r, t, i = env.step(act)
            buf.append(o)

        xs = [o[0] for o in buf]
        ys = [o[1] for o in buf]

        plt.plot(xs, ys)

    plt.show()
