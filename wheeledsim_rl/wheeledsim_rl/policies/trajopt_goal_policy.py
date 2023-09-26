import torch
from torch import nn, distributions, optim

from wheeledsim_rl.util.util import dict_to, dict_map

class TrajOptGoalPolicy:
    """
    A class that designs actions to maximize the epistemic uncertainty of a model.
    Method:
        1. Start with k action seqs of length T
        2. Compute ensemble disagreement by rolling out through the model
        3. Step the actions in the direction that minimizes distance to goal.
    """
    def __init__(self, env, model, n_particles=4, T=100, itrs=100, k_smooth=1.0, opt_every=-1, opt_class=optim.Adam, opt_kwargs={}):
        self.model = model
        self.act_dim = model.act_dim
        self.n_particles = n_particles
        self.T = T
        self.itrs = itrs
        self.k_smooth = k_smooth
        self.goal = torch.zeros(2).to(self.model.device)
        self.opt_every = T if opt_every ==-1 else opt_every
        self.t = opt_every

        self.act_lb = torch.tensor(env.action_space.low)
        self.act_ub = torch.tensor(env.action_space.high)

        self.opt_class = opt_class
        self.opt_kwargs = opt_kwargs
        self.device = 'cpu'

    def action(self, obs, deterministic=True):
        """
        If we reached the end of a traj, re-optimize. Else just execute.
        """
        if self.t == self.opt_every:
            self.seq = self.optimize(dict_map(obs, lambda x: x.unsqueeze(0)))
            self.t = 0

        act = self.seq[self.t]
        self.t += 1
        return act

    def optimize(self, state):
        """
        Generate a T-step sequence of actions that (locally) minimizes distance to goal.
        """
        seqs = self.generate_seqs()
        seqs.requires_grad = True
        opt = self.opt_class([seqs], **self.opt_kwargs)

        best_cost = torch.ones(self.n_particles, device=self.device) * 1e8
        best_seqs = torch.zeros_like(seqs)

        for i in range(self.itrs):
            seqs_clamp = torch.minimum(torch.maximum(seqs, self.act_lb.unsqueeze(0).unsqueeze(0)), self.act_ub.unsqueeze(0).unsqueeze(0))
            preds = self.rollout(state, seqs_clamp) #[particledim x time x edim x state]
            dist_to_goal = torch.linalg.norm(preds['state'].mean[:, :, :2] - self.goal, dim=-1).mean(dim=-1)

            smoothness = self.k_smooth * (seqs_clamp[:, 1:] - seqs_clamp[:, :-1]).pow(2).mean(dim=-1).mean(dim=-1)
            cost = smoothness + dist_to_goal

            loss = cost.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
#            torch.cuda.empty_cache()

            print("ITR {}".format(i+1))
            print("dtg    =", dist_to_goal.detach())
            print("smooth =", smoothness.detach() / self.k_smooth)
            print("cost   =", cost.detach())
            print("update =", cost < best_cost)
            print('_'*30)
#            print('Itr {}'.format(i+1), end='\r')

            best_seqs[best_cost > cost] = seqs_clamp[best_cost > cost].detach()
            best_cost[best_cost > cost] = cost[best_cost > cost].detach()

        seqs = best_seqs[best_cost.argmin()]
        return seqs

    def rollout(self, state, seqs):
        """
        Roll out seqs from the start state
        """
        #Hack it for now
        #Repeat the initial state for each seq
        state = dict_map(state, lambda x: torch.cat([x] * self.n_particles, dim=0))
        preds = self.model.predict(state, seqs)

        return preds

    def generate_seqs(self):
        dist = torch.distributions.Uniform(low=self.act_lb, high=self.act_ub)
        seqs = dist.sample(sample_shape=(self.n_particles, self.T))
        return seqs

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.act_lb = self.act_lb.to(device)
        self.act_ub = self.act_ub.to(device)
        return self

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import copy
    import pybullet as p
    from wheeledSim.envs.pybullet_sim import WheeledSimEnv

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, required=True, help='The config file of the env.')
    parser.add_argument('--model_fp', type=str, required=True, help='The model to optimize over.')

    args = parser.parse_args()

    model = torch.load(args.model_fp)
    env = WheeledSimEnv(args.config_fp, T=500, render=True)

    policy = TrajOptGoalPolicy(env, model, k_smooth=1.0, T=100, opt_kwargs={'lr':1e-1}, itrs=20, opt_every=10).to('cpu')
    policy.goal[0] = 10.0
    policy.goal[1] = -5.0

    obs = env.reset()
    p.addUserDebugLine([policy.goal[0], policy.goal[1], 0.0], [policy.goal[0], policy.goal[1], 0.0])
    init_obs = copy.deepcopy(obs)
#    s = dict_map(dict_to(s, 'cpu'), lambda x:x.unsqueeze(0))
#    acts = policy.optimize(s).cpu()

#    with torch.no_grad():
#        pred_traj = model.predict(s, acts.unsqueeze(0))['state'].mean[0]

    gt_traj = []
    acts = []
    term = False

    while not term and torch.linalg.norm(obs['state'][:2] - policy.goal) > 0.5:
        act = policy.action(obs)
        nobs, r, term, i = env.step(act)

        gt_traj.append(nobs['state'])
        acts.append(act.clone())
        obs = nobs
        print('act =', act)

    gt_traj = torch.stack(gt_traj, dim=0)
    acts = torch.stack(acts, dim=0)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#    axs[0].plot(pred_traj[:, 0], pred_traj[:, 1], marker='.', label='pred')
    axs[0].plot(gt_traj[:, 0], gt_traj[:, 1], marker='.', label='gt')
    axs[0].scatter(init_obs['state'][0], init_obs['state'][1], marker='^', c='y', label='start')
    axs[0].scatter(policy.goal[0], policy.goal[1], marker='x', c='g', label='goal')
    axs[0].legend()
    axs[1].plot(acts[:, 0], label='throttle')
    axs[1].plot(acts[:, 1], label='steer')
    axs[1].legend()
    plt.show()
