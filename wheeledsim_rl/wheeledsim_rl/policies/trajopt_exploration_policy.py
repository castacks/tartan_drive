import torch
from torch import nn, distributions, optim

from wheeledsim_rl.util.util import dict_to, dict_map

class TrajOptExplorationPolicy:
    """
    A class that designs actions to maximize the epistemic uncertainty of a model.
    Method:
        1. Start with k action seqs of length T
        2. Compute ensemble disagreement by rolling out through the model
        3. Step the actions in the direction that increases uncertainty.
    """
    def __init__(self, env, model, n_particles=4, T=100, itrs=100, k_smooth=1.0, opt_class=optim.Adam, opt_kwargs={}):
        self.model = model
        self.act_dim = model.act_dim
        self.obs_dim = model.obs_dim
        self.n_particles = n_particles
        self.T = T
        self.t = T
        self.itrs = itrs
        self.k_smooth = k_smooth

        self.act_lb = torch.tensor(env.action_space.low)
        self.act_ub = torch.tensor(env.action_space.high)

        self.opt_class = opt_class
        self.opt_kwargs = opt_kwargs
        self.device = 'cpu'

    def action(self, obs, deterministic=True):
        """
        If we reached the end of a traj, re-optimize. Else just execute.
        """
        if self.t == self.T:
            self.seq = self.optimize(dict_map(obs, lambda x: x.unsqueeze(0)))
            self.t = 0

        act = self.seq[self.t]
        self.t += 1
        return act

    def optimize(self, state):
        """
        Generate a T-step sequence of actions that (locally) maximizes epistemic uncertainty
        """
        seqs = self.generate_seqs()
        seqs.requires_grad = True
        opt = self.opt_class([seqs], **self.opt_kwargs)

        best_cost = torch.ones(self.n_particles, device=self.device) * 1e8
        best_seqs = torch.zeros_like(seqs)

        for i in range(self.itrs):
            seqs_clamp = torch.minimum(torch.maximum(seqs, self.act_lb.unsqueeze(0).unsqueeze(0)), self.act_ub.unsqueeze(0).unsqueeze(0))
            preds = self.rollout(state, seqs_clamp) #[particledim x time x edim x state]
            unc = preds.std(dim=-2).sum(dim=-1).mean(dim=1)

            smoothness = self.k_smooth * (seqs_clamp[:, 1:] - seqs_clamp[:, :-1]).pow(2).mean(dim=-1).mean(dim=-1)
            cost = smoothness-unc

            loss = cost.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.empty_cache()

            print("ITR {}".format(i+1))
            print("unc    =", unc.detach())
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
        Roll out seqs from the start state (always return ensemble predictions)
        """
        #Hack it for now
        state = state['state']

        cstate = state.repeat(self.n_particles, self.model.n_models, 1)
        preds = []
        for i in range(seqs.shape[1]):
            act = seqs[:, [i]].repeat(1, self.model.n_models, 1)
            cstate = self.model.predict(cstate.view(-1, self.obs_dim), act.view(-1, self.act_dim))
            cstate = cstate.view(self.model.n_models, self.n_particles, self.model.n_models, self.obs_dim).permute(1, 0, 2, 3) #Re-batch to [pdim x edim x edim x statedim]
            cstate = cstate[:, torch.arange(self.model.n_models), torch.arange(self.model.n_models)]

            preds.append(cstate)

        preds = torch.stack(preds, dim=1) #batched as [seqdim x time x edim x statedim]
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
    from wheeledSim.envs.pybullet_sim import WheeledSimEnv

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_fp', type=str, required=True, help='The model to optimize over.')

    args = parser.parse_args()

    terrain_params_in = {
        'cellPerlinScale':0.,
        'perlinScale':0.
    }
    clifford_params_in = {
        'massScale':1.0,
        'maxThrottle':20.0, 
    }

    model = torch.load(args.model_fp)
    env = WheeledSimEnv('../../scripts/data_collection/heightmap_fricmap_flat.yaml', T=100, render=False)

    policy = TrajOptExplorationPolicy(env, model, k_smooth=50.0, T=100, opt_kwargs={'lr':1e-1}).to('cuda')

    s = env.reset()
    s = dict_map(dict_to(s, 'cuda'), lambda x:x.unsqueeze(0))
    acts = policy.optimize(s).cpu()
    print(acts)
    plt.plot(acts[:, 0], label='throttle')
    plt.plot(acts[:, 1], label='steer')
    plt.legend()
    plt.show()
