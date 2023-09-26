import torch
from torch import nn, distributions, optim
import matplotlib.pyplot as plt
import time

from wheeledsim_rl.util.util import dict_to, dict_map

class TreeSearchGoalPolicy:
    """
    Do tree search through the model to reach goal.
    Sketch:
        1. Get state and action primitives
        2. roll them out through the model
        3. Select the best one and execute
    """
    def __init__(self, env, model, n_throttles=5, n_steers=7, T=10, itrs=1, opt_every=10, visualize=True):
        self.model = model
        self.act_dim = model.act_dim
        self.T = T
        self.itrs = itrs
        self.goal = torch.zeros(2).to(self.model.device)
        self.opt_every = T if opt_every ==-1 else opt_every
        self.t = opt_every
        self.n_throttles = n_throttles
        self.n_steers = n_steers

        self.act_lb = torch.tensor(env.action_space.low)
        self.act_ub = torch.tensor(env.action_space.high)

        self.device = 'cpu'
        self.primitives = self.generate_seqs()

        self.visualize = visualize

        if self.visualize:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 4))
            plt.show(block=False)

    def action(self, obs, deterministic=True):
        """
        If we reached the end of a traj, re-optimize. Else just execute.
        """
        if self.t == self.opt_every:
            t = time.time()
#            self.seq = self.optimize(dict_map(obs, lambda x: x.unsqueeze(0)))
            self.seq = self.tree_optimize(dict_map(obs, lambda x: x.unsqueeze(0)))
            print('Took {:.6f}s to plan'.format(time.time() - t))
            self.t = 0

        act = self.seq[self.t]
        self.t += 1
        return act

    def optimize(self, state):
        """
        Generate a T-step sequence of actions that (locally) minimizes distance to goal.
        """
        import pdb;pdb.set_trace()
        cstate = copy.deepcopy(state)
        hidden_cstate = self.model.encode_observation(cstate, normalize_obs=True, rsample=False)

        #Don't forget that you have a distribution you can use
        pred_states, pred_hiddens = self.rollout(cstate, hidden_cstate, self.primitives)
        pred_states = pred_states.mean
        goal_dist = torch.linalg.norm(pred_states[:, -1, :2] - self.goal, dim=-1)
        costs = goal_dist

        best_idx = costs.argmin()
        best_seq = self.primitives[best_idx]
        best_traj = pred_states[best_idx]

        if self.visualize:
            self.ax.cla()

            for traj, cost in zip(pred_states, costs):
                self.ax.plot(traj[:, 0], traj[:, 1], c='k', alpha=0.5, marker='.')
                self.ax.text(traj[-1, 0], traj[-1, 1], '{:.2f}'.format(cost))

            self.ax.plot(best_traj[:, 0], best_traj[:, 1], c='r', marker='.')
            plt.pause(1e-1)

        return self.primitives[goal_dist.argmin()]

    def tree_optimize(self, state):
        """
        Perform tree search in order to find the best path to goal.
        There's not really a notion of traversal cost right now.
        Nodes need CTG, acts, states and prev
        """
        cstate = copy.deepcopy(state)
        hidden_cstate = self.model.encode_observation(cstate, normalize_obs=True, rsample=False)
        current_cost = torch.linalg.norm(state['state'][:, :2] - self.goal)

        openlist = [{'ctg':current_cost, 'prev':None, 'actions':None, 'states':cstate['state'], 'hidden_state':hidden_cstate}]
        closedlist = []

        for itr in range(self.itrs):
            current = openlist.pop(0)

            #Expand
            pred_states, pred_hiddens = self.rollout({'state':current['states'][[-1]]}, current['hidden_state'], self.primitives)
            pred_states = pred_states.mean
            goal_dist = torch.linalg.norm(pred_states[:, -1, :2] - self.goal, dim=-1)
            costs = goal_dist

            #Insert
            for traj, acts, cost, hidden in zip(pred_states, self.primitives, costs, pred_hiddens):
                node = {
                        'ctg':cost,
                        'states':traj,
                        'actions': acts,
                        'hidden_state':hidden,
                        'prev': current
                        }
                openlist.append(node)

            openlist = sorted(openlist, key=lambda x:x['ctg'])
            closedlist.append(current)

        closedlist = sorted(closedlist, key=lambda x:x['ctg'])

        best_traj, best_acts = self.get_path(closedlist[0])
        #Visualize and return
        if self.visualize:
            self.ax.cla()
            for node in closedlist:
                traj, actions = self.get_path(node)
                cost = node['ctg']
                self.ax.plot(traj[:, 0], traj[:, 1], c='k', alpha=0.5, marker='.')
                self.ax.text(traj[-1, 0], traj[-1, 1], '{:.2f}'.format(cost))

            self.ax.plot(best_traj[:, 0], best_traj[:, 1], c='r', marker='.')
            plt.pause(1e-1)

        return best_acts

    def get_path(self, node):
        """
        List util to extract paths.
        """
        current = node
        states = []
        actions = []

        # Special-case the root node
        if current['prev'] is None:
            return torch.cat([current['states']] * self.T, dim=0), torch.zeros(self.T, self.act_dim, device=self.device)

        #Since the root state doesn't have states/actions
        while current['prev'] is not None:
            states.insert(0, current['states'])
            actions.insert(0, current['actions'])
            current = current['prev']

        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        return states, actions


    def rollout(self, state, hidden_state, seqs):
        """
        Roll out seqs from the start state
        """
        #Hack it for now
        #Repeat the initial state for each seq
        #Have to give state
        state = dict_map(state, lambda x: torch.cat([x] * self.primitives.shape[0], dim=0))
        hstate = torch.cat([hidden_state] * self.primitives.shape[0], dim=0)

        with torch.no_grad():
            preds = self.model.hidden_predict(state, hstate, seqs, keys=['state'], return_info=True)

        state_preds = preds['observation']['state']
        hidden_preds = preds['hidden_observation'].swapaxes(0, 1)

        return state_preds, hidden_preds

    def generate_seqs(self):
        #Build the primitives
        throttles = torch.cat([torch.linspace(self.act_lb[0], 0, self.n_throttles//2 + 1)[:-1], torch.linspace(0, self.act_ub[0], self.n_throttles//2 + 1)[1:]])
        steers = torch.linspace(self.act_lb[1], self.act_ub[1], self.n_steers)

        primitives = [torch.zeros(self.T, self.act_dim, device=self.device)]

        for t in throttles:
            for s in steers:
                seq = torch.stack([torch.tensor([t, s], device=self.device)] * self.T, dim=0)
                primitives.append(seq)

        primitives = torch.stack(primitives, dim=0)

        return primitives

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

    policy = TreeSearchGoalPolicy(env, model, T=10, itrs=50, n_throttles=7, n_steers=5, visualize=True, opt_every=10).to('cpu')
    policy.goal[0] = 10.0
    policy.goal[1] = -5.0

    obs = env.reset()
    p.addUserDebugLine([policy.goal[0], policy.goal[1], 0.0], [policy.goal[0], policy.goal[1], 0.0])
    init_obs = copy.deepcopy(obs)

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
