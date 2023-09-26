import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

from wheeledsim_rl.util.util import dict_repeat, dict_stack, dict_to_torch

class RandomActionSequencePolicy:
    """
    Policy that acts by selecting an action sequence to run for k timesteps.
    To keep consistency with other non-hierarchical methods, provide the switching rate.
    Store current action sequence and step once every time action is called.

    action_sequences expected to be a tensor of size: [choicedim x timedim x actdim]
    """
    def __init__(self, env, action_sequences, device='cpu'):
        self.goal = torch.zeros(2)
        self.act_dim = env.action_space.shape[-1]
        assert self.act_dim == action_sequences.shape[-1], "Expected action sequences of dim {}, got {}".format(self.act_dim, action_sequences.shape[-1])
        self.sequences = action_sequences
        self.T = action_sequences.shape[1]
        self.t = 0
        self.current_sequence = None
        self.n_seqs = self.sequences.shape[0]
        self.device = device

    def to(self, device):
        self.device = device

    def action(self, obs, deterministic=False):
        if self.t % self.T == 0:
            self.seq_idx = torch.randint(self.n_seqs, size=(1,)).item()
            self.current_sequence = self.sequences[self.seq_idx]
            print('IDX = {}'.format(self.seq_idx))
            print('SEQ = {}'.format(self.current_sequence))
            self.t = 0

        act = self.current_sequence[self.t]
        self.t += 1
        return act.to(self.device)

class ActionSequenceExplorationPolicy:
    """
    Given a low-level model ensemble, (i.e. f(s, a), where a sequence is a series of actions), choose the sequence with the highest uncertainty.
    """
    def __init__(self, env, action_sequences, model, device='cpu'):
        self.act_dim = env.action_space.shape[-1]
        assert self.act_dim == action_sequences.shape[-1], "Expected action sequences of dim {}, got {}".format(self.act_dim, action_sequences.shape[-1])
        self.sequences = action_sequences
        self.T = action_sequences.shape[1]
        self.t = 0
        self.current_sequence = None
        self.n_seqs = self.sequences.shape[0]
        self.model = model
        self.device = device

    def to(self, device):
        self.device = device
        self.sequences = self.sequences.to(device)
        self.model = self.model.to(device)
        return self

    def action(self, obs, deterministic=False):
        if self.t % self.T == 0:
            self.seq_idx = self.select_sequence(obs)
            self.current_sequence = self.sequences[self.seq_idx]
            print('IDX = {}'.format(self.seq_idx))
            print('SEQ = {}'.format(self.current_sequence))
            self.t = 0

        act = self.current_sequence[self.t]
        self.t += 1
        return act.to(self.device)

    def select_sequence(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = dict_to_torch(obs, device=self.device)
            obs = {k:v.unsqueeze(0) for k,v in obs.items()}

        with torch.no_grad():
            preds = self.rollout(obs, self.sequences) #[seq x time x ensemble x state]
        unc = preds[:, -1].std(dim=-2).sum(dim=-1)

        return torch.argmax(unc)

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

class ActionSequenceGoalPolicy:
    """
    Given a model of where each action sqeuence will take you, given an obs, choose the sequence that minimizes your goal distance.
    """
    def __init__(self, env, action_sequences, predictor, device='cpu'):
        self.goal = torch.zeros(2)
        self.act_dim = env.action_space.shape[-1]
        assert self.act_dim == action_sequences.shape[-1], "Expected action sequences of dim {}, got {}".format(self.act_dim, action_sequences.shape[-1])
        self.sequences = action_sequences
        self.T = action_sequences.shape[1]
        self.t = 0
        self.current_sequence = None
        self.n_seqs = self.sequences.shape[0]
        self.predictor = predictor
        self.device = device

    def to(self, device):
        self.device = device

    def action(self, obs, deterministic=False):
        if self.t % self.T == 0:
            self.seq_idx = self.select_sequence(obs)
            self.current_sequence = self.sequences[self.seq_idx]
            print('IDX = {}'.format(self.seq_idx))
            print('SEQ = {}'.format(self.current_sequence))
            self.t = 0

        act = self.current_sequence[self.t]
        self.t += 1
        return act.to(self.device)

    def select_sequence(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = dict_to_torch(obs, device=self.device)
            obs = {k:v.unsqueeze(0) for k,v in obs.items()}

        with torch.no_grad():
            preds = self.predictor.model_rollout(obs)[:, :2]
            success = self.predictor.forward(obs)[0].argmax(dim=-1) #index 0 = true
        dists = torch.linalg.norm(preds - self.goal.unsqueeze(0), dim=-1)
        dists += success.float() * 1e8

        self.plot_state(obs)

        return torch.argmin(dists)

    def plot_state(self, obs):
        with torch.no_grad():
            preds = self.predictor.model_rollout(obs)[:, :2]
            success = self.predictor.forward(obs)[0].argmax(dim=-1) #index 0 = true

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].scatter(obs['state'][0, 0], obs['state'][0, 1], c='k', marker='x')
        axs[0].scatter(self.goal[0], self.goal[1], marker='x', c='g')

        for i, (p, s) in enumerate(zip(preds, success)):
            axs[0].scatter(p[0], p[1], marker='x', c='r' if s else 'b')
            axs[0].text(p[0], p[1], i)

        axs[1].imshow(obs['image'][0])
        plt.show()

if __name__ == '__main__':
    import argparse

    from wheeledsim_rl.envs.pybullet_sim import WheeledSimEnv
    from wheeledsim_rl.collectors.meta_controller_collector import MetaControllerCollector
    from wheeledsim_rl.replaybuffers.simple_replaybuffer import SimpleReplayBuffer
    from wheeledsim_rl.replaybuffers.dict_replaybuffer import DictReplayBuffer
    from wheeledsim_rl.networks.controller_predictor import ImageControllerPredictor
    from wheeledsim_rl.models.kinematic_bicycle_model import KinematicBicycleModel

    from wheeledSim.FlatRockyTerrain import FlatRockyTerrain

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_T', type=int, required=False, default=100, help='Max number of steps per episode')
    parser.add_argument('--seq_fp', type=str, required=True, help='path to action sequences')
    parser.add_argument('--model_params', type=str, required=False, help='Path to sysid\'ed model params.')
    args = parser.parse_args()

    terrain = FlatRockyTerrain()
    env = WheeledSimEnv(T=args.env_T, existingTerrain=terrain, terrainParamsIn={'N':100}, senseParamsIn={"senseResolution":[32, 32]}, cliffordParams={'maxThrottle':20}, use_images=True)

    seqs = torch.load(args.seq_fp)
    policy = RandomActionSequencePolicy(env, seqs)
    collector = MetaControllerCollector(env, policy)

    model_params = torch.load(args.model_params) if args.model_params else {}
    model = KinematicBicycleModel(hyperparams=model_params)
    predictor = ImageControllerPredictor(env, model, seqs, state_keys=['state'], image_keys=['image'])

    buf = DictReplayBuffer(env, capacity = 100000)
    buf.buffer['action'] = torch.tensor([-1], device=buf.device).repeat(buf.capacity, 1)

#    import pdb;pdb.set_trace()
#    buf = torch.load('img_buffer.pt')

    trajs = collector.collect_trajs(1000)

    labels = predictor.get_training_targets(trajs)
    print(labels)

    buf.insert(trajs)
    print(buf)
    torch.save(buf, 'img_buffer_eval.pt')
