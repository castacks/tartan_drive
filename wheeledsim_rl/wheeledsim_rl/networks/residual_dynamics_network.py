"""
Implements residual dynamics. I.e. given an interaction (state, action, next state),
and a model f(state, action), predict next_state - f(state, action)
"""
import numpy as np
import torch

from torch import nn, optim, distributions

from wheeledsim_rl.util.util import quaternion_multiply

from wheeledsim_rl.networks.mlp import MLP
from wheeledsim_rl.util.ouNoise import ouNoise, sinOuNoise
from wheeledsim_rl.policies.ou_policy import OUPolicy
from wheeledsim_rl.replaybuffers.nstep_replaybuffer import NStepReplayBuffer
from wheeledsim_rl.datasets.normalizer import ObservationNormalizer

class ResidualDynamicsModel(nn.Module):
    def __init__(self, env, model, dt, hiddens=[32, 32], activation=nn.Tanh, input_normalizer=None, output_normalizer=None, zero_init=True, device='cpu'):
        super(ResidualDynamicsModel, self).__init__()

        sample_obs = env.reset()
        sample_act = torch.tensor(env.action_space.low).float()
        self.obs_dim = sample_obs.shape[-1]
        self.act_dim = sample_act.shape[-1]

        self.input_normalizer = input_normalizer if input_normalizer else ObservationNormalizer(env=env)
        self.output_normalizer = output_normalizer if output_normalizer else ObservationNormalizer(env=env)

        self.mlp = MLP(self.obs_dim + self.act_dim, self.obs_dim, hiddens, activation, 0., device)
        self.model = model
        self.dt = dt

        #From Silver et al. 2018, initialize last layer to 0 to match nominal model on initialization.
        if zero_init:
            self.mlp.layers[-1].weight.data.fill_(0.)
            self.mlp.layers[-1].bias.data.fill_(0.)

    def forward(self, obs, action):
        inp = torch.cat([obs, action], dim=-1)
        return self.mlp.forward(inp)

    def predict(self, obs, action):
        """
        Appends an ensembledim to the front of the prediction.
        """
        model_predictions = self.get_model_predictions(obs, action)
        norm_obs = self.input_normalizer.normalize({'observation':obs})['observation']
        residuals = self.forward(norm_obs, action)
        denorm_residuals = self.output_normalizer.denormalize({'observation':residuals})['observation']
        return model_predictions.unsqueeze(0) + denorm_residuals

    def get_model_predictions(self, obs, action):
        """
        Gets the predictions of the analytical model for this batch of state, action
        1. Convert Pybullet state into model state.
        2. Predict using the model.
        3. Convert model predictions into pybullet state.
        """
        #Pybullet state to model state. Convert body velocities into global velocities and extract yaw from quaternion
        """
        x = obs[:, 0]
        y = obs[:, 1]
        z = obs[:, 2]
        qx = obs[:, 3]
        qy = obs[:, 4]
        qz = obs[:, 5]
        qw = obs[:, 6]
        vxb = obs[:, 7]
        vyb = obs[:, 8]
        """

        x, y, z, qx, qy, qz, qw, vxb, vyb, vzb, rolldot, pitchdot, yawdot = obs.moveaxis(-1, 0)

        psi = torch.atan2(2 * (qw*qz + qx*qy), 1 - 2*(qz*qz + qy*qy))
#        v = torch.hypot(vxb, vyb)
        v = vxb

        model_state = torch.stack([x, y, psi, v], dim=-1)

        #Model prediction
        model_prediction = self.model.forward(model_state, action, self.dt)

        #Model state back to pybullet state. Convert global velocities into body velocities and compose yaw into the quaternion.
        p_x, p_y, p_psi, p_vxbody = model_prediction.T

        #TODO: Double-check that you're transforming these correctly.
        p_xbdot = p_vxbody

        dpsi = p_psi - psi

        q = torch.stack([qw, qx, qy, qz], dim=-1)
        dq = torch.stack([torch.cos(dpsi), torch.zeros_like(dpsi), torch.zeros_like(dpsi), torch.sin(dpsi)], dim=-1)
        qnew = quaternion_multiply(dq, q) #Apply the dq first, as the model doesn't account for the 3D orientation. (This is equivalent to rotating the previous orientation around global z).
        
        # Re-order the quaternion to scalar-last.
        return torch.stack([p_x, p_y, z, qnew[:, 1], qnew[:, 2], qnew[:, 3], qnew[:, 0], p_xbdot, vyb, vzb, rolldot, pitchdot, yawdot], dim=-1)

    def get_training_targets(self, batch):
        """
        Get training targets from batch.
        In this case, training targets are next state - model(state, action)
        """
        obs = batch['observation']
        act = batch['action']
        nobs = batch['next_observation']

        model_preds = self.get_model_predictions(obs, act)
        targets = nobs - model_preds

        return targets

    def to(self, device):
        self.device = device
        self.mlp.to(device)
        self.input_normalizer.to(device)
        self.output_normalizer.to(device)
        self.model.to(device)
        return self

class EnsembleResidualDynamicsModel(ResidualDynamicsModel):
    def __init__(self, env, model, dt, hiddens=[32, 32], activation=nn.Tanh, n_models=5, input_normalizer=None, output_normalizer=None, zero_init=True, device='cpu'):
        super(EnsembleResidualDynamicsModel, self).__init__(env, model, dt, hiddens, activation, input_normalizer, output_normalizer, zero_init, device)

        self.n_models = n_models
        self.mlps = nn.ModuleList([MLP(self.obs_dim + self.act_dim, self.obs_dim, hiddens, activation, 0., device) for _ in range(self.n_models)])

    def forward(self, obs, action):
        inp = torch.cat([obs, action], dim=-1)
        return torch.stack([mlp.forward(inp) for mlp in self.mlps], dim=0)

    def predict(self, obs, action, return_all=True):
        model_predictions = self.get_model_predictions(obs, action)
        norm_obs = self.input_normalizer.normalize({'observation':obs})['observation']
        residuals = self.forward(norm_obs, action)
        denorm_residuals = self.output_normalizer.denormalize({'observation':residuals})['observation']

        if return_all:
            return model_predictions + denorm_residuals
        else:
            return model_predictions + denorm_residuals.mean(dim=0)

    def to(self, device):
        self.device = device
        self.mlps = nn.ModuleList([mlp.to(device) for mlp in self.mlps])
        self.input_normalizer.to(device)
        self.output_normalizer.to(device)
        self.model.to(device)
        return self

