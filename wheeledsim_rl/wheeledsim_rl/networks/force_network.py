"""
Provide the kinematics and use the NN to predict forces.
2 things to keep in mind that make this model different.
    1. Since we're going through a kinematics module, forward = predict, except for the input norm
    2. Also no need for an output normalizer. TODO: Maybe need to output normalize forces.
"""
import numpy as np
import torch

from torch import nn, optim, distributions

from wheeledsim_rl.util.util import quaternion_multiply

from wheeledsim_rl.networks.mlp import MLP
from wheeledsim_rl.util.ouNoise import ouNoise, sinOuNoise
from wheeledsim_rl.policies.ou_policy import OUPolicy
from wheeledsim_rl.replaybuffers.nstep_replaybuffer import NStepReplayBuffer
from wheeledsim_rl.vehicle_kinematics.dynamic_bicycle_model import DBMKinematics

from wheeledsim_rl.datasets.normalizer import ObservationNormalizer

class ForceModel(nn.Module):
    def __init__(self, env, kinematics, dt, hiddens=[32, 32], activation=nn.Tanh, input_normalizer=None, device='cpu'):
        super(ResidualDynamicsModel, self).__init__()

        #For now, no dict observations.
        sample_obs = env.reset()
        sample_act = torch.tensor(env.action_space.low).float()
        self.obs_dim = sample_obs.shape[-1]
        self.act_dim = sample_act.shape[-1]

        self.kinematics = kinematics
        self.force_dim = self.kinematics.control_dim

        self.input_normalizer = input_normalizer if input_normalizer else ObservationNormalizer(env=env)

        self.mlp = MLP(self.obs_dim + self.act_dim, self.obs_dim, hiddens, activation, 0., device)
        self.dt = dt

    def forward(self, obs, action):
        inp = torch.cat([obs, action], dim=-1)
        return self.mlp.forward(inp)

    def predict(self, obs, action):
        """
        Appends an ensembledim to the front of the prediction.
        """
        import pdb;pdb.set_trace()
        norm_obs = self.input_normalizer.normalize({'observation':obs})['observation']
        forces = self.forward(norm_obs, action)
        preds = self.get_model_predictions(obs, action, forces)
        return preds.unsqueeze(0)

    def get_model_predictions(self, obs, action, forces):
        """
        Gets the predictions of the analytical model for this batch of state, action
        1. Convert Pybullet state into model state.
        2. Predict using the model.
        3. Convert model predictions into pybullet state.

        State is:
            For KBM: [x, y, theta, v]
            FOR DBM: [x, y, theta, xdot, ydot, thetadot, steer]
        """
        import pdb;pdb.set_trace()
        x, y, z, qx, qy, qz, qw, vxb, vyb, vzb, rolldot, pitchdot, yawdot = obs.moveaxis(-1, 0)
        throttle, steer = action.moveaxis(-1, 0)

        psi = torch.atan2(2 * (qw*qz + qx*qy), 1 - 2*(qz*qz + qy*qy))

        model_state = torch.stack([x, y, psi, vxb, vyb, yawdot, steer], dim=-1)

        #Model prediction
        model_prediction = self.kinematics.forward_dynamics(model_state, forces, self.dt)

        #Model state back to pybullet state. Convert global velocities into body velocities and compose yaw into the quaternion.
        p_x, p_y, p_psi, p_vxbody, p_vybody, p_yawdot, p_steer = model_prediction.moveaxis(-1, 0)

        dpsi = p_psi - psi

        q = torch.stack([qw, qx, qy, qz], dim=-1)
        dq = torch.stack([torch.cos(dpsi), torch.zeros_like(dpsi), torch.zeros_like(dpsi), torch.sin(dpsi)], dim=-1)
        qnew = quaternion_multiply(dq, q) #Apply the dq first, as the model doesn't account for the 3D orientation. (This is equivalent to rotating the previous orientation around global z).
        
        # Re-order the quaternion to scalar-last.
        return torch.stack([p_x, p_y, z, qnew[:, 1], qnew[:, 2], qnew[:, 3], qnew[:, 0], p_vxbody, p_vybody, vzb, rolldot, pitchdot, p_yawdot], dim=-1)

    def get_training_targets(self, batch):
        """
        Get training targets from batch.
        In this case, training targets are next state - model(state, action)
        """
        return batch['next_observation']

    def to(self, device):
        self.device = device
        self.mlp.to(device)
        self.input_normalizer.to(device)
        self.model.to(device)
        return self

class EnsembleForceModel(ForceModel):
    def __init__(self, env, model, dt, hiddens=[32, 32], activation=nn.Tanh, n_models=5, input_normalizer=None, device='cpu'):
        super(EnsembleForceModel, self).__init__(env, model, dt, hiddens, activation, input_normalizer, device)

        self.n_models = n_models
        self.mlps = nn.ModuleList([MLP(self.obs_dim + self.act_dim, self.obs_dim, hiddens, activation, 0., device) for _ in range(self.n_models)])

    def forward(self, obs, action):
        inp = torch.cat([obs, action], dim=-1)
        return torch.stack([mlp.forward(inp) for mlp in self.mlps], dim=0)

    def predict(self, obs, action, return_all=True):
        import pdb;pdb.set_trace()
        norm_obs = self.input_normalizer.normalize({'observation':obs})['observation']
        forces = self.forward(norm_obs, action)
        preds = self.get_model_predictions(obs, action, forces)
        return preds.unsqueeze(0)

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

