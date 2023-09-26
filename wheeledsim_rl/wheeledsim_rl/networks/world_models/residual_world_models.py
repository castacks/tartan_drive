import torch

from torch import nn, optim, distributions

from wheeledsim_rl.networks.world_models.world_models import WorldModel
from wheeledsim_rl.networks.world_models.cnn_vae import CNNEncoder, CNNDecoder
from wheeledsim_rl.util.util import quaternion_multiply

class ResidualWorldModel(WorldModel):
    """
    Same as the regular world model, but learn a residual on top of an analytical model.
    """
    def __init__(self, encoders, decoders, state_insize, action_insize, rnn_hidden_size, rnn_layers, mlp_encoder_hidden_size, mlp_decoder_hidden_size, input_normalizer, output_normalizer, model, dt, zero_init=True, device='cpu'):
        super(ResidualWorldModel, self).__init__(encoders, decoders, state_insize, action_insize, rnn_hidden_size, rnn_layers, mlp_encoder_hidden_size, mlp_decoder_hidden_size, input_normalizer, output_normalizer)
        self.model = model
        self.dt = dt

        if zero_init:
            self.state_decoder.weight.data.fill_(0.)
            self.state_decoder.bias.data.fill_(0.)

    def predict(self, obs, act, return_info=False):
        """
        Same as other predicts - add on the initial state.
        """
        #TODO: Get model predictions (rolled-out and batched)
        model_preds = self.get_multistep_model_predictions(obs['state'], act)

        if return_info:
            norm_obs = self.input_normalizer.normalize({'observation':obs})['observation']
            pred_obs = self.forward(norm_obs, act, return_info)
            denorm_obs = self.output_normalizer.denormalize({'observation':pred_obs['observation']})['observation']
#            denorm_obs['state'] = torch.cumsum(denorm_obs['state'], dim=-2) + obs['state'].unsqueeze(-2)
            denorm_obs['state'] = torch.cumsum(denorm_obs['state'], dim=-2) + model_preds
        
            return {
                    'observation': denorm_obs,
                    'encoder_dist':pred_obs['encoder_dist'],
                    'latent_observation':pred_obs['latent_observation']
                    }

        else:
            norm_obs = self.input_normalizer.normalize({'observation':obs})['observation']
            pred_obs = self.forward(norm_obs, act, return_info)
            denorm_obs = self.output_normalizer.denormalize({'observation':pred_obs})['observation']
#            denorm_obs['state'] = torch.cumsum(denorm_obs['state'], dim=-2) + obs['state'].unsqueeze(-2)
            denorm_obs['state'] = torch.cumsum(denorm_obs['state'], dim=-2) + model_preds
            return denorm_obs

    def get_training_targets(self, batch):
        """
        Extract features and states for reconstruction loss.
        """
        #TODO: Get model predictions (rolled-out and batched)
        model_preds = self.get_model_predictions(batch['observation']['state'], batch['action'])

        targets = {k:batch['next_observation'][k] for k in self.obs_keys + ['state']}
        targets['state'] -= model_preds

        return targets

    def get_multistep_model_predictions(self, obs, actions):
        """
        Get model predictions for a series of actions
        Args:
            obs: Expects dict of tensors with state batched as [batchdim x statedim]
            actions: Batched as [batchdim x timedim x actdim]
        """
        res = []
        cstate = obs
        for t in range(actions.shape[1]):
            preds = self.get_model_predictions(cstate, actions[:, t])
            res.append(preds)
            cstate = preds.clone()

        res = torch.stack(res, dim=1)
        return res

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

    def to(self, device):
        self.device = device
        self.encoders = nn.ModuleDict({k:net.to(device) for k,net in self.encoders.items()})
        self.decoders = nn.ModuleDict({k:net.to(device) for k,net in self.decoders.items()})
        self.mlp = self.mlp.to(device)
        self.state_decoder = self.state_decoder.to(device)
        self.rnn = self.rnn.to(device)
        self.input_normalizer = self.input_normalizer.to(device)
        self.output_normalizer = self.output_normalizer.to(device)
        self.model = self.model.to(device)
        return self
