import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn

from wheeledsim_rl.networks.mlp import MLP
from wheeledsim_rl.networks.cnn_blocks.cnn_blocks import ResnetCNN
from wheeledsim_rl.models.kinematic_bicycle_model import KinematicBicycleModel

def observation_to_model_state(obs):
    """
    Need to transform the observations to model state.
    1. Vels to global frame
    2. Quaternion to yaw
    """
    if len(obs.shape) == 1:
        return observation_to_model_state(obs.unsqueeze(0)).squeeze()

    x = obs[:, 0]
    y = obs[:, 1]
    qx = obs[:, 3]
    qy = obs[:, 4]
    qz = obs[:, 5]
    qw = obs[:, 6]
    psi = torch.atan2(2 * (qw*qz + qx*qy), 1 - 2*(qz*qz + qy*qy))

    vxb = obs[:, 7]
    vyb = obs[:, 8]
    v = torch.hypot(vxb, vyb)

    return torch.stack([x, y, psi, v], dim=-1)

class ControllerPredictor(nn.Module):
    """
    Network that takes as input some state and a set of n-step maneuvers and predicts whether they will be successful from a given state.
    Success is defined here as:
        1. Forward-simulate current state through an analytical model
        2. If real traj is within a threshold of the analytical model, call it a success.
    """
    def __init__(self, env, model, seqs, threshold=1.0, dt=0.1, mlp_hiddens=[32, 32], mlp_activation=nn.Tanh, device='cpu'):
        super(ControllerPredictor, self).__init__()

        self.env = env
        sample_obs = torch.tensor(env.observation_space.low)
        self.insize = np.prod(sample_obs.shape)
        self.outsize = seqs.shape[0] * 2 #Predict T/F for each action.

        self.model = model
        self.dt = dt
        self.sequences = seqs
        self.threshold = threshold

        self.mlp = MLP(self.insize, self.outsize, mlp_hiddens, mlp_activation)
        self.device = device

    def forward(self, obs):
        """
        Should predict all seqs from obs only.
        """
        logits = self.mlp(obs)
        keep_shape = obs.shape[:-1]
        logits = logits.view(*keep_shape, self.sequences.shape[0], 2)
        return logits

    def get_training_targets(self, batch):
        """
        Get the labels for a batch by forward-simulating the model and checking if the actual result was inside the threshold.
        """
        cseqs = batch['action'].squeeze()
        x = observation_to_model_state(batch['observation'])
        for t in range(self.sequences.shape[1]):
            u = self.sequences[cseqs, t]
            x = self.model.forward(x, u, dt=self.dt)

        xpos = x[:, :2]
        gt_xpos = batch['next_observation'][:, :2]
        dist = torch.linalg.norm(xpos - gt_xpos, dim=-1)
        labels = (dist > self.threshold).long() #class 0 = True, class 1 = False
        return labels

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.sequences = self.sequences.to(device)
        self.mlp = self.mlp.to(device)
        return self

class ImageControllerPredictor(nn.Module):
    def __init__(self, env, model, seqs, state_keys, image_keys, cnn_outsize=16, threshold=1.0, dt=0.1, mlp_hiddens=[32, 32], mlp_activation=nn.Tanh, device='cpu'):
        """
        Args:
            state_keys: list of fields in the observation spec that should be concatenated into the state (expects 1D)
            image_keys: list of fields that should be concatenated into the image. Assumes all images same size.
        """
        super(ImageControllerPredictor, self).__init__()

        self.env = env
        self.state_keys = state_keys
        self.image_keys = image_keys

        self.cnn_insize = [1] + list(self.env.observation_space[self.image_keys[0]].shape)
        self.cnn_outsize = cnn_outsize

        self.insize = sum([env.observation_space[k].low.shape[0] for k in self.state_keys])
        self.outsize = seqs.shape[0] * 2 #Predict T/F for each action.

        self.model = model
        self.dt = dt
        self.sequences = seqs
        self.threshold = threshold

        self.cnn = ResnetCNN(self.cnn_insize, self.cnn_outsize, 1, [64, ], pool=4)
        self.mlp = MLP(self.insize + self.cnn_outsize, self.outsize, mlp_hiddens, mlp_activation)
        self.device = device

    def forward(self, obs):
        """
        Should predict all seqs from obs only.
        """
        images = torch.cat([obs[k] for k in self.image_keys], dim=-3).unsqueeze(1)
        image_features = self.cnn(images)
        state = torch.cat([obs[k] for k in self.state_keys], dim=-1)
        mlp_in = torch.cat([state, image_features], dim=-1)
        logits = self.mlp(mlp_in)
        keep_shape = logits.shape[:-1]
        logits = logits.view(*keep_shape, self.sequences.shape[0], 2)
        return logits

    def model_rollout(self, obs):
        """
        Roll out all sequences for ONE given state.
        """
        x = observation_to_model_state(obs['state'])
        x = x.repeat(self.sequences.shape[0], 1)
        for t in range(self.sequences.shape[1]):
            u = self.sequences[:, t]
            x = self.model.forward(x, u, dt=self.dt)

        return x

    def get_training_targets(self, batch):
        """
        Get the labels for a batch by forward-simulating the model and checking if the actual result was inside the threshold.
        """
        cseqs = batch['action'].squeeze()
        x = observation_to_model_state(batch['observation']['state'])
        for t in range(self.sequences.shape[1]):
            u = self.sequences[cseqs, t]
            x = self.model.forward(x, u, dt=self.dt)

        xpos = x[:, :2]
        gt_xpos = batch['next_observation']['state'][:, :2]
        dist = torch.linalg.norm(xpos - gt_xpos, dim=-1)
        labels = (dist > self.threshold).long() #class 0 = True, class 1 = False
        labels[self.sequences[cseqs, 0, 0] == 0] = 0 #always allow staying still
        return labels

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.sequences = self.sequences.to(device)
        self.mlp = self.mlp.to(device)
        self.cnn = self.cnn.to(device)
        return self

if __name__ == '__main__':
    from wheeledSim.FlatRockyTerrain import FlatRockyTerrain
    from wheeledsim_rl.envs.pybullet_sim import WheeledSimEnv

    from torch import optim

    terrain = FlatRockyTerrain()
    env = WheeledSimEnv(terrainParamsIn={'N':50}, existingTerrain = terrain, senseParamsIn={"senseResolution":[32, 32]}, T=30, use_images=True)
    
    buf = torch.load('img_buffer.pt')
    model_params = torch.load('KBM_params.pt')
    seqs = torch.load('sequences.pt')
    model = KinematicBicycleModel(hyperparams=model_params)

    import pdb;pdb.set_trace()
    predictor = ImageControllerPredictor(env, model, seqs, state_keys=['state'], image_keys=['image'])

    buf = buf.to('cuda')
    predictor = predictor.to('cuda')

    opt = optim.Adam(predictor.parameters())
    criterion = nn.CrossEntropyLoss()

    for i in range(1000):
#        batch = buf.sample(64)
        batch = buf.sample_idxs(torch.arange(len(buf)))
        labels = predictor.get_training_targets(batch)
        preds = predictor.forward(batch['observation'])
        acts = batch['action'].squeeze()
        preds = preds[torch.arange(acts.shape[0]), acts]
        loss = criterion(preds, labels)

        with torch.no_grad():
            acc = (preds.argmax(dim=-1) == labels).float().mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        print("ITR {}:".format(i+1))
        print("Loss = {:.6f}".format(loss.detach().item()))
        print("Accuracy = {:.6f}".format(acc.item()))

    import pdb;pdb.set_trace()
    predictor = predictor.to('cpu')
    torch.save(predictor.to('cpu'), 'predictor.pt')
