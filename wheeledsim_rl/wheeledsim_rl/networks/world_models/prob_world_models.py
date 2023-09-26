import torch

from torch import nn, optim, distributions

from wheeledsim_rl.networks.world_models.world_models import WorldModel
from wheeledsim_rl.networks.mlp import MLP
from wheeledsim_rl.networks.gaussian_mlp import GaussianMLP

class ProbWorldModel(WorldModel):
    """
    Same as world model, but use a Gaussian on the state decoder for aleatoric uncertainty
    """
    def __init__(self, encoders, decoders, state_insize, action_insize, rnn_hidden_size, rnn_layers, mlp_encoder_hidden_size, mlp_decoder_hidden_size, act_encoder_hidden_size, input_normalizer, output_normalizer, device='cpu'):
        """
        Args:
            encoders: A dict of {<obs key>:<network>} that maps each obs to a latent
            decoders: A dict of {<obs key>:<network>} that maps latent back to obs.
            state_insize: An int with the dim of the state
            hidden_size: The size of the hidden state
            num_layers: Number of RNN layers
        """
        super(WorldModel, self).__init__()

        assert encoders.keys() == decoders.keys(), "Mismatch between encoder and decoder keys. Encoders = {}, Decoders = {}".format(encoders.keys(), decoders.keys())
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)
        self.state_dim = state_insize
        self.act_dim = action_insize
        self.insize = {k:v.insize for k,v in self.encoders.items()}
        self.insize['state'] = torch.Size([state_insize])

        self.obs_keys = list(self.encoders.keys())

        self.act_mlp = MLP(self.act_dim, act_encoder_hidden_size[-1], act_encoder_hidden_size[:-1])

        self.rnn_hiddensize = rnn_hidden_size # + sum([net.latent_size for net in self.encoders.values()])
        self.rnn_num_layers = rnn_layers
        self.rnn = torch.nn.GRU(act_encoder_hidden_size[-1], self.rnn_hiddensize, batch_first=True, num_layers=rnn_layers) #I like batch-first. i.e. [batch x time x feat]

#        self.mlp = MLP(self.state_dim + sum([net.latent_size for net in self.encoders.values()]), self.rnn_hiddensize, mlp_encoder_hidden_size, dropout=0.0)
        self.mlp = MLP(self.state_dim + self.encoders[self.obs_keys[0]].latent_size, self.rnn_hiddensize, mlp_encoder_hidden_size)
        self.state_decoder = GaussianMLP(self.rnn_hiddensize, self.state_dim, mlp_decoder_hidden_size, dropout=0.0)

        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer

        self.device = device

    def predict(self, obs, act, return_info=False, keys=None):
        """
        Handle differently from base world model. Have to append existing state/preds to the distribution mean
        """
        if return_info:
            norm_obs = self.input_normalizer.normalize({'observation':obs})['observation']
            pred_obs = self.forward(norm_obs, act, return_info, keys)
            denorm_obs = self.output_normalizer.denormalize({'observation':pred_obs['observation']})['observation']

            denorm_obs['state'] = torch.distributions.Normal(
                    loc = torch.cumsum(denorm_obs['state'].mean, dim=-2) + obs['state'].unsqueeze(-2), 
                    scale=denorm_obs['state'].scale
                    )

            return {
                    'observation': denorm_obs,
                    'encoder_dist':pred_obs['encoder_dist'],
                    'latent_observation':pred_obs['latent_observation']
                    }

        else:
            norm_obs = self.input_normalizer.normalize({'observation':obs})['observation']
            pred_obs = self.forward(norm_obs, act, return_info, keys)
            denorm_obs = self.output_normalizer.denormalize({'observation':pred_obs})['observation']

            denorm_obs['state'] = torch.distributions.Normal(
                    loc = torch.cumsum(denorm_obs['state'].mean, dim=-2) + obs['state'].unsqueeze(-2), 
                    scale=denorm_obs['state'].scale
                    )

            return denorm_obs

    def hidden_predict(self, obs, h0, act, return_info=False, keys=None):
        """
        Same as other predicts - add on the initial state.
        """
        if return_info:
            pred_obs = self.hidden_forward(h0, act, return_info, keys)
            denorm_obs = self.output_normalizer.denormalize({'observation':pred_obs['observation']})['observation']
            denorm_obs['state'] = torch.distributions.Normal(
                    loc = torch.cumsum(denorm_obs['state'].mean, dim=-2) + obs['state'].unsqueeze(-2), 
                    scale=denorm_obs['state'].scale
                    )
            return {
                    'observation': denorm_obs,
                    'hidden_observation':pred_obs['hidden_observation'],
                    'latent_observation':pred_obs['latent_observation']
                    }

        else:
            pred_obs = self.hidden_forward(h0, act, return_info, keys)
            denorm_obs = self.output_normalizer.denormalize({'observation':pred_obs})['observation']
            denorm_obs['state'] = torch.distributions.Normal(
                    loc = torch.cumsum(denorm_obs['state'].mean, dim=-2) + obs['state'].unsqueeze(-2), 
                    scale=denorm_obs['state'].scale
                    )
            return denorm_obs

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img_topic1='front_camera'
    img_topic2='heightmap'
    
    buf = torch.load('buffer.pt')
    batch = buf.sample(5, N=3)
    print(len(buf))
    img1 = batch['observation'][img_topic1][:, 0]
    img2 = batch['observation'][img_topic2][:, 0]

    rnn_hidden = 64

    encoder1 = CNNEncoder(cnn_insize=img1.shape[1:], latent_size=16)
    decoder1 = CNNDecoder(cnn_outsize=img1.shape[1:], channels=[64, 32, 16, 8], latent_size=rnn_hidden)

    encoder2 = CNNEncoder(cnn_insize=img2.shape[1:], latent_size=16)
    decoder2 = CNNDecoder(cnn_outsize=img2.shape[1:], channels=[64, 32, 16, 8], latent_size=rnn_hidden)
    
    model = WorldModel({img_topic1:encoder1, img_topic2:encoder2}, {img_topic1:decoder1, img_topic2:decoder2}, state_insize=13, action_insize=2, hidden_size=rnn_hidden, rnn_layers=1)

    batch = buf.sample(3, N=10)
    z, dist = model.forward({k:v[:, 0] for k,v in batch['observation'].items()}, batch['action'], return_dist=True)
    print({k:v.shape for k,v in z.items()})
    print({k:v.mean.shape for k,v in dist.items()})
