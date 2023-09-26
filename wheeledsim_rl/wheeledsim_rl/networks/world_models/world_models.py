import torch

from torch import nn, optim, distributions

from wheeledsim_rl.networks.world_models.cnn_vae import CNNEncoder, CNNDecoder
from wheeledsim_rl.networks.mlp import MLP

class WorldModel(nn.Module):
    """
    Implementation of the world models idea, i.e. use an RNN as a latent space model, and map state + obs thru it as encoders, decoders.
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
        self.rnn_num_layers = num_layers
        self.rnn = torch.nn.GRU(act_encoder_hidden_size[-1], self.rnn_hiddensize, batch_first=True, num_layers=self.rnn_layers) #I like batch-first. i.e. [batch x time x feat]

#        self.mlp = nn.Linear(self.state_dim + sum([net.latent_size for net in self.encoders.values()]), self.rnn_hiddensize)
#        self.mlp = MLP(self.state_dim + sum([net.latent_size for net in self.encoders.values()]), self.rnn_hiddensize, mlp_encoder_hidden_size)
        self.mlp = MLP(self.state_dim + self.encoders[self.obs_keys[0]].latent_size, self.rnn_hiddensize, mlp_encoder_hidden_size)
#        self.state_decoder = nn.Linear(self.rnn_hiddensize, self.state_dim)
        self.state_decoder = MLP(self.rnn_hiddensize, self.state_dim, mlp_decoder_hidden_size)

        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer

        self.device = device

    def predict(self, obs, act, return_info=False, keys=None):
        """
        Same as other predicts - add on the initial state.
        """
        if return_info:
            norm_obs = self.input_normalizer.normalize({'observation':obs})['observation']
            pred_obs = self.forward(norm_obs, act, return_info, keys)
            denorm_obs = self.output_normalizer.denormalize({'observation':pred_obs['observation']})['observation']
            denorm_obs['state'] = torch.cumsum(denorm_obs['state'], dim=-2) + obs['state'].unsqueeze(-2)
            return {
                    'observation': denorm_obs,
                    'encoder_dist':pred_obs['encoder_dist'],
                    'latent_observation':pred_obs['latent_observation']
                    }

        else:
            norm_obs = self.input_normalizer.normalize({'observation':obs})['observation']
            pred_obs = self.forward(norm_obs, act, return_info, keys)
            denorm_obs = self.output_normalizer.denormalize({'observation':pred_obs})['observation']
            denorm_obs['state'] = torch.cumsum(denorm_obs['state'], dim=-2) + obs['state'].unsqueeze(-2)
            return denorm_obs

    def forward(self, obs, act, return_info=False, keys=None):
        """
        Args:
            obs: batched as {<key>:[batch x feat]}
            act: a series of actions batched as [batch x time x feat]
            return_info: if false, just return the predictions. If True, return a dict containing:
                observation: The prediction
                encoder_dist: The encoder's mapping of obs to the latent space
                latent_observation: The embedding of the predictions in latent space
        Returns:
            preds: batched as {<obs label>:[batch x time x feat]}
        """
#        encoder_dists = {k:self.encoders[k].forward(obs[k], return_dist=True) for k in self.obs_keys}
#        inp = torch.cat([obs['state']] + [encoder_dists[k].rsample() for k in self.obs_keys], dim=-1)
#        h0 = self.mlp.forward(inp).unsqueeze(0)

        h0, encoder_dists = self.encode_observation(obs, rsample=True, normalize_obs=False, return_encoder_dists=True)
#        h0, encoder_dists = self.encode_observation(obs, rsample=False, normalize_obs=False, return_encoder_dists=True)
        out, hf = self.rnn.forward(self.act_mlp.forward(act), torch.stack(self.rnn_num_layers * [h0], dim=0))

        #Out is what matters here. Is [batch x time x hidden dim]
        #Rebuild into dict of obs keys + state
        states = self.state_decoder.forward(out)

        predict_keys = self.obs_keys if keys is None else [k for k in keys if k in self.obs_keys]
        obses = {k:self.decoders[k].forward(out) for k in predict_keys}
        obses['state'] = states

        if return_info:
            return {
                    'observation': obses,
                    'encoder_dist': encoder_dists,
                    'latent_observation':out
                    }
        else:
            return obses

    def hidden_predict(self, obs, h0, act, return_info=False, keys=None):
        """
        Same as other predicts - add on the initial state.
        """
        if return_info:
            pred_obs = self.hidden_forward(h0, act, return_info, keys)
            denorm_obs = self.output_normalizer.denormalize({'observation':pred_obs['observation']})['observation']
            denorm_obs['state'] = torch.cumsum(denorm_obs['state'], dim=-2) + obs['state'].unsqueeze(-2)
            return {
                    'observation': denorm_obs,
                    'hidden_observation':pred_obs['hidden_observation'],
                    'latent_observation':pred_obs['latent_observation']
                    }

        else:
            pred_obs = self.hidden_forward(h0, act, return_info, keys)
            denorm_obs = self.output_normalizer.denormalize({'observation':pred_obs})['observation']
            denorm_obs['state'] = torch.cumsum(denorm_obs['state'], dim=-2) + obs['state'].unsqueeze(-2)
            return denorm_obs

    def hidden_forward(self, h0, act, return_info=False, keys=None):
        """
        Args:
            h0: The hidden state to pass into the RNN at init
            act: a series of actions batched as [batch x time x feat]
            return_info: if false, just return the predictions. If True, return a dict containing:
                observation: The prediction
                encoder_dist: The encoder's mapping of obs to the latent space
                latent_observation: The embedding of the predictions in latent space
        Returns:
            preds: batched as {<obs label>:[batch x time x feat]}
        """
        out, hf = self.rnn.forward(self.act_mlp.forward(act), torch.stack(self.rnn_num_layers * [h0], dim=0))

        #Out is what matters here. Is [batch x time x hidden dim]
        #Rebuild into dict of obs keys + state
        states = self.state_decoder.forward(out)

        predict_keys = self.obs_keys if keys is None else [k for k in keys if k in self.obs_keys]
        obses = {k:self.decoders[k].forward(out) for k in predict_keys}
        obses['state'] = states

        if return_info:
            return {
                    'observation': obses,
                    'hidden_observation': hf,
                    'latent_observation':out
                    }
        else:
            return obses

    def encode_observation(self, obs, rsample=False, normalize_obs=True, return_encoder_dists=False):
        """
        Encode observations for contrastive losses
        Args:
            obs: The data to get the latent of
            rsample: Whether to sample from the VAE encoder or take the mean
            normalize_obs: Whether to pass obs through the input normalizer first. NOTE: It's VERY important to get this arg right.
        """
        norm_obs = self.input_normalizer.normalize({'observation':obs})['observation'] if normalize_obs else obs

        #Have to flatten and unflatten to allow arbitrary number of batchdims
        batchshape = obs['state'].shape[:-1]
        flat_obs = {k:norm_obs[k].view(-1, *self.insize[k]) for k in self.obs_keys + ['state']}

        flat_encoder_dists = {k:self.encoders[k].forward(flat_obs[k], return_dist=True) for k in self.obs_keys}
        encoder_dists = {k:distributions.Normal(loc=v.loc.view(*batchshape, self.encoders[k].latent_size), scale=v.scale.view(*batchshape, self.encoders[k].latent_size)) for k,v in flat_encoder_dists.items()}

#        import pdb;pdb.set_trace()
        #TODO: Implement as a product of Gaussian Experts.
        #There is a closed-form for diagonal gaussians.
        stacked_dist = distributions.Normal(loc=torch.stack([d.mean for d in encoder_dists.values()], dim=1), scale=torch.stack([d.scale for d in encoder_dists.values()], dim=1))
        recip_std = (1./stacked_dist.scale).sum(dim=1)
        poe_mean = (stacked_dist.mean/stacked_dist.scale).sum(dim=1) * (1./recip_std)
        poe_scale = 1./recip_std

        poe_dist = distributions.Normal(poe_mean, poe_scale)
        latents = poe_dist.rsample()

#        latents = torch.stack([(encoder_dists[k].rsample() if rsample else encoder_dists[k].mean) for k in self.obs_keys], dim=-1)
#        latents = latents.mean(dim=-1)
        inp = torch.cat([flat_obs['state'], latents], dim=-1)
        h0 = self.mlp.forward(inp)

        if return_encoder_dists:
            return h0, encoder_dists
        else:
            return h0

    def get_training_targets(self, batch):
        """
        Extract features and states for reconstruction loss.
        """
        targets = {k:batch['next_observation'][k] for k in self.obs_keys + ['state']}
        targets['state'] -= batch['observation']['state']

        return targets

    def to(self, device):
        self.device = device
        self.encoders = nn.ModuleDict({k:net.to(device) for k,net in self.encoders.items()})
        self.decoders = nn.ModuleDict({k:net.to(device) for k,net in self.decoders.items()})
        self.mlp = self.mlp.to(device)
        self.act_mlp = self.act_mlp.to(device)
        self.state_decoder = self.state_decoder.to(device)
        self.rnn = self.rnn.to(device)
        self.input_normalizer = self.input_normalizer.to(device)
        self.output_normalizer = self.output_normalizer.to(device)
        return self

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
