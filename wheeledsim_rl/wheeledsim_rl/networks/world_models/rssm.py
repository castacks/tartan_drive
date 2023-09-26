import torch

from torch import nn, optim, distributions

from wheeledsim_rl.networks.world_models.world_models import WorldModel
from wheeledsim_rl.networks.mlp import MLP
from wheeledsim_rl.networks.gaussian_mlp import GaussianMLP
from wheeledsim_rl.networks.categorical_mlp import CategoricalMLP
from wheeledsim_rl.distributions import StraightThroughOneHotCategorical

class RecurrentStateSpaceModel(nn.Module):
    """
    From Hafner et al 2018, use an RNN that has both stochastic and deterministic components.
    Keep in mind that there are two latents here: h=the deterministic path through the RNN. z=the stochastic latent space.
    We enforce a loss with the following parts:
        1. Reconstruction/contrastive. Reconstruct preds in latent/observation space
        2. KL loss. Minimize the KL-divergence bet. the state prior z from the model and state posterior z from the encoder.
    This means that we have the following:
        1. RNN: h_t = f(h_t-1, z_t-1, a_t-1)
        2. Latent state model z_t ~ g(.|h_t)
        3. Observation decoder: o_t ~ h(.|z_t) OR Observation  encoder z_t ~ h(.|o_t)
        4. State posterior encoder: z_t ~ q(.|o_t, h_t). Use to a) Get the initial latent state and b) state posterior for training.
    """
    def __init__(self, encoders, decoders, state_insize, action_insize, rnn_hidden_size, latent_size, mlp_encoder_hidden_size, mlp_decoder_hidden_size, act_encoder_hidden_size, input_normalizer, output_normalizer, device='cpu'):
        """
        Args:
            encoders: A dict of {<obs key>:<network>} that maps each obs to a latent
            decoders: A dict of {<obs key>:<network>} that maps latent back to obs.
            state_insize: An int with the dim of the state
            hidden_size: The size of the hidden state
            num_layers: Number of RNN layers
        """
        super(RecurrentStateSpaceModel, self).__init__()

        assert encoders.keys() == decoders.keys(), "Mismatch between encoder and decoder keys. Encoders = {}, Decoders = {}".format(encoders.keys(), decoders.keys())
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)
        self.state_dim = state_insize
        self.act_dim = action_insize
        self.insize = {k:v.insize for k,v in self.encoders.items()}
        self.insize['state'] = torch.Size([state_insize])

        self.obs_keys = list(self.encoders.keys())

        self.act_mlp = MLP(self.act_dim, act_encoder_hidden_size[-1], act_encoder_hidden_size[:-1])

        self.rnn_hiddensize = rnn_hidden_size
        self.latent_size = latent_size

        #RNN: h_t = f(h_t-1, z_t-1, a_t-1)
        self.rnn = torch.nn.GRUCell(self.latent_size + act_encoder_hidden_size[-1], self.rnn_hiddensize)

        #Latent state model: z_t ~ g(.|h_t)
        self.latent_state_decoder = GaussianMLP(self.rnn_hiddensize, latent_size, mlp_decoder_hidden_size)

        #State posterior encoder: z_t ~ q(.|o_t, h_t)
#        self.mlp = MLP(self.state_dim + sum([net.latent_size for net in self.encoders.values()]), self.latent_size, mlp_encoder_hidden_size)
        self.state_posterior_encoder = GaussianMLP(self.rnn_hiddensize + self.state_dim + sum([net.latent_size for net in self.encoders.values()]), latent_size, mlp_encoder_hidden_size)

        #Decoder to recover the actual state from the latent state.
        self.state_decoder = GaussianMLP(self.latent_size + self.rnn_hiddensize, self.state_dim, hiddens=[])

        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer

        self.device = device

    def predict(self, obs, act, return_info=False):
        """
        Handle differently from base world model. Have to append existing state/preds to the distribution mean
        """
        if return_info:
            norm_obs = self.input_normalizer.normalize({'observation':obs})['observation']
            pred_obs = self.forward(norm_obs, act, return_info)
            denorm_obs = self.output_normalizer.denormalize({'observation':pred_obs['observation']})['observation']

            denorm_obs['state'] = torch.distributions.Normal(
                    loc = torch.cumsum(denorm_obs['state'].mean, dim=-2) + obs['state'].unsqueeze(-2), 
                    scale=denorm_obs['state'].scale
                    )

            return {
                    'observation': denorm_obs,
                    'hidden_states':pred_obs['hidden_states'],
                    'latent_states':pred_obs['latent_states'],
                    'latent_prior':pred_obs['latent_prior']
                    }

        else:
            norm_obs = self.input_normalizer.normalize({'observation':obs})['observation']
            pred_obs = self.forward(norm_obs, act, return_info)
            denorm_obs = self.output_normalizer.denormalize({'observation':pred_obs})['observation']

            denorm_obs['state'] = torch.distributions.Normal(
                    loc = torch.cumsum(denorm_obs['state'].mean, dim=-2) + obs['state'].unsqueeze(-2), 
                    scale=denorm_obs['state'].scale
                    )

            return denorm_obs

    def forward(self, obs, act, return_info=False):
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
        h0 = torch.zeros(*obs['state'].shape[:-1], self.rnn_hiddensize, device=self.device)
        z0 = self.get_latent_posterior(obs, h0).rsample()

        zs = []
        hs = []
        zdists = []
        curr_z = z0
        curr_h = h0
        for t in range(act.shape[-2]):
            act_embed = self.act_mlp.forward(act[:, t])
            rnn_in = torch.cat([curr_z, act_embed], dim=-1)
            curr_h = self.rnn.forward(rnn_in, curr_h)
            hs.append(curr_h)

            zdist = self.latent_state_decoder(curr_h)
            zdists.append(zdist)
            curr_z = zdist.rsample()
            zs.append(curr_z)

        hs = torch.stack(hs, dim=-2)
        zs = torch.stack(zs, dim=-2)
        zdists = torch.distributions.Normal(loc=torch.stack([x.mean for x in zdists], dim=-2), scale=torch.stack([x.scale for x in zdists], dim=-2))

        #Out is what matters here. Is [batch x time x hidden dim]
        #Rebuild into dict of obs keys + state
        decoder_in = torch.cat([zs, hs], dim=-1)
        obses = {k:self.decoders[k].forward(decoder_in) for k in self.obs_keys}
        obses['state'] = self.state_decoder.forward(decoder_in)

        if return_info:
            return {
                    'observation': obses,
                    'hidden_states': hs,
                    'latent_states':zs,
                    'latent_prior':zdists
                    }
        else:
            return obses

    def get_latent_posterior(self, obs, h, rsample=False, normalize_obs=True):
        """
        Get the posterior distribution of latent state z by running the encoder on batches of obs, h.
        Args:
            obs: A dict of tensors batched as [batch x [obs feats]]
            h: A tensor batched as [batch x rnn_hidden_dim] (likely zeros for a lot of uses)
        """
        norm_obs = self.input_normalizer.normalize({'observation':obs})['observation'] if normalize_obs else obs

        #Have to flatten and unflatten to allow arbitrary number of batchdims
        batchshape = obs['state'].shape[:-1]
        flat_obs = {k:obs[k].view(-1, *self.insize[k]) for k in self.obs_keys + ['state']}

        flat_encoder_dists = {k:self.encoders[k].forward(flat_obs[k], return_dist=True) for k in self.obs_keys}
        encoder_dists = {k:distributions.Normal(loc=v.loc.view(*batchshape, self.encoders[k].latent_size), scale=v.scale.view(*batchshape, self.encoders[k].latent_size)) for k,v in flat_encoder_dists.items()}

        inp = torch.cat([h] + [obs['state']] + [(encoder_dists[k].rsample() if rsample else encoder_dists[k].mean) for k in self.obs_keys], dim=-1)

        return self.state_posterior_encoder.forward(inp)

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
        flat_obs = {k:obs[k].view(-1, *self.insize[k]) for k in self.obs_keys + ['state']}

        flat_encoder_dists = {k:self.encoders[k].forward(flat_obs[k], return_dist=True) for k in self.obs_keys}
        encoder_dists = {k:distributions.Normal(loc=v.loc.view(*batchshape, self.encoders[k].latent_size), scale=v.scale.view(*batchshape, self.encoders[k].latent_size)) for k,v in flat_encoder_dists.items()}

        inp = torch.cat([obs['state']] + [(encoder_dists[k].rsample() if rsample else encoder_dists[k].mean) for k in self.obs_keys], dim=-1)
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
        self.rnn = self.rnn.to(device)
        self.latent_state_decoder = self.latent_state_decoder.to(device)
        self.state_posterior_encoder = self.state_posterior_encoder.to(device)
        self.state_decoder = self.state_decoder.to(device)
        self.act_mlp = self.act_mlp.to(device)
        self.input_normalizer = self.input_normalizer.to(device)
        self.output_normalizer = self.output_normalizer.to(device)
        return self

class DiscreteRecurrentStateSpaceModel(RecurrentStateSpaceModel):
    """
    From Hafner et al 2021, use an RSSM, but replace the Gaussian zs with Categorical ones.
    Keep in mind that there are two latents here: h=the deterministic path through the RNN. z=the stochastic latent space.
    We enforce a loss with the following parts:
        1. Reconstruction/contrastive. Reconstruct preds in latent/observation space
        2. KL loss. Minimize the KL-divergence bet. the state prior z from the model and state posterior z from the encoder.
    This means that we have the following:
        1. RNN: h_t = f(h_t-1, z_t-1, a_t-1)
        2. Latent state model z_t ~ g(.|h_t)
        3. Observation decoder: o_t ~ h(.|z_t) OR Observation  encoder z_t ~ h(.|o_t)
        4. State posterior encoder: z_t ~ q(.|o_t, h_t). Use to a) Get the initial latent state and b) state posterior for training.
    """
    def __init__(self, encoders, decoders, state_insize, action_insize, rnn_hidden_size, latent_size, latent_width, mlp_encoder_hidden_size, mlp_decoder_hidden_size, input_normalizer, output_normalizer, device='cpu'):
        """
        Args:
            encoders: A dict of {<obs key>:<network>} that maps each obs to a latent
            decoders: A dict of {<obs key>:<network>} that maps latent back to obs.
            state_insize: An int with the dim of the state
            action_insize: An int with the dim of action
            rnn_hidden_size: The dim of h
            latent_size: The number of categorical variables to use
            latent_width: The number of classes per categorical variable
            mlp_encoder_hidden_size: The hidden layers for the linear portion of the posterior encoder
            mlp_decoder_hidden_size: The hidden layers for the mapping from h to z
        """
        nn.Module.__init__(self)

        assert encoders.keys() == decoders.keys(), "Mismatch between encoder and decoder keys. Encoders = {}, Decoders = {}".format(encoders.keys(), decoders.keys())
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)
        self.state_dim = state_insize
        self.act_dim = action_insize
        self.insize = {k:v.insize for k,v in self.encoders.items()}
        self.insize['state'] = torch.Size([state_insize])

        self.obs_keys = list(self.encoders.keys())

        self.rnn_hiddensize = rnn_hidden_size
        self.latent_size = latent_size
        self.latent_width = latent_width

        #RNN: h_t = f(h_t-1, z_t-1, a_t-1)
        self.rnn = torch.nn.GRUCell(self.latent_size*self.latent_width + self.act_dim, self.rnn_hiddensize)

        #Latent state model: z_t ~ g(.|h_t)
        self.latent_state_decoder = CategoricalMLP(self.rnn_hiddensize, latent_size, latent_width, mlp_decoder_hidden_size)

        #State posterior encoder: z_t ~ q(.|o_t, h_t)
#        self.mlp = MLP(self.state_dim + sum([net.latent_size for net in self.encoders.values()]), self.latent_size, mlp_encoder_hidden_size)
        self.state_posterior_encoder = CategoricalMLP(self.rnn_hiddensize + self.state_dim + sum([net.latent_size for net in self.encoders.values()]), latent_size, latent_width, mlp_encoder_hidden_size)

        #Decoder to recover the actual state from the latent state.
        self.state_decoder = GaussianMLP(self.latent_size*self.latent_width + self.rnn_hiddensize, self.state_dim, hiddens=[])

        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer

        self.device = device

    def forward(self, obs, act, return_info=False):
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
        h0 = torch.zeros(*obs['state'].shape[:-1], self.rnn_hiddensize, device=self.device)
        z0 = self.get_latent_posterior(obs, h0).rsample()

        zs = []
        hs = []
        zdists = []
        curr_z = z0
        curr_h = h0
        for t in range(act.shape[-2]):
            rnn_in = torch.cat([curr_z.flatten(start_dim=-2), act[:, t]], dim=-1)
            curr_h = self.rnn.forward(rnn_in, curr_h)
            hs.append(curr_h)

            zdist = self.latent_state_decoder(curr_h)
            zdists.append(zdist)
            curr_z = zdist.rsample()
            zs.append(curr_z)

        hs = torch.stack(hs, dim=-2)
        zs = torch.stack(zs, dim=-3)
        zdists = StraightThroughOneHotCategorical(torch.stack([d.probs for d in zdists], dim=-3))

        #Out is what matters here. Is [batch x time x hidden dim]
        #Rebuild into dict of obs keys + state
        decoder_in = torch.cat([zs.flatten(start_dim=-2), hs], dim=-1)
        obses = {k:self.decoders[k].forward(decoder_in) for k in self.obs_keys}
        obses['state'] = self.state_decoder.forward(decoder_in)

        if return_info:
            return {
                    'observation': obses,
                    'hidden_states': hs,
                    'latent_states':zs,
                    'latent_prior':zdists
                    }
        else:
            return obses
