import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, insize, outsize, hiddens, hidden_activation=nn.Tanh, dropout=0.0, device='cpu'):
        """
        Note: no output activation included. Leave that for the individual policies.
        """
        super(MLP, self).__init__()

        #Error check for the yaml
        if isinstance(insize, int):
            self.insize = torch.Size([insize])
        else:
            assert len(insize)==1, 'Insize is not int and is not a length-1 iterable'
            self.insize = insize

        if isinstance(outsize, int):
            self.outsize = torch.Size([outsize])
        else:
            assert len(outsize)==1, 'Insize is not int and is not a length-1 iterable'
            self.outsize = outsize

        layer_sizes = [self.insize[0]] + list(hiddens) + [self.outsize[0]]

        self.layers = nn.ModuleList()
        self.activation = hidden_activation()
        self.dropout = nn.Dropout(p=dropout)
        self.device=device

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

        self.to(self.device)

    def forward(self, inp):
        out = self.layers[0](inp)
        for layer in self.layers[1:]:
            out = self.activation(out)
            out = self.dropout(out)
            out = layer.forward(out)

        return out

    def to(self, device):
        self.device = device
        self.activation.to(device)
        self.dropout.to(device)
        self.layers.to(device)
        return self

class MLPDecoder(MLP):
    def __init__(self, insize, outsize, hiddens, hidden_activation=nn.Tanh, dropout=0.0, device='cpu'):
        super(MLPDecoder, self).__init__(insize, outsize, hiddens, hidden_activation, dropout, device)
        self.insize = self.insize[0]
        self.latent_size = self.insize
