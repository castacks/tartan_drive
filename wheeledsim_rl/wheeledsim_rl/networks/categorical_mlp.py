import torch

from torch import nn, distributions

from wheeledsim_rl.distributions import StraightThroughOneHotCategorical

class CategoricalMLP(nn.Module):
    def __init__(self, insize, outsize, outwidth, hiddens, hidden_activation=nn.Tanh, dropout=0.0, device='cpu'):
        """
        Network that outputs a set of (differentiable) categoricals of size [outsize x outwidth]
        """
        super(CategoricalMLP, self).__init__()

        layer_sizes = [insize] + list(hiddens) + [outsize * outwidth]

        self.outsize = outsize
        self.outwidth = outwidth
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

        logits = out.view(*inp.shape[:-1], self.outsize, self.outwidth) #.exp() + 1e-6 #Add a numerical stability constant

        return StraightThroughOneHotCategorical(logits=logits)

    def to(self, device):
        self.device = device
        self.activation.to(device)
        self.dropout.to(device)
        self.layers.to(device)
        return self

if __name__ == '__main__':
    """
    Set up a toy task.
    """
    import matplotlib.pyplot as plt

    target_dist = StraightThroughOneHotCategorical(torch.rand(8, 8))

    net = CategoricalMLP(10, 8, 8, [32,])
    opt = torch.optim.Adam(net.parameters())

    epochs = 1000
    import pdb;pdb.set_trace()
    for e in range(epochs):
        x = torch.rand(32, 10)
        pred_dist = net.forward(x)

        kld = torch.distributions.kl_divergence(target_dist, pred_dist)

        kl_loss = kld.mean()
        opt.zero_grad()
        kl_loss.backward()
        opt.step()

        print('EPOCH {}\n\tKL = {:.6f}'.format(e+1, kl_loss.detach()))

    import pdb;pdb.set_trace()
