import torch

from torch import nn, distributions

class GaussianMLP(nn.Module):
    def __init__(self, insize, outsize, hiddens, hidden_activation=nn.Tanh, dropout=0.0, device='cpu'):
        """
        MLP that outputs a diagonal Gaussian distribution of dim outsize
        """
        super(GaussianMLP, self).__init__()

        layer_sizes = [insize] + list(hiddens) + [2 * outsize]

        self.outsize = outsize
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

        means = torch.index_select(out, -1, torch.arange(self.outsize, device=self.device))
        stds = torch.index_select(out, -1, torch.arange(self.outsize, device=self.device) + self.outsize).exp().clamp(1e-3, 1e1) #TODO: For stability

        return distributions.Normal(means, stds)

    def to(self, device):
        self.device = device
        self.activation.to(device)
        self.dropout.to(device)
        self.layers.to(device)
        return self

class GaussianMLPEncoder(GaussianMLP):
    def __init__(self, insize, outsize, hiddens, hidden_activation=nn.Tanh, dropout=0.0, device='cpu'):
        if not isinstance(outsize, int):
            outsize = outsize[0]
        if isinstance(insize, int):
            insize = torch.Size([insize])
        super(GaussianMLPEncoder, self).__init__(insize[0], outsize, hiddens, hidden_activation, dropout, device)
        self.insize = insize
        self.outsize = outsize[0] if not isinstance(outsize, int) else outsize
        self.latent_size = self.outsize

    def forward(self, inp, return_dist=False):
        dist = super(GaussianMLPEncoder, self).forward(inp)
        return dist if return_dist else dist.loc

if __name__ == '__main__':
    """
    Set up a toy task.
    """
    import matplotlib.pyplot as plt

    x = 10. * (torch.rand(size=[10000, 2]) - 0.5)
#    fn = lambda x: (x[:, [0]] / 3.).exp()
    fn = lambda x: x[:, [0]]
    y = fn(x)
    eps = torch.randn_like(y) * x[:, [0]]/5.
    y += eps

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].scatter(x[:, 0], y, s=1)
    axs[1].scatter(x[:, 1], y, s=1)
    plt.show()

    net = GaussianMLP(x.shape[-1], y.shape[-1], hiddens=[16, 16])
    opt = torch.optim.Adam(net.parameters())
    print(net)

    for i in range(10000):
        bidxs = torch.randint(high=x.shape[0], size=[32, ])
        xb = x[bidxs]
        yb = y[bidxs]

        dist = net.forward(xb)

        probs = dist.log_prob(yb)

        #NLL
        loss = -probs.mean()

        """
        #RSAMPLE GARBAGE
        samples = dist.rsample()
        loss = (yb - samples).pow(2).mean()
        """

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (i%100) == 0:
            print('EPOCH = {}\t LOSS = {:.4f}'.format(i+1, loss.detach().item()))

    #Viz
    xt = 10. * (torch.rand(size=[1000, 2]) - 0.5)
    with torch.no_grad():
        yp = net(xt)
    ygt = fn(xt)
    eps = torch.randn_like(ygt) * xt[:, [0]]/5.
    ygt += eps

    print("TEST LOG LIKELIHOOD: {:.4f}".format(yp.log_prob(ygt).mean()))
    print("TEST MSE TO MEAN: {:.4f}".format((yp.mean - ygt).pow(2).mean()))

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].scatter(xt[:, 0], ygt, s=1, c='g', label='gt')
    axs[1].scatter(xt[:, 1], ygt, s=1, c='g', label='gt')
    axs[0].scatter(xt[:, 0], yp.mean, s=1, c='b', label='mean')
    axs[1].scatter(xt[:, 1], yp.mean, s=1, c='b', label='mean')
    axs[0].scatter(xt[:, 0], yp.mean - yp.scale, s=1, c='r', label='lb')
    axs[1].scatter(xt[:, 1], yp.mean - yp.scale, s=1, c='r', label='lb')
    axs[0].scatter(xt[:, 0], yp.mean + yp.scale, s=1, c='r', label='ub')
    axs[1].scatter(xt[:, 1], yp.mean + yp.scale, s=1, c='r', label='ub')

    for ax in axs:
        ax.legend()

    plt.show()
