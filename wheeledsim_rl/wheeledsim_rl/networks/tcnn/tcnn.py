import torch

from torch import nn, distributions

class TanhSigmoidActivation(nn.Module):
    """
    Implements the activation function tanh(x) * sigmoid(x) per van den Oord 2016.
    """
    def __init__(self):
        super(TanhSigmoidActivation, self).__init__()

    def forward(self, f, g):
        tanh = torch.tanh(f)
        sig = torch.sigmoid(g)

        return tanh * sig


class Causal1DConv(nn.Module):
    """
    Implements a 1D causal convolution with dilation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, device='cpu'):
        super(Causal1DConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=self.dilation)
        self.device = device

    def forward(self, inp):
        c_out = self.conv(inp)
        if self.padding > 0:
            chomp = c_out[:, :, :-self.padding]
        else:
            chomp = c_out
        return chomp

    def to(self, device):
        self.device = device
        self.conv = self.conv.to(device)
        return self

class TCNNDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, out_length=None, activation=nn.ReLU, device='cpu'):
        super(TCNNDeconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = [out_length, 1]
        self.scale = scale
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = activation()
        self.device = device

    def forward(self, x):
        extra_dims = x.shape[:-2]
        _x = x.flatten(end_dim=-3)

        if self.out_shape[0] is None:
            out_shape = [self.scale * x.shape[-1], 1]
            _x = nn.functional.interpolate(_x.unsqueeze(-1), size=out_shape).squeeze(-1)
        else:
            _x = nn.functional.interpolate(_x.unsqueeze(-1), size=self.out_shape).squeeze(-1)

        _x = self.conv(_x)
        _x = _x.view(extra_dims + _x.shape[-2:])
        _x = self.activation(_x)
        return _x

    def to(self, device):
        self.device = device
        self.conv = self.conv.to(device)
        self.activation = self.activation.to(device)
        return self

class TCNN(nn.Module):
    def __init__(self, insize, outsize, hidden_dims, kernel_sizes, dilations, activation=nn.Tanh):
        """
        Args:
            insize: A 2-tuple of (seq length, seq inchannels)
            outsize: The number of output features
            hidden_dims: The intermediate channel sizes
            kernel_sizes: The kernel sizes
            dilations: The dilation scale of each level
        """
        super(TCNN, self).__init__()
        assert len(kernel_sizes) == len(dilations) == len(hidden_dims), 'expected kernel sizes, hidden dims and \
        dilations to match, got kernels = {}, hidden dims = {}, dilations ={}'.format(len(kernel_sizes), len(dilations), len(hidden_dims))

        self.in_length, self.in_channels = insize
        self.outsize = outsize
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.hidden_dims = hidden_dims

        self.activation = activation()

        self.convs = nn.ModuleList()

        dims = [self.in_channels] + self.hidden_dims

        self.linear = nn.Linear(self.hidden_dims[-1]*self.in_length, outsize)

        for i in range(len(dims) - 1):
            self.convs.append(Causal1DConv(dims[i], dims[i+1], self.kernel_sizes[i], self.dilations[i]))

    def forward(self, inp):
        out = inp
        for conv in self.convs:
            out = self.activation(conv(out))

        out = self.linear(out.flatten(start_dim=-2))

        return out

class WaveNetDecoder(nn.Module):
    def __init__(self, insize, outsize, hidden_dims, scales, device='cpu'):
        """
        Args:
            insize: The number of input features
            outsize: A 2-tuple of (final seq length, seq channels)
            hidden_dims: A list of the intermediate channel sizes
            scales: The amout to dilate by each layer
        """
        super(WaveNetDecoder, self).__init__()
        self.insize = insize
        self.out_length, self.out_channels = outsize
        self.hidden_dims = hidden_dims
        self.scales = scales

        self.activation = TanhSigmoidActivation()
        self.filter_blocks = nn.ModuleList()
        self.gate_blocks = nn.ModuleList()

        dims = [self.insize] + self.hidden_dims + [self.out_channels]

        for i in range(len(dims) - 2):
            self.filter_blocks.append(TCNNDeconv(dims[i], dims[i+1], scale=scales[i], activation=nn.Identity))
            self.gate_blocks.append(TCNNDeconv(dims[i], dims[i+1], scale=scales[i], activation=nn.Identity))

        self.last_filter = TCNNDeconv(dims[-2], dims[-2], out_length=self.out_length, activation=nn.Identity)
        self.last_conv = nn.Conv1d(dims[-2], dims[-1], kernel_size=1)

        self.device = device

    def forward(self, x):
        extra_dims = x.shape[:-1]
        _x = x.flatten(end_dim=-2).unsqueeze(-1)

        for filter_conv, gate_conv in zip(self.filter_blocks, self.gate_blocks):
            f = filter_conv(_x)
            g = gate_conv(_x)
            _x = self.activation(f, g)

        _x = self.last_filter(_x)
        _x = self.last_conv(_x)
        _x = _x.view(extra_dims + _x.shape[-2:])

        return _x.swapdims(-2, -1)

    def to(self, device):
        self.device = device
        self.activation = self.activation.to(device)
        self.filter_blocks = nn.ModuleList([conv.to(device) for conv in self.filter_blocks])
        self.gate_blocks = nn.ModuleList([conv.to(device) for conv in self.gate_blocks])
        self.last_filter = self.last_filter.to(device)
        self.last_conv = self.last_conv.to(device)
        return self

class WaveNetEncoder(nn.Module):
    def __init__(self, insize, latent_size, hidden_dims, kernel_sizes, dilations, device='cpu'):
        """
        Args:
            insize: A 2-tuple of (seq length, seq inchannels)
            outsize: The number of output features
            hidden_dims: The intermediate channel sizes
            kernel_sizes: The kernel sizes
            dilations: The dilation scale of each level
        """
        super(WaveNetEncoder, self).__init__()
        assert len(kernel_sizes) == len(dilations) == len(hidden_dims), 'expected kernel sizes, hidden dims and \
        dilations to match, got kernels = {}, hidden dims = {}, dilations ={}'.format(len(kernel_sizes), len(dilations), len(hidden_dims))

        self.in_length, self.in_channels = insize
        self.insize = insize
        self.latent_size = latent_size
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.hidden_dims = hidden_dims

        self.activation = TanhSigmoidActivation()

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        dims = [self.in_channels] + self.hidden_dims

        self.linear = nn.Linear(self.hidden_dims[-1]*self.in_length, 2*self.latent_size)

        for i in range(len(dims) - 1):
            self.filter_convs.append(Causal1DConv(dims[i], dims[i+1], self.kernel_sizes[i], self.dilations[i]))
            self.gate_convs.append(Causal1DConv(dims[i], dims[i+1], self.kernel_sizes[i], self.dilations[i]))

        self.device = device

    def forward(self, inp, return_dist=True):
        _x = inp.swapaxes(-1, -2)
        for filter_conv, gate_conv in zip(self.filter_convs, self.gate_convs):
            f = filter_conv(_x)
            g = gate_conv(_x)
            _x = self.activation(f, g)

        _x = self.linear(_x.flatten(start_dim=-2))

        if return_dist:
            means = torch.index_select(_x, -1, torch.arange(self.latent_size, device=self.device))
            stds = torch.index_select(_x, -1, torch.arange(self.latent_size, device=self.device)+self.latent_size).exp() + 1e-6 #numerical stability
            try:
                dist = distributions.Normal(loc=means, scale=stds)
            except:
                import pdb;pdb.set_trace()

            return dist
        else:
            return _x

    def to(self, device):
        self.device = device
        self.activation = self.activation.to(device)
        self.filter_convs = nn.ModuleList([conv.to(device) for conv in self.filter_convs])
        self.gate_convs = nn.ModuleList([conv.to(device) for conv in self.gate_convs])
        self.linear = self.linear.to(device)
        return self

if __name__ == '__main__':
    """
    Toy task: predict function class (one of 3 x [sin, tanh])
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import pi
    def generate_data(N, T):
        channels = torch.randint(2, size=(N, 3))

        xs = torch.linspace(-2*pi, 2*pi, T).unsqueeze(0).unsqueeze(0)
        shifts = (torch.rand(N, 3, 1)-0.5) * 2 * pi #[-pi ,pi]
        scales = (torch.rand(N, 3, 1)-0.5) * 4      #[-2, 2]
        sins = (xs + shifts).sin() * scales
        tanhs = (xs + shifts).tanh() * scales

        data = (sins * channels.unsqueeze(-1)) + (tanhs * (1-channels.unsqueeze(-1)))

        labels = channels[:, 0] + 2*channels[:, 1] + 4*channels[:, 2]
        return data, labels

    def viz(x, y):
        """
        viz a single datapoint
        """
        fns = ['tanh', 'sin']
        for i, xx in enumerate(x):
            plt.plot(xx, label='channel {}'.format(i+1))
        plt.title("Class {}, ({}, {}, {})".format(y.item(), fns[y % 2], fns[(y//2) % 2], fns[(y//4) % 2]))
        plt.legend()
        plt.show()

    def viz_net(x, y, net):
        """
        viz a single datapoint
        """
        fns = ['tanh', 'sin']
        with torch.no_grad():
            py = net.forward(x.unsqueeze(0))

        for i, xx in enumerate(x):
            plt.plot(xx, label='channel {}'.format(i+1))
        plt.title("Class {}, Pred {}, ({}, {}, {})".format(y.item(), py.argmax().item(), fns[y % 2], fns[(y//2) % 2], fns[(y//4) % 2]))
        plt.legend()
        plt.show()

    train_n = 10000
    test_n = 10000
    T = 100
    epochs = 10000
    B = 32

    X, Y = generate_data(train_n, T)
    Xtest, Ytest = generate_data(test_n, T)

    net = WaveNet([T, 3], 8, hidden_dims=[6, 12, 24], kernel_sizes=[2, 2, 2], dilations=[2, 4, 8])
#    net = TCNN([T, 3], 8, hidden_dims=[6, 12, 24], kernel_sizes=[2, 2, 2], dilations=[2, 4, 8])

    deconv = WaveNetDecoder(300, [50, 3], hidden_dims = [128, 64, 32, 16, 8], scales=[2, 2, 2, 2, 2])
    print("NET NPARAMS = {}".format(sum([np.prod(x.shape) for x in deconv.parameters()])))

    opt = torch.optim.Adam(deconv.parameters())
    for i in range(1000):
        pred = deconv(torch.rand(1000, 300))
        gt = torch.stack([torch.linspace(0, 6.28, 50).sin(), torch.linspace(0, 6.28, 50).cos(), torch.linspace(0, 6.28, 50).tanh()]) * 2
        loss = (pred - gt).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('Loss = {:.6f}\r'.format(loss.detach()), end='\r')

    print(loss)

    with torch.no_grad():
        plt.plot(deconv(torch.rand(1, 300)).squeeze().T)
        plt.show()

    exit(0)

    opt = torch.optim.Adam(net.parameters())
    print(net)
    print("NET NPARAMS = {}".format(sum([np.prod(x.shape) for x in net.parameters()])))
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(epochs):
        bidxs = torch.randint(train_n, (B, ))
        bx = X[bidxs]
        by = Y[bidxs]
        py = net.forward(bx)

        loss = criterion(py, by) 
        opt.zero_grad()
        loss.backward()
        opt.step()

        tbidxs = torch.randint(test_n, (B, ))
        bxt = Xtest[tbidxs]
        byt = Ytest[tbidxs]
        with torch.no_grad():
            pyt = net.forward(bxt).argmax(dim=-1)
        correct = (byt == pyt)
        accuracy = 100 * (correct.sum() / len(correct))

        if (e % 100) == 0:
            print('Epoch = {}\n\tLoss = {:.6f}\n\tEval Acc = {:.2f}%'.format(e, loss.detach().item(), accuracy))

    for i in range(test_n):
        viz_net(Xtest[i], Ytest[i], net)
