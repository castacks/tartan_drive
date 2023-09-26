import numpy as np
import torch
from torch import nn

from wheeledsim_rl.networks.mlp import MLP

"""
A collection of basic CNN blocks to try.
"""

class ResnetCNN(nn.Module):
    def __init__(self, insize, outsize, n_blocks, mlp_hiddens, hidden_activation=nn.Tanh, dropout=0.0, pool=2, device='cpu'):
        """
        Args:
            insize: The size of the input images. Expects a 3-tuple (nchannels, height, width)
            outsize: A scalar, same as MLP
            n_blocks: The number of CNN blocks to use.
            The rest is the same as MLP
        """
        super(ResnetCNN, self).__init__()
        self.cnn_insize = insize
        self.in_channels = insize[0]
        self.outsize = outsize

        self.cnn = nn.ModuleList()
        for i in range(n_blocks):
            self.cnn.append(ResnetBlock(in_channels=self.in_channels * 2**(i), out_channels=self.in_channels * 2**(i+1), pool=pool))
        self.cnn = torch.nn.Sequential(*self.cnn)

        with torch.no_grad():
            self.mlp_insize = self.cnn(torch.zeros(1, *insize)).flatten(start_dim=-3).shape[-1]

        self.mlp = MLP(self.mlp_insize, outsize, mlp_hiddens, hidden_activation, dropout, device)

    def forward(self, x):
        cnn_out = self.cnn.forward(x)
        mlp_in = cnn_out.flatten(start_dim=-3)
        out = self.mlp.forward(mlp_in)
        return out

class ResnetBlock(nn.Module):
    """
    A ResNet-style block that does VGG + residual. Like the VGG-style block, output size is half of input size.
    """
    def __init__(self, in_channels, out_channels, pool=2):
        super(ResnetBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=pool)
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        _x = self.batch_norm(x)
        _x = self.conv1(_x)
        _x = self.activation(_x)
        _x = self.conv2(_x)
        res = self.projection(x)
        _x = self.activation(_x + res)
        _x = self.max_pool(_x)
        return _x

class VGGBlock(nn.Module):
    """
    A VGG-style block that batch-norms, does two 3x3 convs+relus and max-pools
    Doubles n_channels, halves width and height
    """
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x

if __name__ == '__main__':
    envmap = torch.load('12x12_map_no_obstacles.pt')

    import pdb;pdb.set_trace()
    inp = torch.tensor(envmap.mapRGB).unsqueeze(0) / 255.
    inp = inp.permute(0, 3, 1, 2)

    vgg = VGGBlock(inp.shape[1], 1)
    res = ResnetBlock(inp.shape[1], 1)

    yvgg = vgg(inp)
    yres = res(inp)

    print(yvgg)
    print(yres)
