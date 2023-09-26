import torch

from torch import nn, optim, distributions

class DownsampleBlock(nn.Module):
    """
    Generic CNN to downsample an image
    TODO: Figure out if VGG/Resnet blocks matter.
    """
    def __init__(self, in_channels, out_channels, pool=2, activation=nn.ReLU, device='cpu'):
        super(DownsampleBlock, self).__init__()
        #For now, let's leave out batchnorm.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = activation()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.device = device

    def forward(self, x):
        extra_dims = x.shape[:-3]
        _x = self.conv(x.flatten(end_dim=-4))
        _x = _x.view(extra_dims + _x.shape[-3:])
        _x = self.activation(_x)
        _x = self.pool(_x)
        return _x

    def to(self, device):
        self.device = device
        self.conv = self.conv.to(device)
        self.activation = self.activation.to(device)
        self.pool = self.pool.to(device)
        return self

class UpsampleBlock(nn.Module):
    """
    Simple generic upsampling block
    TODO: Same as DownsampleBlock.
    """
    def __init__(self, in_channels, out_channels, scale=2, out_shape=None, activation=nn.ReLU, device='cpu'):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = activation()
        self.scale = scale
        self.out_shape = out_shape
        self.device = device

    def forward(self, x):
        extra_dims = x.shape[:-3]
        _x = x.flatten(end_dim=-4)

        if self.out_shape is None:
            _x = nn.functional.interpolate(_x, scale_factor=self.scale)
        else:
            _x = nn.functional.interpolate(_x, size=self.out_shape)

        _x = self.conv(_x)
        _x = _x.view(extra_dims + _x.shape[-3:])
        _x = self.activation(_x)
        return _x

    def to(self, device):
        self.device = device
        self.conv = self.conv.to(device)
        self.activation = self.activation.to(device)
        return self

class CNNEncoder(nn.Module):
    """
    CNN-based encoder to get latents from images.
    """
    def __init__(self, cnn_insize=[1, 64, 64], latent_size=16, channels = [2, 4, 8, 16], pool=None, activation=nn.ReLU, device='cpu'):
        """
        Args:
            insize: The expected size of the input image.
            latent_size: The number of latent dimensions.
            channels: A list of intermediate channel sizes corresponding to downsample blocks.
            activation: List of activations to use.
        """
        super(CNNEncoder, self).__init__()
        self.insize = cnn_insize
        self.latent_size = latent_size
        self.activation = activation
        self.blocks = nn.ModuleList()
        self.channels = [self.insize[0]] + channels
        self.pool = [2] * (len(self.channels)-1) if pool is None else pool
        for i in range(len(self.channels)-1):
            di = self.channels[i]
            do = self.channels[i+1]
            p = self.pool[i]
            self.blocks.append(DownsampleBlock(di, do, p, self.activation))
        self.blocks = torch.nn.Sequential(*self.blocks)
        self.flatten = nn.Flatten(start_dim=-3)

        with torch.no_grad():
            self.mlp_insize = self.flatten(self.blocks(torch.zeros(1, *self.insize))).shape[-1]

        self.linear = nn.Linear(self.mlp_insize, 2*self.latent_size)
        self.device = device

    def forward(self, x, return_dist=False):
        _x = self.blocks(x)
        _x = self.linear(self.flatten(_x))

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
        self.blocks = self.blocks.to(device)
        self.linear = self.linear.to(device)
        return self

class CNNDecoder(nn.Module):
    """
    CNN-based decoder to get images from latents.
    """
    def __init__(self, latent_size=16, cnn_outsize=[1, 64, 64], channels = [32, 16, 8, 4], scale=[4, 2, 2], activation=nn.ReLU, final_activation=nn.Identity, device='cpu'):
        """
        Args:
            latent_size: The number of latent dimensions
            cnn_outsize: The size of the image produced
            channels: The intermediate number of channels after each block. Note that unlike the encoder, we need one more channel.
            scale: The amount to upscale each image by
            activation: The activations for intermediate blocks
            final_activation: The activation for the last block. Should probably be sigmoid to enforce [0, 1] outputs for images, else Identity
        """
        super(CNNDecoder, self).__init__()
        self.outsize = cnn_outsize
        self.latent_size = latent_size
        self.activation = activation
        self.final_activation = final_activation
        self.channels = channels
        self.scale = scale
        self.blocks = nn.ModuleList()
        for i in range(len(self.channels)-1):
            di = self.channels[i]
            do = self.channels[i+1]
            s = self.scale[i]
            self.blocks.append(UpsampleBlock(di, do, s, activation=self.activation))
        #Final block
        self.blocks.append(UpsampleBlock(self.channels[-1], self.outsize[0], out_shape=self.outsize[1:], activation=self.final_activation))
        self.blocks = nn.Sequential(*self.blocks)
        self.linear = nn.Linear(self.latent_size, self.channels[0])

        self.device = device

    def forward(self, x, return_dist=False):
        _x = self.linear(x)
        _x = _x.unsqueeze(-1).unsqueeze(-1) #Pad two image dims.
        _x = self.blocks(_x)

        return _x

    def to(self, device):
        self.device = device
        self.blocks = self.blocks.to(device)
        self.linear = self.linear.to(device)
        return self

class CNNAE(nn.Module):
    """
    VAE is giving me sone difficulty/not neccesary for latent space losses. So just use an AE
    """
    def __init__(self, encoder, decoder, device='cpu'):
        assert encoder.insize == decoder.outsize, "Expects encoder/decoder to work in same image shapes. Encoder images are {}, decoder are {}".format(encoder.insize, decoder.outsize)
        assert encoder.latent_size == decoder.latent_size, "Expects encoder/decoder to have same latent size. Got {} for encoder, {} for decoder.".format(encoder.latent_size, decoder.latent_size)

        super(CNNVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = encoder.latent_size
        self.device = device

    def forward(self, x):
        e_out = self.encoder(x)
        decoder_out = self.decoder(e_out)

        return decoder_out

    def to(self, device):
        self.device = device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        return self

class CNNVAE(nn.Module):
    """
    CNN-based VAE
    """
    def __init__(self, encoder, decoder, device='cpu'):
        assert encoder.insize == decoder.outsize, "Expects encoder/decoder to work in same image shapes. Encoder images are {}, decoder are {}".format(encoder.insize, decoder.outsize)
        assert encoder.latent_size == decoder.latent_size, "Expects encoder/decoder to have same latent size. Got {} for encoder, {} for decoder.".format(encoder.latent_size, decoder.latent_size)

        super(CNNVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = encoder.latent_size
        self.device = device

    def forward(self, x, return_dist=False):
        e_out = self.encoder(x)
        means = torch.index_select(e_out, -1, torch.arange(self.latent_size))
        stds = torch.index_select(e_out, -1, torch.arange(self.latent_size)+self.latent_size).exp()
        dist = distributions.Normal(loc=means, scale=stds)
        samples = dist.rsample()
        decoder_out = self.decoder(samples)

        if return_dist:
            return decoder_out, dist
        else:
            return decoder_out

    def to(self, device):
        self.device = device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        return self

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img_topic='front_camera'
    
    buf = torch.load('buffer.pt')
    batch = buf.sample(5, N=3)
    print(len(buf))
    img = batch['observation'][img_topic][:, 0]

    encoder = CNNEncoder(cnn_insize=img.shape[1:], latent_size=8)
    decoder = CNNDecoder(cnn_outsize=img.shape[1:], channels=[64, 32, 16, 8], latent_size=8)
    vae = CNNVAE(encoder, decoder)

    opt = optim.Adam(vae.parameters())

    itrs = 10000
    beta = 1e-4

    for itr in range(itrs):
        batch = buf.sample(256, N=1)
        img = batch['observation'][img_topic][:, 0]
        img_r, dist = vae.forward(img, return_dist=True)
        means = dist.mean
        stds = dist.scale

        reconstruction_loss = (img - img_r).pow(2).mean()
        kl_loss = -0.5 * (1 + stds.pow(2).log() - means.pow(2) - stds.pow(2)).sum(dim=-1).mean()

        loss = reconstruction_loss + beta*kl_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (itr % 100) == 0:
            print("Itr {}/{}: R_loss = {:.6f}, KL_loss = {:.6f}".format(itr, itrs, reconstruction_loss.detach().item(), kl_loss.detach().item()))
            print("Sample Latent:")
            print(means.detach()[0])
            print(stds.detach()[0])

    torch.save(vae, 'vae.pt')
    vae = torch.load('vae.pt')

    import pdb;pdb.set_trace()
    for i in range(10):
        batch = buf.sample(1, N=1)
        img = batch['observation'][img_topic][:, 0]
        with torch.no_grad():
            img_r = vae(img)
        r = (img - img_r).pow(2).mean()
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#        axs[0].imshow(img[0, 0], cmap='gray')
#        axs[1].imshow(img_r[0, 0], cmap='gray')
        axs[0].imshow(img[0, :3].permute(1, 2, 0), cmap='gray')
        axs[1].imshow(img_r[0, :3].permute(1, 2, 0), cmap='gray')
        plt.title("Reconstruction Loss = {:.6f}".format(r.item()))
        plt.show()
