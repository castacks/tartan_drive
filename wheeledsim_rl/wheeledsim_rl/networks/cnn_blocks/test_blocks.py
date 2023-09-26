"""
Small script for testing cnn blocks on mnist
"""

import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from airsim_mbrl.networks.cnn_blocks.cnn_blocks import VGGBlock, ResnetBlock, ResnetCNN

if __name__ == '__main__':
    mnist = MNIST(root='~/Desktop/datasets', transform=transforms.Compose([transforms.Resize([80, 80]), transforms.ToTensor()]))
    dl = DataLoader(mnist, batch_size = 32, shuffle=True)

    """
    cnn = torch.nn.Sequential(
        VGGBlock(1, 2),
        VGGBlock(2, 4),
        VGGBlock(4, 8),
        VGGBlock(8, 16),
        torch.nn.Flatten(),
        torch.nn.Linear(16*4, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 10),
    )

    cnn = torch.nn.Sequential(
        ResnetBlock(1, 2),
        ResnetBlock(2, 4),
        ResnetBlock(4, 8),
        ResnetBlock(8, 16),
        torch.nn.Flatten(),
        torch.nn.Linear(16*4, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 10),
    )
    print(cnn)
    """

    cnn = ResnetCNN(insize=(1, 80, 80), outsize=10, n_blocks=5, mlp_hiddens=[32, ])

    print(cnn)

    opt = torch.optim.Adam(cnn.parameters())

    epochs = 100
    for i in range(epochs):
        print('___________EPOCH {}__________'.format(i + 1))
        acc_buf = 0
        n = 0
        for ii, (x, y) in enumerate(iter(dl)):
            preds = cnn(x)
            loss = torch.nn.functional.cross_entropy(preds, y)
            argmax_preds = preds.detach().argmax(dim=1)
            acc = (argmax_preds == y).sum().float() / y.shape[0]
            n_new = n + y.shape[0]
            acc_buf = (n/n_new) * acc_buf + (y.shape[0]/n_new) * acc

            opt.zero_grad()
            loss.backward()
            opt.step()

            print('LOSS = {:.6f}, ACC = {:.6f}'.format(loss, acc_buf), end='\r')
            if ii > 100:
                break
        
        print('LOSS = {:.6f}, ACC = {:.6f}'.format(loss, acc_buf))

"""
    batch = next(iter(dl))
    for x, y in zip(batch[0], batch[1]):
        print(x.shape)
        plt.imshow(x[0])
        plt.title(y)
        plt.show()
"""
