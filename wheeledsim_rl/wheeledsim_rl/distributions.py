import torch
from torch import distributions

class StraightThroughOneHotCategorical(distributions.OneHotCategorical):
    """
    Implementation of OneHotCategorical with rsample.
    """
    def rsample(self):
        """
        From Bengio et al 2013
        """
        sample = self.sample()
        return sample + self.probs - self.probs.detach()

if __name__ == '__main__':
    from torch import optim
    logits = torch.rand(8, 8, requires_grad=True)
    dist = StraightThroughOneHotCategorical(logits)
    x = dist.sample()
    print(x)
    x2 = dist.rsample()
    print(x2)
