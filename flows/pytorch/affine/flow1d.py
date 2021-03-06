from torch import nn, squeeze
from torch.nn import functional as F
import torch
from torch.distributions import Normal, Laplace
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import timeit
import seaborn as sns
import numpy as np
torch.manual_seed(0)
np.random.seed(0)

# Dataset


def get_laplace(n_samples):
    laplace_dist = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
    data = laplace_dist.sample([1000, ])
    return data


def get_gaussian(n_samples):
    base_mu, base_cov = torch.zeros(1), torch.eye(1)
    base_dist = Normal(base_mu, base_cov)
    z = base_dist.rsample(sample_shape=(1000,))
    return z

# UTITILITY FUNCTIONS


# FLOW LAYER


class TransformationLayer(nn.Module):
    """Create forward and inverse of the normalizing network"""

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.rand(1))
        self.beta = nn.Parameter(torch.rand(1))

    def forward(self, x):
        _z = torch.exp(self.alpha)*x+self.beta
        log_det = -self.alpha
        return _z, log_det

    def inverse(self, z):
        _x = (z-self.beta)*torch.exp(-self.alpha)
        return _x


# FLOW MODEL
class NormalizingFlowModel(nn.Module):

    """Wrapper provide the general skeleton/structure for Normalizing Flow models with Gaussian Prior"""

    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        log_det = torch.zeros(x.shape[0],)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z):
        for flow in self.flows[::-1]:
            z = flow.inverse(z)
        x = z
        return x

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,)).squeeze()
        x = self.inverse(z)
        return x

# TRAIN LOOP

def train(model, data, epochs, optim):
    losses = []
    start = timeit.default_timer()
    for _ in tqdm(range(epochs)):
        optim.zero_grad()
        z, prior_logprob, log_det = model(data)
        loss = (-prior_logprob - log_det).mean()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    stop = timeit.default_timer()
    print(z)
    print('Time: ', stop - start)
    plt.figure(figsize=(16, 6))
    plt.subplot(2, 3, 1)
    sns.distplot(data)
    plt.title("Training data distribution")
    plt.subplot(2, 3, 2)
    sns.distplot(z.data)
    plt.title("Latent distribution")
    plt.subplot(2, 3, 3)
    samples = model.sample(len(z.data)).data
    sns.distplot(samples)
    plt.title("Generated data distribution")
    plt.subplot(2, 3, 4)
    plt.plot(data)
    plt.title('1D training data')
    plt.subplot(2, 3, 5)
    plt.plot(z.data)
    plt.title('1D latent data')
    plt.subplot(2, 3, 6)
    plt.plot(samples)
    plt.title('1D generated data')
    plt.show()

    return losses


if __name__ == '__main__':

    # Dataset
    x = get_laplace(1000)
    z = get_gaussian(1000)
    # Number of flows
    num_flows = 10
    flows = [TransformationLayer() for _ in range(num_flows)]
    # Source distribution
    prior = Normal(torch.zeros(1), torch.eye(1))
    # Model instance
    model = NormalizingFlowModel(prior, flows)
    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = train(model, x, 5000, optim)
    plt.plot(losses)
