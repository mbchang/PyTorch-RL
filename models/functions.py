import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

class Feedforward(nn.Module):
    """
        x --> h
    """
    def __init__(self, dims, out_act=None):
        super(Feedforward, self).__init__()
        assert len(dims) >= 2
        self.dims = dims
        self.act = F.relu
        self.out_act = out_act
        self.layers = nn.ModuleList()
        for i in range(len(self.dims)-1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers)-1:
                if self.out_act:
                    x = self.out_act(layer(x))
                else:
                    x = layer(x)
            else:
                x = self.act(layer(x))
        return x

class InformationBottleneck(nn.Module):
    def __init__(self, hdim, zdim, device):
        super(InformationBottleneck, self).__init__()
        self.hdim = hdim
        self.zdim = zdim
        self.device = device
        self.parameter_producer = GaussianParams(hdim=self.hdim, zdim=self.zdim)

    def standard_normal_prior(self):
        prior_mu = torch.zeros(self.zdim).to(self.get_device())
        prior_std = torch.ones(self.zdim).to(self.get_device())
        prior = MultivariateNormal(loc=prior_mu, scale_tril=torch.diag(prior_std))
        return prior

    def kl_standard_normal(self, dist):
        prior = self.standard_normal_prior()
        kl = torch.distributions.kl.kl_divergence(p=dist, q=prior)
        return kl

    def forward(self, x):
        mu, logstd = self.parameter_producer(x)
        dist = MultivariateNormal(loc=mu, scale_tril=torch.diag_embed(torch.exp(logstd)))
        z = dist.rsample()
        kl = self.kl_standard_normal(dist)
        return z, kl

    def get_device(self):
        if next(self.parameters()).is_cuda:
            return self.device
        else:
            return torch.device('cpu')

class GaussianParams(nn.Module):
    """
        h --> z
    """
    def __init__(self, hdim, zdim):
        super(GaussianParams, self).__init__()
        self.mu = nn.Linear(hdim, zdim)
        self.logstd = nn.Linear(hdim, zdim)

        # TODO: you can initialize them to standard normal.

    def forward(self, x):
        mu = self.mu(x)
        logstd = self.logstd(x)
        return mu, logstd#torch.exp(logstd)