import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.1)

class Feedforward(nn.Module):
    """
        x --> h
    """
    def __init__(self, dims, out_act=None, custom_init=False):
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

def standard_normal_prior(zdim, device):
    prior_mu = torch.zeros(zdim).to(device)
    prior_std = torch.ones(zdim).to(device)
    prior = MultivariateNormal(loc=prior_mu, scale_tril=torch.diag(prior_std))
    return prior

def kl_standard_normal(dist, device):
    zdim = dist.event_shape
    prior = standard_normal_prior(zdim, device)
    kl = torch.distributions.kl.kl_divergence(p=dist, q=prior)
    return kl

def reparametrize(mu, std, device):
    bsize = mu.size(0)
    dist = MultivariateNormal(loc=mu, scale_tril=torch.diag_embed(std))
    z = dist.rsample()
    kl = kl_standard_normal(dist, device).view(bsize, 1)
    return z, kl

class InformationBottleneck(nn.Module):
    def __init__(self, hdim, zdim, device):
        super(InformationBottleneck, self).__init__()
        self.hdim = hdim
        self.zdim = zdim
        self.device = device
        self.parameter_producer = GaussianParams(hdim=self.hdim, zdim=self.zdim)

    def forward(self, x):
        bsize = x.size(0)
        mu, logstd = self.parameter_producer(x)
        z, kl = reparametrize(mu, torch.exp(logstd), device=self.get_device())
        return z, kl

    def get_device(self):
        if next(self.parameters()).is_cuda:
            return self.device
        else:
            return torch.device('cpu')

class DeterministicBottleneck(nn.Module):
    def __init__(self, hdim, zdim, device):
        super(DeterministicBottleneck, self).__init__()
        self.hdim = hdim
        self.zdim = zdim
        self.device = device
        self.network = nn.Linear(hdim, zdim)

    def forward(self, x):
        z = self.network(x)
        kl = torch.zeros(self.zdim).to(self.get_device())  # dummy kl
        return z, kl

    def get_device(self):
        if next(self.parameters()).is_cuda:
            return self.device
        else:
            return torch.device('cpu')

class ConstantLogStd(nn.Module):
    def __init__(self, zdim, std):
        self.logstd = np.log(std)
        self.zdim = zdim

    def forward(self, x):
        return self.logstd * torch.ones((x.size(0), self.zdim))

class GaussianParams(nn.Module):
    """
        h --> z
    """
    def __init__(self, hdim, zdim, device, custom_init=False, fixed_std=None):
        super(GaussianParams, self).__init__()
        self.device=device
        self.hdim = hdim
        self.zdim = zdim
        self.mu = nn.Linear(hdim, zdim)
        self.fixed_std = fixed_std

        # if not self.fixed_std:
        if self.fixed_std is None:
            self.logstd = nn.Linear(hdim, zdim)

            if custom_init:
                self.logstd.weight.data.mul_(0.1)  # Can take out
                nn.init.constant_(self.logstd.bias, np.log(0.1))

        if custom_init:
            # nn.init.uniform_(self.mu.weight)
            self.mu.weight.data.mul_(0.1)
            nn.init.constant_(self.mu.bias, 0.0)

    def forward(self, x):
        mu = self.mu(x)
        if self.fixed_std is not None:
            logstd = np.log(self.fixed_std) * torch.ones((x.size(0), self.zdim)).to(self.get_device())
        else:
            logstd = self.logstd(x)
        return mu, logstd

    def get_device(self):
        if next(self.parameters()).is_cuda:
            return self.device
        else:
            return torch.device('cpu')

