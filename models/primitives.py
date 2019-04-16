import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

from models.functions import Feedforward, InformationBottleneck, GaussianParams, DeterministicBottleneck


class GaussianPolicy(nn.Module):
    def __init__(self):
        super(GaussianPolicy, self).__init__()
        self.is_disc_action = False

    def forward(self, x):
        raise NotImplementedError

    def select_action(self, state, deterministic=False):
        mu, std, kl = self.forward(state)
        if deterministic:
            return mu
        else:
            dist = MultivariateNormal(loc=mu, scale_tril=torch.diag_embed(std))
            action = dist.sample()
            return action

    def get_log_prob(self, state, action):
        mu, std, kl = self.forward(state)
        bsize = mu.size(0)
        dist = MultivariateNormal(loc=mu, scale_tril=torch.diag_embed(std))
        log_prob = dist.log_prob(action).view(bsize, 1)
        entropy = dist.entropy().view(bsize, 1)
        return {'log_prob': log_prob, 'kl': kl, 'entropy': entropy}

    def post_process(self, action):
        return action

class WeightNetwork(GaussianPolicy):
    def __init__(self, state_dim, goal_dim, encoder_dims, bottleneck_dim, decoder_dims, device, vib=False):
        super(WeightNetwork, self).__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.encoder_dims = encoder_dims
        self.bottleneck_dim = bottleneck_dim
        self.decoder_dims = decoder_dims

        self.state_trunk = Feedforward([state_dim] + encoder_dims, out_act=F.relu)
        bottleneck = InformationBottleneck if vib else DeterministicBottleneck
        self.bottleneck = bottleneck(encoder_dims[-1], bottleneck_dim, device=device)
        self.goal_trunk = Feedforward([goal_dim] + encoder_dims + [bottleneck_dim])
        self.decoder = Feedforward([bottleneck_dim*2]+decoder_dims)  # TODO put the sigmoid outside!

        self.std = 0.2

    def forward(self, state):
        x, g = state[..., :self.state_dim], state[...,self.state_dim:]
        z, kl = self.bottleneck(self.state_trunk(x))
        goal_embedding = self.goal_trunk(g)
        h = torch.cat((z, goal_embedding), dim=1)
        weights = self.decoder(h)
        std = self.std*torch.ones(*weights.size()).to(self.get_device())
        return weights, std, kl

    def get_device(self):
        if next(self.parameters()).is_cuda:
            return self.device
        else:
            return torch.device('cpu')

class PrimitivePolicy(GaussianPolicy):
    """

    """
    def __init__(self, encoder, bottleneck_dim, decoder_dims, device, id, fixed_var=False, vib=False):
        super(PrimitivePolicy, self).__init__()
        self.outdim = decoder_dims[-1]
        self.encoder = encoder
        bottleneck = InformationBottleneck if vib else DeterministicBottleneck
        self.bottleneck = bottleneck(encoder.dims[-1], bottleneck_dim, device=device)
        self.decoder = nn.Linear(bottleneck_dim, decoder_dims[0])
        self.parameter_producer = GaussianParams(decoder_dims[0], decoder_dims[1], custom_init=True, fixed_var=fixed_var)
        self.name = 'primitive-{}'.format(id)

    def forward(self, x):
        x = self.encoder(x)
        z, kl = self.bottleneck(x)
        h = F.relu(self.decoder(z))
        mu, logstd = self.parameter_producer(h)
        return mu, torch.exp(logstd), kl

class CompositePolicy(GaussianPolicy):
    def __init__(self, weight_network, primitives, obs_dim, freeze_primitives=False):
        super(CompositePolicy, self).__init__()
        self.primitives = primitives
        self.weight_network = weight_network
        self.k = len(self.primitives)
        self.outdim = self.primitives[0].outdim
        self.name = 'composite'
        self.obs_dim = obs_dim
        self.freeze_primitives = freeze_primitives

    def forward(self, state):
        obs, goal = state[..., :self.obs_dim], state[...,self.obs_dim:]
        bsize = state.size(0)
        ##############################
        weights, weights_std, weights_kl = self.weight_network(state)
        weights = F.sigmoid(weights)
        broadcasted_weights = weights.view(bsize, self.k, 1)
        ##############################
        if self.freeze_primitives:
            with torch.no_grad():
                mus, stds, kls = zip(*[p(obs) for p in self.primitives])  # list of length k of (bsize, adim)
        else:
            mus, stds, kls = zip(*[p(obs) for p in self.primitives])  # list of length k of (bsize, adim)
        mus = torch.stack(mus, dim=1)  # (bsize, k, outdim)
        stds = torch.stack(stds, dim=1)  # (bsize, k, outdim)
        kls = torch.stack(kls, dim=1)  # (bsize)
        ##############################
        weights_over_variance = broadcasted_weights/(stds*stds)  # (bsize, k, zdim)
        inverse_variance = torch.sum(weights_over_variance, dim=1)  # (bsize, zdim)
        ##############################
        composite_std = 1.0/torch.sqrt(inverse_variance)
        composite_logstd = -0.5 * torch.log(inverse_variance)
        ##############################
        weighted_mus = weights_over_variance * mus
        composite_mu = torch.sum(weighted_mus, dim=1)/inverse_variance  # (bsize, zdim)
        ##############################
        composite_kl = kls.sum()
        return composite_mu, composite_std, kls

class CompositeTransferPolicy(CompositePolicy):
    def __init__(self, weight_network, primitives, obs_dim):
        super(CompositeTransferPolicy, self).__init__(weight_network, primitives, obs_dim, freeze_primitives=True)

    def forward(self, state):
        weights = self.weight_network.select_action(state)
        return weights

    def post_process(self, weights):
        weights = F.sigmoid(weights)
        broadcasted_weights = weights.view(bsize, self.k, 1)
        with torch.no_grad():
            mus, stds, kls = zip(*[p(state) for p in self.primitives])  # list of length k of (bsize, adim)
        mus = torch.stack(mus, dim=1)  # (bsize, k, outdim)
        stds = torch.stack(stds, dim=1)  # (bsize, k, outdim)
        kls = torch.stack(kls, dim=1)  # (bsize)
        ##############################
        weights_over_variance = broadcasted_weights/(stds*stds)  # (bsize, k, zdim)
        inverse_variance = torch.sum(weights_over_variance, dim=1)  # (bsize, zdim)
        ##############################
        weighted_mus = weights_over_variance * mus
        composite_mu = torch.sum(weighted_mus, dim=1)/inverse_variance  # (bsize, zdim)
        return composite_mu

def debug2():
    state_dim = 60
    action_dim = 12
    encoder = Feedforward([state_dim, 512, 256], out_act=F.relu)
    policy = PrimitiveVIBPolicy(encoder=encoder, ib_dims=[256, 128], hdim=256, outdim=action_dim)
    print(policy)

def debug():
    BSIZE = 5
    K = 4
    ZDIM = 3

    # list of length k of (bsize, zdim)
    mus = [torch.rand(BSIZE, ZDIM) for k in range(K)]
    logstds = [torch.rand(BSIZE, ZDIM) for k in range(K)]
    weights = torch.rand(BSIZE, K)
    mus = torch.stack(mus, dim=1)  # (bsize, k, zdim)
    logstds = torch.stack(logstds, dim=1)  # (bsize, k, zdim)
    stds = torch.exp(logstds)  # (bsize, k, zdim)
    bsize, k, zdim = stds.size()
    broadcasted_weights = weights.view(bsize, k, 1)
    print('mus', mus.size())
    print('logstds', logstds.size())
    print('stds', stds.size())
    print('weights', weights.size())
    print('broadcasted_weights', broadcasted_weights.size())
    ##############################
    weights_over_variance = broadcasted_weights/(stds*stds)  # (bsize, k, zdim)
    inverse_variance = torch.sum(weights_over_variance, dim=1)  # (bsize, zdim)
    print('weights_over_variance', weights_over_variance.size())
    print('inverse_variance', inverse_variance.size())
    ##############################
    composite_std = 1.0/torch.sqrt(inverse_variance)  # (bsize, zdim)
    composite_logstd = -0.5 * torch.log(inverse_variance)  # (bsize, zdim)
    print('composite_std', composite_std.size())
    print('composite_logstd', composite_logstd.size())
    ##############################
    weighted_mus = weights_over_variance * mus  # (bsize, k, zsize)
    composite_mu = torch.sum(weighted_mus, dim=1)/inverse_variance  # (bsize, zdim)
    print('weighted_mus', weighted_mus.size())
    print('composite_mu', composite_mu.size())
    ##############################

    """
    mus torch.Size([5, 4, 3])
    logstds torch.Size([5, 4, 3])
    stds torch.Size([5, 4, 3])
    weights torch.Size([5, 4])
    broadcasted_weights torch.Size([5, 4, 1])
    weights_over_variance torch.Size([5, 4, 3])
    inverse_variance torch.Size([5, 3])
    composite_std torch.Size([5, 3])
    composite_logstd torch.Size([5, 3])
    weighted_mus torch.Size([5, 4, 3])
    composite_mu torch.Size([5, 3])
    """


if __name__ == '__main__':
    # debug()

    debug2()
