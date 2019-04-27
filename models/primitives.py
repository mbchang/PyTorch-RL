import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

from models.functions import Feedforward, InformationBottleneck, GaussianParams, DeterministicBottleneck, reparametrize


class GaussianPolicy(nn.Module):
    def __init__(self, device):
        super(GaussianPolicy, self).__init__()
        self.is_disc_action = False
        self.device=device

    def forward(self, x):
        raise NotImplementedError

    def select_action(self, state, deterministic=False):
        mu, std, kl, info = self.forward(state)
        if deterministic:
            return mu
        else:
            dist = MultivariateNormal(loc=mu, scale_tril=torch.diag_embed(std))
            action = dist.sample()
            return action

    def get_log_prob(self, state, action):
        mu, std, kl, info = self.forward(state)
        bsize = mu.size(0)
        dist = MultivariateNormal(loc=mu, scale_tril=torch.diag_embed(std))
        log_prob = dist.log_prob(action).view(bsize, 1)
        entropy = dist.entropy().view(bsize, 1)
        info.update({'log_prob': log_prob, 'kl': kl, 'entropy': entropy})
        return info

    def post_process(self, state, action):
        return action

    def get_device(self):
        if next(self.parameters()).is_cuda:
            return self.device
        else:
            return torch.device('cpu')

class StochasticGaussianPolicy(GaussianPolicy):
    def __init__(self, device):
        super(StochasticGaussianPolicy, self).__init__(device)
        self.is_stochastic = True

class DeterministicGaussianPolicy(GaussianPolicy):
    def __init__(self, device):
        super(DeterministicGaussianPolicy, self).__init__(device)
        self.is_stochastic = False

class WeightNetwork(GaussianPolicy):
    def __init__(self, state_dim, goal_dim, encoder_dims, bottleneck_dim, decoder_dims, device, vib=False, fixed_std=None):
        super(WeightNetwork, self).__init__(device)
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.encoder_dims = encoder_dims
        self.bottleneck_dim = bottleneck_dim
        self.decoder_dims = decoder_dims
        self.out_dim = self.decoder_dims[-1]

        self.state_trunk = Feedforward([state_dim] + encoder_dims, out_act=F.relu)
        bottleneck = InformationBottleneck if vib else DeterministicBottleneck
        self.bottleneck = bottleneck(encoder_dims[-1], bottleneck_dim, device=device)
        self.goal_trunk = Feedforward([goal_dim] + encoder_dims + [bottleneck_dim])
        self.decoder = Feedforward([bottleneck_dim*2, decoder_dims[0]])
        self.parameter_producer = GaussianParams(decoder_dims[0], decoder_dims[1], device=device, custom_init=True, fixed_std=fixed_std)

    def forward(self, state):
        x, g, noise = state[..., :self.state_dim], state[...,self.state_dim:self.state_dim+self.goal_dim], state[...,self.state_dim+self.goal_dim:]
        z, kl = self.bottleneck(self.state_trunk(x))
        goal_embedding = self.goal_trunk(g)
        h = torch.cat((z, goal_embedding), dim=1)
        h = self.decoder(h)
        weights, logstd = self.parameter_producer(h)
        std = torch.exp(logstd)  # this should around 0.2 or 0.3
        return weights, std, kl, {'weight_std': std}

class WeightNetworkNoState(GaussianPolicy):
    def __init__(self, state_dim, goal_dim, encoder_dims, bottleneck_dim, decoder_dims, device, vib=False, fixed_std=None):
        super(WeightNetworkNoState, self).__init__(device)
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.encoder_dims = encoder_dims
        self.bottleneck_dim = bottleneck_dim
        self.decoder_dims = decoder_dims
        self.out_dim = self.decoder_dims[-1]

        self.state_trunk = Feedforward([state_dim] + encoder_dims, out_act=F.relu)
        bottleneck = InformationBottleneck if vib else DeterministicBottleneck
        self.bottleneck = bottleneck(encoder_dims[-1], bottleneck_dim, device=device)
        self.goal_trunk = Feedforward([goal_dim] + encoder_dims + [bottleneck_dim])
        self.decoder = Feedforward([bottleneck_dim, decoder_dims[0]])
        # self.decoder = Feedforward([bottleneck_dim*2, decoder_dims[0]])
        self.parameter_producer = GaussianParams(decoder_dims[0], decoder_dims[1], device=device, custom_init=True, fixed_std=fixed_std)

    def forward(self, state):
        x, g, noise = state[..., :self.state_dim], state[...,self.state_dim:self.state_dim+self.goal_dim], state[...,self.state_dim+self.goal_dim:]
        z, kl = self.bottleneck(self.state_trunk(x))
        goal_embedding = self.goal_trunk(g)
        # h = torch.cat((z, goal_embedding), dim=1)
        h = self.decoder(goal_embedding)
        weights, logstd = self.parameter_producer(h)
        std = torch.exp(logstd)  # this should around 0.2 or 0.3
        return weights, std, kl, {'weight_std': std}

class GoalEmbedder(GaussianPolicy):
    def __init__(self, dims, obs_dim, goal_dim, device, fixed_std=None, ignore_obs=True):
        super(GoalEmbedder, self).__init__(device)
        assert len(dims) >= 3
        self.dims = dims
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.ignore_obs = ignore_obs
        self.network = Feedforward(dims=dims[:-2], out_act=F.relu)
        self.bottleneck = DeterministicBottleneck(self.dims[-3], self.dims[-2], device=device)
        self.parameter_producer = GaussianParams(self.dims[-2], self.dims[-1], device=device, custom_init=True, fixed_std=fixed_std)
        self.out_dim = self.dims[-1]

    def forward(self, state):
        obs, goal = state[..., :self.obs_dim], state[...,self.obs_dim:self.obs_dim+self.goal_dim]
        if self.ignore_obs:
            h = self.network(goal)
        else:
            h = self.network(state)
        # h = self.network(goal) if self.ignore_obs else self.network(state)
        z, kl = self.bottleneck(h)  # dummy kl
        mu, logstd = self.parameter_producer(z)
        return mu, torch.exp(logstd), kl, {}

class PrimitivePolicy(DeterministicGaussianPolicy):
    def __init__(self, encoder, bottleneck_dim, decoder_dims, device, id, fixed_std=None, vib=False):
        super(PrimitivePolicy, self).__init__(device)
        self.outdim = decoder_dims[-1]
        self.encoder = encoder
        bottleneck = InformationBottleneck if vib else DeterministicBottleneck
        self.bottleneck = bottleneck(encoder.dims[-1], bottleneck_dim, device=device)
        self.decoder = nn.Linear(bottleneck_dim, decoder_dims[0])
        self.parameter_producer = GaussianParams(decoder_dims[0], decoder_dims[1], device=device, custom_init=True, fixed_std=fixed_std)
        self.name = 'primitive-{}'.format(id)

    def forward(self, x):
        x = self.encoder(x)
        z, kl = self.bottleneck(x)
        h = F.relu(self.decoder(z))
        mu, logstd = self.parameter_producer(h)
        return mu, torch.exp(logstd), kl, {}

class CompositePolicy(StochasticGaussianPolicy):
    def __init__(self, weight_network, primitives, obs_dim, device, freeze_primitives=False):
        super(CompositePolicy, self).__init__(device)
        self.primitives = primitives
        self.weight_network = weight_network
        self.k = len(self.primitives)
        self.outdim = self.primitives[0].outdim
        self.name = 'composite'
        self.obs_dim = obs_dim
        self.freeze_primitives = freeze_primitives
        self.zdim = self.weight_network.out_dim

    def get_composite_mu(self, mus, weights_over_variance, inverse_variance):
        weighted_mus = weights_over_variance * mus
        composite_mu = torch.sum(weighted_mus, dim=1)/inverse_variance  # (bsize, zdim)
        return composite_mu

    def execute_primitives(self, obs, no_grad):
        if no_grad:
            with torch.no_grad():
                mus, stds, kls, infos = zip(*[p(obs) for p in self.primitives])  # list of length k of (bsize, adim)
        else:
            mus, stds, kls, infos = zip(*[p(obs) for p in self.primitives])  # list of length k of (bsize, adim)
        mus = torch.stack(mus, dim=1)  # (bsize, k, outdim)
        stds = torch.stack(stds, dim=1)  # (bsize, k, outdim)
        kls = torch.stack(kls, dim=1)  # (bsize)
        return mus, stds, kls

    def get_composite_parameters(self, mus, stds, kls, weights):
        weights_over_variance = weights/(stds*stds)  # (bsize, k, zdim)
        inverse_variance = torch.sum(weights_over_variance, dim=1)  # (bsize, zdim)
        ##############################
        composite_std = 1.0/torch.sqrt(inverse_variance)
        composite_logstd = -0.5 * torch.log(inverse_variance)
        ##############################
        composite_mu = self.get_composite_mu(mus, weights_over_variance, inverse_variance)  # (bsize, zdim)
        ##############################
        composite_kl = kls.sum()
        return composite_mu, composite_std, kls

    def forward(self, state):
        obs, goal, noise = state[..., :self.obs_dim], state[...,self.obs_dim:self.obs_dim+self.weight_network.goal_dim], state[...,self.obs_dim+self.weight_network.goal_dim:]
        bsize = state.size(0)
        weights, weights_std, weights_bottleneck_kl, weight_info = self.weight_network(state)
        weights, weights_kl, weight_dist = reparametrize(mu=weights, std=weights_std, noise=noise, device=self.get_device())
        weight_entropy = weight_dist.entropy().view(bsize, 1)
        weights = F.sigmoid(weights).view(bsize, self.k, 1)
        mus, stds, kls = self.execute_primitives(obs, no_grad=self.freeze_primitives)
        composite_mu, composite_std, kls = self.get_composite_parameters(mus, stds, kls, weights)
        return composite_mu, composite_std, kls, {'weight_entropy': weight_entropy, 'weight_std': weights_std}

class CompositeTransferPolicy(CompositePolicy):
    def __init__(self, weight_network, primitives, obs_dim, device):
        super(CompositeTransferPolicy, self).__init__(weight_network, primitives, obs_dim, device, freeze_primitives=True)

    def forward(self, state):
        weights, std, kl = self.weight_network(state)
        return weights, std, kl, {}

    def select_action(self, state, deterministic=False):
        return self.weight_network.select_action(state, deterministic)

    def get_log_prob(self, state, action):
        return self.weight_network.get_log_prob(state, action)

    def post_process(self, state, weights):
        obs = state[..., :self.obs_dim]
        bsize = state.size(0)
        weights = F.sigmoid(weights).view(bsize, self.k, 1)
        mus, stds, kls = self.execute_primitives(obs, no_grad=True)
        composite_mu, composite_std, kls = self.get_composite_parameters(mus, stds, kls, weights)
        return composite_mu

class LatentPolicy(StochasticGaussianPolicy):
    def __init__(self, goal_embedder, network_dims, outdim, obs_dim, device):
        super(LatentPolicy, self).__init__(device)
        self.goal_embedder = goal_embedder
        self.network_dims = network_dims
        self.outdim = outdim
        self.network = Feedforward(dims=network_dims[:-1], out_act=F.relu)
        self.parameter_producer = GaussianParams(network_dims[-1], outdim, device=device, custom_init=True)
        self.decoder = nn.Sequential(self.network, self.parameter_producer)
        self.obs_dim = obs_dim
        self.name = 'latent'
        self.zdim = self.goal_embedder.out_dim

    def forward(self, state):
        obs, goal, noise = state[..., :self.obs_dim], state[...,self.obs_dim:self.obs_dim+self.goal_embedder.goal_dim], state[...,self.obs_dim+self.goal_embedder.goal_dim:]
        bsize = state.size(0)
        goal_mu, goal_std, goal_bottleneck_kl, goal_info = self.goal_embedder(state)
        goal_embedding, goal_embedding_kl, goal_embedding_dist = reparametrize(mu=goal_mu, std=goal_std, noise=noise, device=self.get_device())
        goal_embedding = F.sigmoid(goal_embedding)
        inp = torch.cat((obs, goal_embedding), dim=-1)
        mu, logstd = self.decoder(inp)
        return mu, torch.exp(logstd), goal_embedding_kl, {'goal_std': goal_std}

class LatentTransferPolicy(DeterministicGaussianPolicy):
    def __init__(self, goal_embedder, decoder, obs_dim, device):
        super(LatentTransferPolicy, self).__init__(device)
        self.goal_embedder = goal_embedder
        self.decoder = decoder
        self.obs_dim = obs_dim
        self.name = 'latent'
        self.zdim = self.goal_embedder.out_dim

    def forward(self, state):
        goal_mu, goal_std, goal_bottleneck_kl = self.goal_embedder(state)
        return goal_mu, goal_std, goal_bottleneck_kl, {}

    def select_action(self, state, deterministic=False):
        return self.goal_embedder.select_action(state, deterministic)

    def get_log_prob(self, state, action):
        return self.goal_embedder.get_log_prob(state, action)

    def post_process(self, state, goal_embedding):
        obs  = state[..., :self.obs_dim]
        bsize = state.size(0)
        goal_embedding = F.sigmoid(goal_embedding)
        inp = torch.cat((obs, goal_embedding), dim=-1)
        with torch.no_grad():
            mu, logstd = self.decoder(inp)
        return mu

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
