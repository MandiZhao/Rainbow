# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5, noiseless=False):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    # TODO: reinit noise or swap with eps greedy 
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()
    self.noiseless = noiseless

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def reset_sigmas(self):
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features)) 
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def forward(self, input):
    if self.training and not self.noiseless:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)

class DQN(nn.Module):
  def __init__(self, args, action_space):
    super(DQN, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space
    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
    elif args.architecture == 'data-effx2':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
    
    self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std, noiseless=args.noiseless)
    self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std, noiseless=args.noiseless)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std, noiseless=args.noiseless)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std, noiseless=args.noiseless)

  def forward(self, x, log=False):
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()

  def reset_sigmas(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_sigmas()

  def reinit_fc(self, args):
    if args.reinit_fc == 0:
      print('Not reinitializing any fc. layers')
      return 
    if args.reinit_fc >= 1:
      print('Reinitializing last fully connected layers')
      self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
      self.fc_z_a = NoisyLinear(args.hidden_size, self.action_space * self.atoms, std_init=args.noisy_std)
    if args.reinit_fc == 2:
      print('Reinitializing early fully connected layers')
      self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
      self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    
  def freeze_conv(self):
    print('Freezing all conv. layers')
    for p in self.convs.parameters():
      p.requires_grad = False
    
  def unfreeze_conv(self):
    print('Unfreezing all conv. layers')
    for p in self.convs.parameters():
      if p.requires_grad:
        print('Warining, conv. layer was not frozen')
      p.requires_grad = True

def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)

    return mu, sigma_squared

def _compute_kl_loss(z_means, z_vars):
    """ compute KL( q(z|c) || r(z) ) """
    latent_dim = z_means.shape[-1]
    device = z_means.device
    assert z_vars.shape[-1] == latent_dim
    prior = torch.distributions.Normal(
        torch.zeros(latent_dim).to(device), 
        torch.ones(latent_dim).to(device)
        )
    posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) \
      for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
    kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
    kl_div_sum = torch.sum(torch.stack(kl_divs))
    return kl_div_sum

class PearlDQN(DQN):
  def __init__(self, args, action_space):
    self.atoms = args.atoms
    self.action_space = action_space
    nn.Module.__init__(self)
    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576

    self.pearl_z_size = args.pearl_z_size
    print('Initializing pearl network with z size:', self.pearl_z_size)
    self.fc_h_v = NoisyLinear(self.conv_output_size + self.pearl_z_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.conv_output_size + self.pearl_z_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

    self.pearl_enc = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
    self.pearl_mlp = nn.Sequential(
      nn.Linear(576 + 2, args.pearl_hidden_size), nn.ReLU(), nn.Linear(args.pearl_hidden_size, self.pearl_z_size * 2),
    )


  def forward(self, x, context, log=False):
    x = self.convs(x).view(-1, self.conv_output_size)
    z, kl_loss = self.sample_z(context)
    z = z.to(x.device)
    if len(z.shape) == 1:
      z = z.unsqueeze(0)
    if z.shape[0] != x.shape[0]:
      z = repeat(z, '1 z -> b z', b=x.shape[0])
    x = torch.cat([x, z], dim=1)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    
    return q, kl_loss

  def sample_z(self, context):
    if context is None:
      z_means = torch.zeros(self.pearl_z_size)
      z_vars = torch.ones(self.pearl_z_size)
      posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) \
            for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
      z = [d.rsample() for d in posteriors]
      z = torch.stack(z)

      kl_loss = 0
      return z, kl_loss
    
    conv_in = context['state']
    # print('conv in shape', conv_in.shape)
    # assert len(conv_in.shape) == 5, 'Must have shape batch, n_frames, channels, height, width'
    # if len(conv_in.shape) == 5:
    #   need_reshape = True
    #   batch, n_frames, c, h, w = conv_in.shape
    # conv_in = rearrange(conv_in, 'b n c h w -> (b n) c h w')
    # conv_z = self.pearl_enc(rearrange(conv_in, 'b n c h w -> (b n) c h w'))
    # conv_z = rearrange(conv_z, '(b n) d -> b n d', b=batch, n=n_frames)
    conv_z = self.pearl_enc(conv_in).view(-1, self.conv_output_size)
    actions, rewards = context['action'], context['reward']
    actions = rearrange(actions, 'b -> b 1')
    rewards = rearrange(rewards, 'b -> b 1')
    mlp_in = torch.cat([conv_z, actions, rewards], dim=1)
    out = self.pearl_mlp(mlp_in).unsqueeze(0)

    size = self.pearl_z_size
    mus, sigmas = out[:, :, :size], F.softplus(out[:, :, size:]) # shape (b, k, z_size)
    z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mus), torch.unbind(sigmas))]
    # print(z_params[0].shape)
    z_means = torch.stack([p[0] for p in z_params])
    z_vars = torch.stack([p[1] for p in z_params])
    # print(z_means.shape)
    posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) \
            for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
    z = [d.rsample() for d in posteriors]
    # print(z[0].shape)
    z = torch.stack(z) 
    kl_loss = _compute_kl_loss(z_means, z_vars)
    return z, kl_loss
