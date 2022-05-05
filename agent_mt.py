# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import _sample_dirichlet, optim
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn 
from model import DQN
from copy import deepcopy
import torch.nn.functional as F 
from einops.layers.torch import Rearrange, Reduce
from torchvision.transforms import RandomAffine, RandomResizedCrop


class MultiTaskAgent():
  def __init__(self, args, action_space):
    self.action_space = action_space
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip

    self.online_net = DQN(args, self.action_space).to(device=args.device) 
    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary

        if args.load_conv_only:
          self.online_net.convs.load_state_dict({
            '.'.join(k.split('.')[1:]): v for k, v in state_dict.items() if 'convs' in k})
          self.online_net.freeze_conv()
          print('Loaded only Conv. Layers and freezing convolutions')
        elif args.load_conv_fc_h:
          self.online_net.convs.load_state_dict({
            '.'.join(k.split('.')[1:]): v for k, v in state_dict.items() if 'convs' in k})
          self.online_net.freeze_conv()
          
          self.online_net.fc_h_a.load_state_dict({
            '.'.join(k.split('.')[1:]): v for k, v in state_dict.items() if 'fc_h_a' in k})
          self.online_net.fc_h_v.load_state_dict({
            '.'.join(k.split('.')[1:]): v for k, v in state_dict.items() if 'fc_h_v' in k})
          print('Loaded only Conv. Layers and Hidden FC layers')
        else:
          self.online_net.load_state_dict(state_dict)
        print("Loading pretrained model: " + args.model)
        if args.reset_sigmas:
          print('Re-initializing the NoisyLinear sigmas in online net')
          self.online_net.reset_sigmas()
        
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)
  
      self.online_net = self.online_net.to(device=args.device)
        

    self.online_net.train()

    self.target_net = DQN(args, self.action_space).to(device=args.device)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    self.num_games_per_batch = args.num_games_per_batch
    self.apply_aug = args.apply_aug
    if self.apply_aug:
      print('Applying Random Resized Crop')
    self.aug = RandomResizedCrop(84, scale=(0.8, 1.0), ratio=(0.9, 1.1))
    self.device = args.device
    
  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    if type(state) == np.ndarray:
      if state.shape[-1] == 3:
        state = state.transpose((2,0,1))
      state = torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    if type(state) == np.ndarray:
      if state.shape[-1] == 3:
        state = state.transpose((2,0,1))
      state = torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def sample_batch(self, mems):
    """ Samples game-specific buffers first """
    assert len(mems) >= self.num_games_per_batch, 'Not enough buffers to learn from'
    sub_batch_size = int(self.batch_size / self.num_games_per_batch)
    sampled_buffer_ids = np.random.choice(len(mems), self.num_games_per_batch, replace=(len(mems) < self.num_games_per_batch))
    actual_bsize = int(sub_batch_size * self.num_games_per_batch)
    sub_batches = dict()
    for _id in sampled_buffer_ids:
      mem = mems[_id]
      idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(sub_batch_size)
      sub_batches[_id] = (idxs, states, actions, returns, next_states, nonterminals, weights)
    
    states = torch.cat([sub_batches[_id][1] for _id in sampled_buffer_ids])
    actions = torch.cat([sub_batches[_id][2] for _id in sampled_buffer_ids])
    returns = torch.cat([sub_batches[_id][3] for _id in sampled_buffer_ids])
    next_states = torch.cat([sub_batches[_id][4] for _id in sampled_buffer_ids])
    nonterminals = torch.cat([sub_batches[_id][5] for _id in sampled_buffer_ids])
    weights = torch.cat([sub_batches[_id][6] for _id in sampled_buffer_ids])

    if self.apply_aug:
      states = self.aug(states)
      next_states = self.aug(next_states)
    return states, actions, returns, next_states, nonterminals, weights, sampled_buffer_ids, sub_batches


  def learn(self, mems):
    states, actions, returns, next_states, nonterminals, weights, \
      sampled_buffer_ids, sub_batches = self.sample_batch(mems)
    
    sub_batch_size = int(self.batch_size / self.num_games_per_batch)
    actual_bsize = int(sub_batch_size * self.num_games_per_batch)

    loss_np = self.one_batch_update(states, actions, returns, next_states, nonterminals, weights, actual_bsize)
    for i, _id in enumerate(sampled_buffer_ids):
      mems[_id].update_priorities(sub_batches[_id][0], loss_np[i * sub_batch_size: (i + 1) * sub_batch_size])
   
  # def learn_single_buffer(self, mem): 
  #   # Sample transitions
  #   idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
  #   loss_np = self.one_batch_update(states, actions, returns, next_states, nonterminals, weights, self.batch_size)
  #   mem.update_priorities(idxs, loss_np)  # Update priorities of sampled transitions

  def learn_reptile(self, mems, frac_done, reptile_eps=[1, 0], inner_steps=5):
    """ samples 1 game and multiple updates """
    buffer_id = np.random.choice(len(mems))
    mem = mems[buffer_id]
    old_net = deepcopy(self.online_net)
    for step in range(inner_steps):
      # Sample transitions
      idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
      loss_np = self.one_batch_update(states, actions, returns, next_states, nonterminals, weights, self.batch_size)
      mem.update_priorities(idxs, loss_np)
    # update params
    eps = frac_done * reptile_eps[1] + (1 - frac_done) * reptile_eps[0]

    for old_param, new_param in zip(old_net.parameters(), self.online_net.parameters()):
        new_param.data.copy_(
            new_param.data + eps * (old_param.data - new_param.data))
    return

  def one_batch_update(self, states, actions, returns, next_states, nonterminals, weights, batch_size):
    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    log_ps_a = log_ps[range(batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(batch_size, self.atoms)
      offset = torch.linspace(0, ((batch_size - 1) * self.atoms), batch_size).unsqueeze(1).expand(batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    loss_np = loss.detach().cpu().numpy()
    if np.any(np.isnan(loss_np)):
      print(f'Batch update loss, {sum(np.isnan(loss_np))} elements got nan!')
      loss_np[np.isnan(loss_np)] = 0
      
    return loss_np
    
  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    if type(state) == np.ndarray:
      if state.shape[-1] == 3:
        state = state.transpose((2,0,1))
      state = torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()


class BCAgent(MultiTaskAgent):
  def __init__(self, args, env):
    self.action_space = env.action_space() 
    self.batch_size = args.batch_size
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

    if len(args.load_convs) > 0:
      state_dict = torch.load(args.load_convs, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
      self.convs.load_state_dict({
            '.'.join(k.split('.')[1:]): v for k, v in state_dict.items() if 'convs' in k}
      )
      print('Loaded convs from', args.load_convs)
    mlps = []
    mlp_sizes = [int(size) for size in args.mlps]
    print('MLP sizes:', mlp_sizes)
    for i, size in enumerate(mlp_sizes):
      mlps.append(nn.Linear(self.conv_output_size if i == 0 else mlp_sizes[i-1], size))
      mlps.append(nn.ReLU()) 
    mlps.append(nn.Linear(mlp_sizes[-1], self.action_space))
    self.mlps = nn.Sequential(*mlps)

    self.online_net = nn.Sequential(
      self.convs, 
      Rearrange('b c h w -> b (c h w)'), 
      self.mlps).to(args.device)
    self.online_net.train()

    self.optimiser = optim.Adam(self.online_net.parameters(), 
      lr=args.learning_rate, eps=args.adam_eps)
    self.num_games_per_batch = args.num_games_per_batch
    self.device = args.device

  def learn(self, states, actions, eval=False): 
    pred_actions = self.online_net(states) 
    loss = nn.CrossEntropyLoss()(input=pred_actions, target=actions)
    acc = (pred_actions.argmax(1) == actions).float().mean().item()
    if eval:
      return loss.detach().cpu().numpy().item(), acc
    self.online_net.zero_grad()
    loss.mean().backward()  # Backpropagate importance-weighted minibatch loss
    self.optimiser.step()
    
    return loss.detach().cpu().numpy().item(), acc

  def reset_noise(self):
    return

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      action = self.online_net(state.unsqueeze(0)) 
      return action.argmax(1).item()
  
  def update_target_net(self):
    return 

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    return 0

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
