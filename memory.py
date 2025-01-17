# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import torch

Transition_dtype = np.dtype([('timestep', np.int32), ('state', np.uint8, (84, 84)), ('action', np.int32), ('reward', np.float32), ('nonterminal', np.bool_)])
blank_trans = (0, np.zeros((84, 84), dtype=np.uint8), 0, 0.0, False)

Context_dtype = np.dtype([('state', np.uint8, (4, 84, 84)), ('action', np.int32), ('reward', np.float32)])
blank_context = (np.zeros((4, 84, 84), dtype=np.uint8), 0, 0.0)

Procgen_Transition_dtype = np.dtype([('timestep', np.int32), ('state', np.uint8, (3, 64, 64)), ('action', np.int32), ('reward', np.float32), ('nonterminal', np.bool_)])
blank_procgen_trans = (0, np.zeros((3, 64, 64), dtype=np.uint8), 0, 0.0, False)
# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
  def __init__(self, size):
    self.index = 0
    self.size = size
    self.full = False  # Used to track actual capacity
    self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
    self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
    self.data = np.array([blank_trans] * size, dtype=Transition_dtype)  # Build structured array
    self.max = 1  # Initial max value to return (1 = 1^ω)
    self.rewards = np.zeros(size)
    self.reward_stat = [0, 1]

  # Updates nodes values from current tree
  def _update_nodes(self, indices):
    children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
    self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

  # Propagates changes up tree given tree indices
  def _propagate(self, indices):
    parents = (indices - 1) // 2
    unique_parents = np.unique(parents)
    self._update_nodes(unique_parents)
    if parents[0] != 0:
      self._propagate(parents)

  # Propagates single value up tree given a tree index for efficiency
  def _propagate_index(self, index):
    parent = (index - 1) // 2
    left, right = 2 * parent + 1, 2 * parent + 2
    self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
    if parent != 0:
      self._propagate_index(parent)

  # Updates values given tree indices
  def update(self, indices, values):
    self.sum_tree[indices] = values  # Set new values
    self._propagate(indices)  # Propagate values
    current_max_value = np.max(values)
    self.max = max(current_max_value, self.max)

  # Updates single value given a tree index for efficiency
  def _update_index(self, index, value):
    self.sum_tree[index] = value  # Set new value
    self._propagate_index(index)  # Propagate value
    self.max = max(value, self.max)

  def append(self, data, value):
    self.data[self.index] = data  # Store data in underlying data structure
    self._update_index(self.index + self.tree_start, value)  # Update tree
    self.index = (self.index + 1) % self.size  # Update index
    self.full = self.full or self.index == 0  # Save when capacity reached
    self.max = max(value, self.max)
    self.rewards[self.index] = data[3]
    self.reward_stat[0] = np.mean(self.rewards) if self.full else np.mean(self.rewards[:self.index]) 
    self.reward_stat[1] = np.std(self.rewards) if self.full else np.std(self.rewards[:self.index])
    if np.isnan(self.reward_stat[1]) or self.reward_stat[1] == 0:
      self.reward_stat[1] = 1
    # print(self.reward_stat)

  # Searches for the location of values in sum tree
  def _retrieve(self, indices, values):
    children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
    # If indices correspond to leaf nodes, return them
    if children_indices[0, 0] >= self.sum_tree.shape[0]:
      return indices
    # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
    elif children_indices[0, 0] >= self.tree_start:
      children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
    left_children_values = self.sum_tree[children_indices[0]]
    successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
    successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
    successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
    return self._retrieve(successor_indices, successor_values)

  # Searches for values in sum tree and returns values, data indices and tree indices
  def find(self, values):
    indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
    data_index = indices - self.tree_start
    return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

  # Returns data given a data index
  def get(self, data_index):
    return self.data[data_index % self.size]

  def total(self):
    return self.sum_tree[0]

class ContextBuffer():
  def __init__(self, args):
    size = args.context_length
    self.size = size 
    self.data = np.array([blank_context] * size, dtype=Context_dtype)
    self.index = 0
    self.full = False
    self.device = args.device

  def append(self, data):
    self.data[self.index] = data  # Store data in underlying data structure
    self.index = (self.index + 1) % self.size  # Update index
    self.full = self.full or self.index == 0  # Save when capacity reached
  
  def sample(self, batch_size):
    if self.full:
      data_idxs = np.random.randint(0, self.size, size=batch_size)
    elif self.index == 0:
      return None # Buffer is empty
    elif 0 < self.index < batch_size:
      data_idxs = np.arange(self.index)
    else:
      data_idxs = np.random.randint(0, self.index, size=batch_size)
    data = self.data[data_idxs]

    states = torch.tensor(data['state'], device=self.device, dtype=torch.float32).div_(255)
    # Discrete actions to be used as index
    actions = torch.tensor(data['action'], dtype=torch.int64, device=self.device)
    rewards = torch.tensor(data['reward'], dtype=torch.float32, device=self.device)
    return {'state': states, 'action': actions, 'reward': rewards}

class ReplayMemory():
  def __init__(self, args, capacity):
    self.device = args.device
    self.capacity = capacity
    self.history = args.history_length
    self.discount = args.discount
    self.n = args.multi_step
    self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
    self.priority_exponent = args.priority_exponent
    self.t = 0  # Internal episode timestep counter
    self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32, device=self.device)  # Discount-scaling vector for n-step returns
    self.transitions = SegmentTree(capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
    self.use_procgen = args.use_procgen
    if args.use_procgen:
      print('Updating replay buffer data type to 3 channel and single step!')
      self.transitions.data = np.array([blank_procgen_trans] * capacity, dtype=Procgen_Transition_dtype)
      self.history = 1 
    self.normalize_reward = args.normalize_reward
    
    self.context_buffer = None
    if args.pearl:
      self.context_buffer = ContextBuffer(args)

  # Adds state and action at time t, reward and terminal at time t + 1
  def append(self, state, action, reward, terminal):
    if self.use_procgen:
      if state.shape[-1] == 3:
        state = state.transpose((2,0,1))
      state = torch.tensor(state, dtype=torch.uint8, device=torch.device('cpu'))
      context_state = state.to(dtype=torch.uint8, device=torch.device('cpu'))
      state = context_state
    else:
      context_state = state.mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))
      state = context_state[-1] # Only store last frame and discretise to save memory
    self.transitions.append((self.t, state, action, reward, not terminal), self.transitions.max)  # Store new transition with maximum priority
    self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0
    if self.context_buffer is not None:
      self.context_buffer.append((context_state, action, reward))

  # Returns the transitions with blank states where appropriate
  def _get_transitions(self, idxs):
    transition_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1)
    transitions = self.transitions.get(transition_idxs)
    transitions_firsts = transitions['timestep'] == 0
    blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
    for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
      blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1], transitions_firsts[:, t + 1]) # True if future frame has timestep 0
    for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
      blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t]) # True if current or past frame has timestep 0
    transitions[blank_mask] = blank_procgen_trans if self.use_procgen else blank_trans 
    return transitions

  # Returns a valid sample from each segment
  def _get_samples_from_segments(self, batch_size, p_total):
    segment_length = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
    segment_starts = np.arange(batch_size) * segment_length
    # print('Segment starts:', segment_starts, 'Segment length:', segment_length)
    valid = False
    while not valid: 
      samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts  # Uniformly sample from within all segments
      probs, idxs, tree_idxs = self.transitions.find(samples)  # Retrieve samples from tree with un-normalised probability
      idxs[(self.transitions.index - idxs) % self.capacity <= self.n] = self.n+1
      idxs[(idxs - self.transitions.index) % self.capacity < self.history] = self.n + 1
      if np.all((self.transitions.index - idxs) % self.capacity > self.n) and np.all((idxs - self.transitions.index) % self.capacity >= self.history) and np.all(probs != 0):
        valid = True  # Note that conditions are valid but extra conservative around buffer index 0
    # Retrieve all required transition data (from t - h to t + n)
    transitions = self._get_transitions(idxs)
    # Create un-discretised states and nth next states
    all_states = transitions['state']
    states = torch.tensor(all_states[:, :self.history], device=self.device, dtype=torch.float32).div_(255)
    next_states = torch.tensor(all_states[:, self.n:self.n + self.history], device=self.device, dtype=torch.float32).div_(255)
    # Discrete actions to be used as index
    actions = torch.tensor(np.copy(transitions['action'][:, self.history - 1]), dtype=torch.int64, device=self.device)
    # Calculate truncated n-step discounted returns R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
    if self.normalize_reward:
      [rew_mean, rew_std] = self.transitions.reward_stat 
      transitions['reward'] = transitions['reward'] / rew_std # + 1e-8
      if np.any(np.isnan(transitions['reward'])):
        print('got nan rew!', rew_std, transitions['reward'])
        transitions['reward'][np.isnan(transitions['reward'])] = 0
    rewards = torch.tensor(np.copy(transitions['reward'][:, self.history - 1:-1]), dtype=torch.float32, device=self.device)
    R = torch.matmul(rewards, self.n_step_scaling)
    # Mask for non-terminal nth next states
    nonterminals = torch.tensor(np.expand_dims(transitions['nonterminal'][:, self.history + self.n - 1], axis=1), dtype=torch.float32, device=self.device)
    return probs, idxs, tree_idxs, states, actions, R, next_states, nonterminals

  def sample(self, batch_size):
    p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
    assert not np.isnan(p_total), 'Total priority in replay buffer is nan.'
    probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = self._get_samples_from_segments(batch_size, p_total)  # Get batch of valid samples
    probs = probs / p_total  # Calculate normalised probabilities
    capacity = self.capacity if self.transitions.full else self.transitions.index
    weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
    weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch
    return tree_idxs, states, actions, returns, next_states, nonterminals, weights

  def sample_context(self, window):
    assert self.context_buffer is not None, 'Context buffer not initialised.'
    return self.context_buffer.sample(window)

  def update_priorities(self, idxs, priorities):
    priorities = np.power(priorities, self.priority_exponent)
    self.transitions.update(idxs, priorities)

  # Set up internal state for iterator
  def __iter__(self):
    self.current_idx = 0
    return self

  # Return valid states for validation
  def __next__(self):
    if self.current_idx == self.capacity:
      raise StopIteration
    transitions = self.transitions.data[np.arange(self.current_idx - self.history + 1, self.current_idx + 1)]
    transitions_firsts = transitions['timestep'] == 0
    blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
    for t in reversed(range(self.history - 1)):
      blank_mask[t] = np.logical_or(blank_mask[t + 1], transitions_firsts[t + 1]) # If future frame has timestep 0
    transitions[blank_mask] = blank_procgen_trans if self.use_procgen else blank_trans
    state = torch.tensor(transitions['state'], dtype=torch.float32, device=self.device).div_(255)  # Agent will turn into batch
    self.current_idx += 1
    return state

  next = __next__  # Alias __next__ for Python 2 compatibility
