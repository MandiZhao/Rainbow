# -*- coding: utf-8 -*-
"""
python main_bc.py --games pong --batch_size 128 --learning_rate 1e-3 --id 128-Lr1e-3
"""

from __future__ import division
import argparse 
import os
import pickle

import atari_py
import numpy as np
import torch
from tqdm import trange

from agent_mt import BCAgent  
from test import test, test_all_games

from env_mt import MultiTaskEnv
from omegaconf import DictConfig, OmegaConf
from memory_dataset import Transition
import wandb 
from os.path import join 
from glob import glob
from natsort import natsorted
from torch.utils.data import IterableDataset

DATA_DIR = [
  '/home/mandi/Rainbow/results',
  '/shared/mandi/rainbow_data'
]

class BCDataset(IterableDataset):
    def __init__(self, data, batch_size, device):
        self.batch_size = batch_size   
        self.data = data
        self.device = device
    
    def sample(self):
        idxs = np.random.choice(len(self.data), self.batch_size, replace=False)
        states, actions = [], []
        for i in idxs:
          with open(self.data[i], 'rb') as pickle_file:
            step = pickle.load(pickle_file)
          states.append(step.state)
          actions.append(step.action)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        # states = torch.tensor(states, dtype=torch.float32, device=self.device).div_(255)
        # actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        return states, actions

    
    def _generator(self):
        while True:
            yield self.sample()

    def __iter__(self):
        return iter(self._generator())

def test_bc(games, env_cfg, args, T, bc, metrics, results_dir):
  env = MultiTaskEnv(env_cfg)
  env.eval()
  for _id, game in enumerate(games):  
    game_metrics = metrics[f'game_{game}'] # metrics['game_{:02d}'.format(_id)]
    env.reset()
    env._set_game(_id)
    game_metrics['steps'].append(T)
    T_rewards, T_Qs, traj_lengths = [], [], []

    # Test performance over several episodes
    done = True
    for _ in range(args.evaluation_episodes):
      while True:
        if done:
          state, reward_sum, done = env.reset(resample_game=False), 0, False
          step_count = 0 
        action = bc.act(state)  # Choose an action NO ε-greedily
        state, reward, done, info = env.step(action)  # Step
        step_count += 1
        assert info.get('game_id') == _id, 'Game ID mismatch: {} != {}'.format(info.get('game_id'), _id)
        reward_sum += reward
        if args.render:
          env.render()
        if done:
          T_rewards.append(reward_sum) 
          traj_lengths.append(step_count)
          break
          
    env.close()
    avg_reward = sum(T_rewards) / len(T_rewards) 
    game_metrics['avg_reward'] = avg_reward
    game_metrics['avg_traj_length'] = sum(traj_lengths) / len(traj_lengths)
  return  

 


# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='BC')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
# parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
parser.add_argument('--T_max', type=int, default=int(50_000), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='data-efficient', choices=['canonical', 'data-efficient', 'data-effx2'], metavar='ARCH', help='Network architecture')
parser.add_argument('--noisy_std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay_frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning_rate', type=float, default=3e-4, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch_size', type=int, default=64, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn_start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation_interval', type=int, default=int(100), metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation_episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation_size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint_interval', type=int, default=500000, help='How often to checkpoint the model, defaults to 0.5M ')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')

# takes in a list of names
parser.add_argument('--games', nargs='+', default=['pong'], help='Environment names')
parser.add_argument('--no_wb', action='store_true', help='Skip wandb for logging')
parser.add_argument('--separate_buffer', action='store_true', help='Use separate buffer for training and evaluation')
parser.add_argument('--num_games_per_batch', type=int, default=1, help='Number of games in one update batch, if >1 games are used the default is num_games')
parser.add_argument('--reptile_k', type=int, default=0, help='set > 0 for reptile')
parser.add_argument('--load_memory', action='store_true', help='Load memory from file, fintuning runs dont need to load memory')
parser.add_argument('--normalize_reward', action='store_true', help='Normalize rewards')
parser.add_argument('--load_conv_only', action='store_true', help='Load convolutional layers only')
parser.add_argument('--load_conv_fc_h', action='store_true', help='Load convolutional layers and hidden FC layers')
parser.add_argument('--reinit_fc', type=int, default=1, help='Reinitialize fully connected layers, 0 means no reinit')
parser.add_argument('--unfreeze_conv_when', type=int, default=50e6, help='Unfreeze convolutional layers when this many steps have passed')
parser.add_argument('--pad_action_space', type=int, default=0, help='Pad action space with zeros, use for preparing single-task agent to fine-tune')
parser.add_argument('--act_greedy_until', type=int, default=0, help='Act greedily until this many steps have passed')
parser.add_argument('--greedy_eps', type=float, default=0.1, help='Act greedily every n steps')
parser.add_argument('--reset_sigmas', action='store_true', help='Reset sigmas in Noisy Linear nets')
parser.add_argument('--noiseless', action='store_true', help='Disable sigmas in Noisy Linear nets')

# offline dataset
parser.add_argument('--dataset_dir', type=str, default='/home/mandi/rainbow_data/', help='Path to offline dataset')
parser.add_argument('--val_split', type=float, default=0.1, help='Validation split')
parser.add_argument('--val_steps', type=int, default=10, help='Validation steps')
parser.add_argument('--cap_data', type=int, default=10000, help='cap num of episodes')
parser.add_argument('--mlps', nargs='+', default=[512])
# Setup
args = parser.parse_args()

args.id = '{}-'.format('-'.join(sorted(args.games))) + args.id + '-B{}'.format(args.batch_size)
args.id += "-MLP" + f"{'x'.join([str(size) for size in args.mlps])}"

results_dir = os.path.join('bc_results', args.id)
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda:0')
  torch.cuda.manual_seed(np.random.randint(1, 10000))
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')

# Environment
cfg = OmegaConf.load('conf/config.yaml').env
cfg.games = list(args.games)
games = sorted(cfg.games)
env = MultiTaskEnv(cfg)
env.train()
action_space = env.action_space()
if not args.no_wb:
  wandb.init(project='Rainbow', group='bc', name=args.id)
  wandb.config.update(vars(args))

bc = BCAgent(args, env)

metrics = {} 
for _id, game in enumerate(games):
  # metrics['game_{:02d}'.format(_id)] = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
  metrics[f'game_{game}'] = {
      'steps': [], 
      'rewards': [], 
      }

def log_metrics(T):
  tolog = {'Env Step': T}
  log_string = f"T = {T}/{args.T_max}"
  for key, v in metrics.items():
    if 'game_' in key:
      for kk, vv in v.items():
        if not isinstance(vv, list):
          tolog[key + '/' + kk] = vv 
          tolog[kk + '/' + key] = vv 
  tolog['train_loss'] = metrics['train_loss']
  tolog['val_loss'] = metrics['val_loss']
  if not args.no_wb: 
      wandb.log(tolog)

print("Loading dataset...")
train_data, val_data = [], []
for game in args.games:
  game_path = join(args.dataset_dir, f'{game}_rollouts')
  assert os.path.exists(game_path), f"{game_path} does not exist"
  
  episode_paths = natsorted(glob( join(game_path, 'episode*')))
  episode_paths = episode_paths[:args.cap_data]
  num_val = max(1, int(len(episode_paths) * args.val_split))
  num_train = len(episode_paths) - num_val
  print(f'Loading {num_train} training episodes and {num_val} for validation' )
  for i, eps in enumerate(episode_paths):
    steps = natsorted(glob(join(eps + '/*.pkl')))
    for step in steps:
      # pickle load step
      with open(step, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        if i < num_train: 
          train_data.append(step) #(data.state, data.action))
        else:
          val_data.append(step) # (data.state, data.action))
    if i % 50 == 0:
      print('Done loading episode {}'.format(i))

train_dataset, val_dataset = BCDataset(train_data, args.batch_size, args.device), BCDataset(val_data, args.batch_size, args.device)
train_iter, val_iter = iter(train_dataset), iter(val_dataset)

bc.train()  
for T in trange(0, args.T_max + 1):
  states, actions = next(train_iter)

  loss = bc.learn(states, actions)
  metrics['train_loss'] = loss 

  if T % args.evaluation_interval == 0:
    bc.eval()  # Set DQN (online network) to evaluation mode
    with torch.no_grad():
      val_loss = []
      for _ in range(args.val_steps):
        val_states, val_actions = next(val_iter)
        val_loss.append(bc.learn(val_states, val_actions, eval=True))
    val_loss_mean = np.mean(val_loss)
    metrics['val_loss'] = val_loss_mean
    test_bc(games, cfg, args, T, bc, metrics, results_dir)  # Test
    log_metrics(T)
    bc.train() 

  # Checkpoint the network
  if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
    bc.save(results_dir, f'checkpoint_{T}.pth')


env.close()