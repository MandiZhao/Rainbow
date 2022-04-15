# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle

import atari_py
import numpy as np
import torch
from tqdm import trange

from agent_mt import MultiTaskAgent
from agent_pearl import PearlAgent
from env import Env
from memory import ReplayMemory
from test import test, test_all_games

from env_mt import MultiTaskEnv
from omegaconf import DictConfig, OmegaConf
import wandb 
GAME_NAMES = {
   'alien': 'Alien',
   'amidar': 'Amidar',
   'assault': 'Assault',
   'bank_heist': 'BankHeist',
   'battle_zone': 'BattleZone',
   'breakout': 'Breakout',
   'enduro': 'Enduro',
   'pong': 'Pong',
   'asteroids': 'Asteroids',
   'qbert': 'Qbert',
   'seaquest': 'Seaquest',
   'space_invaders': 'SpaceInvaders',
   'up_n_down': 'UpNDown',
   'kangaroo': 'Kangaroo',
   'demon_attack': 'DemonAttack',
   'krull': 'Krull',
   'frostbite': 'FrostBite',
   'ms_pacman': 'MsPacman',
   'jamesbond': 'JamesBond',
   'demon_attack': 'DemonAttack'
 }

GAME_SETS = {
  '8task-v1': ['asteroids', 'alien', 'beam_rider', 'frostbite', 'krull', 'ms_pacman', 'road_runner', 'seaquest'],
  '8task-v2': ['asteroids', 'alien', 'beam_rider', 'breakout', 'frostbite', 'krull', 'road_runner', 'seaquest'],
  '20-task': ['asteroids', 'alien', 'beam_rider', 'breakout', 'frostbite'] + \
       ['krull', 'road_runner', 'seaquest', 'amidar', 'assault'] + \
       ['asterix', 'bank_heist', 'boxing', 'chopper_command', 'private_eye'] + \
       ['crazy_climber', 'kung_fu_master', 'freeway', 'gopher', 'hero']
}

DATA_DIR = [
  '/home/mandi/Rainbow/results',
  '/shared/mandi/rainbow_data'
]
# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
# parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
parser.add_argument('--T_max', type=int, default=int(5e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient', 'data-effx2'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
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
parser.add_argument('--learning_rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch_size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn_start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation_interval', type=int, default=int(2e4), metavar='STEPS', help='Number of training steps between evaluations')
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

# PEARL
parser.add_argument('--pearl', action='store_true', help='Use PEARL')
parser.add_argument('--pearl_z_size', type=int, default=32, help='latent size for PEARL')
parser.add_argument('--pearl_hidden_size', type=int, default=512, help='hidden size for PEARL')
parser.add_argument('--context_length', type=int, default=100, help='num. of transitions to sample from context')
parser.add_argument('--context_window', type=int, default=10, help='context buffer window size')
# Setup
args = parser.parse_args()

if args.reptile_k > 0:
  args.separate_buffer = True
  print('Using Reptile with inner step size: {}'.format(args.reptile_k))
  args.id = f'Reptile{args.reptile_k}-' + args.id

if args.pearl: 
  args.separate_buffer = True
  args.id = f'PEARL-' + args.id
if len(args.games) == 1 and GAME_SETS.get(args.games[0], None) is not None:
  print('Using pre-defined game set: {}'.format(args.games[0]))
  key = args.games[0]
  args.games = GAME_SETS[key]
  args.id = f'{key}-' + args.id
  args.id += f'-B{args.batch_size}' + ('-NormRew' if args.normalize_reward else '')

else:
  args.id = '{}-'.format(GAME_NAMES[args.games[0]]) + args.id

args.id += '-SepBuf' if args.separate_buffer else ''
args.id += '-Seed{}'.format(args.seed)

if args.model is not None:
  args.id += "-FreezeConv-ReInit{}FC-UnfreezeAt{:0.0e}".format(args.reinit_fc, args.unfreeze_conv_when) if args.load_conv_only else '-NoFreezeConv'
  print('Pad test-time environment action space to 18 for finetuning')
  
  if len(args.model.split('/')) == 2:
    for append_dir in DATA_DIR:
      full_path = os.path.join(append_dir, args.model)
      if os.path.exists(full_path):
        print('Prepending model path to: {}'.format(full_path))
        args.model = full_path
        break

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))


results_dir = os.path.join('results', args.id)
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(np.random.randint(1, 10000))
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')


# Simple ISO 8601 timestamped logger
def log(s, args):
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'rb') as pickle_file:
      return pickle.load(pickle_file)
  else:
    with bz2.open(memory_path, 'rb') as zipped_pickle_file:
      return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'wb') as pickle_file:
      pickle.dump(memory, pickle_file)
  else:
    with bz2.open(memory_path, 'wb') as zipped_pickle_file:
      pickle.dump(memory, zipped_pickle_file)

# Environment
cfg = OmegaConf.load('conf/config.yaml').env
cfg.games = list(args.games)
games = sorted(cfg.games)
if len(games) > 1:
  args.learn_start = int(args.learn_start / len(games))
  print('Setting learn_start to **shorter** for multi-task runs {}'.format(args.learn_start))
  if args.num_games_per_batch == 1:
    print('Default setting num_games_per_batch to {} for multi-task runs'.format(len(games)))
    args.num_games_per_batch = len(games)

if args.model is not None and args.reinit_fc == 0:
  cfg.modify_action_size = 18 
if args.pad_action_space > 0:
  print('Warning! Padding action space with {} zeros'.format(args.pad_action_space))
  cfg.modify_action_size = args.pad_action_space

env = MultiTaskEnv(cfg)
env.train()
action_space = env.action_space()
if not args.no_wb:
  wandb.init(project='Rainbow', group='multitask', name=args.id)
  wandb.config.update(vars(args))

# Agent
if args.pearl:
  dqn = PearlAgent(args, env)
else:
  dqn = MultiTaskAgent(args, env)

metrics = {} 
for _id, game in enumerate(games):
  # metrics['game_{:02d}'.format(_id)] = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
  metrics[f'game_{game}'] = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}

def log_metrics(T):
  tolog = {'Env Step': T}
  log_string = f"T = {T}/{args.T_max}"
  for key, v in metrics.items():
    if 'game_' in key:
      for kk, vv in v.items():
        if not isinstance(vv, list):
          tolog[key + '/' + kk] = vv 
          tolog[kk + '/' + key] = vv 
      log_string += f" {key} | Avg. reward: {v.get('avg_reward', 0):2f} | Avg. Q: {v.get('avg_q', 0):2f}"
  if not args.no_wb: 
    wandb.log(tolog)

# If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
# if args.model is not None and not args.evaluate and :
if args.load_memory:
  if not args.memory:
    raise ValueError('Cannot resume training without memory save path. Aborting...')
  elif not os.path.exists(args.memory):
    raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))
  else:
    mem = load_memory(args.memory, args.disable_bzip_memory)

else:
  if args.normalize_reward:
    args.separate_buffer = True
    print('Using separate game-specific buffer for normalizing rewards')
  if args.separate_buffer:
    mems = dict()
    for i in range(env.num_games):
      mems[i] = ReplayMemory(args, int(args.memory_capacity / env.num_games))
  else:
    mems = {0: ReplayMemory(args, args.memory_capacity)}


priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


# Construct validation memory
# val_mem = ReplayMemory(args, args.evaluation_size)
if args.separate_buffer:
  val_mems = dict()
  for i in range(env.num_games):
    val_mems[i] = ReplayMemory(args, int(args.memory_capacity / env.num_games))
else:
  val_mems = {0: ReplayMemory(args, args.memory_capacity)}
T, done = 0, True
while T < args.evaluation_size:
  if done:
    state = env.reset()

  next_state, _, done, info = env.step(np.random.randint(0, action_space))
  buffer_idx = info.get('game_id') if args.separate_buffer else 0
  val_mem = val_mems[buffer_idx]
  val_mem.append(state, -1, 0.0, done)
  state = next_state
  T += 1

if args.evaluate:
  dqn.eval()  # Set DQN (online network) to evaluation mode
  avg_rewards, avg_Qs = test_all_games(games, args, 0, dqn, val_mems, metrics, results_dir, evaluate=True)  # Test
  for i, game in enumerate(games):
    print('Avg. reward: ' + str(avg_rewards[i]) + ' | Avg. Q: ' + str(avg_Qs[i]))
else:
  # Training loop
  print('Evaluate before training')
  dqn.eval()  # Set DQN (online network) to evaluation mode
  print(dqn.action_space)
  #raise ValueError('Not implemented')
  test_all_games(games, cfg, args, 0, dqn, val_mems, metrics, results_dir)  # Test
  log_metrics(0) 
  dqn.train()
  total_T = 0
  for _id, game in enumerate(env.games):
    env.reset()
    env._set_game(_id)
    done = True
    for env_T in range(args.learn_start):
      if done:
        state = env.reset(resample_game=False)

      if env_T % args.replay_frequency == 0:
        dqn.reset_noise()  # Draw a new set of noisy weights
      buffer_idx = env.get_current_game_id() if args.separate_buffer else 0
      mem = mems[buffer_idx]
      if args.pearl:
        action = dqn.act(state, mem)
      else:
        action = dqn.act(state)  # Choose an action greedily (with noisy weights)
      
      next_state, reward, done, info = env.step(action)  # Step
      if args.reward_clip > 0:
        reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
      
      mem.append(state, action, reward, done)  # Append transition to memory
      total_T += 1
    print('done appending to buffer index {}'.format(buffer_idx))
  print('done with all games, total T {}'.format(total_T))
  
  done = True
  for T in trange(total_T, args.T_max + 1):
    if done:
      state = env.reset()

    if T == int(args.unfreeze_conv_when) and args.load_conv_only:
      dqn.online_net.unfreeze_conv()
      print('unfreezing conv layers at T {}'.format(T))

    if T % args.replay_frequency == 0:
      dqn.reset_noise()  # Draw a new set of noisy weights
    buffer_idx = env.get_current_game_id() if args.separate_buffer else 0
    mem = mems[buffer_idx]
    mem.append(state, action, reward, done)  # Append transition to memory

    if args.pearl:
      action = dqn.act(state, mem)
    else:
      action = dqn.act(state)  # Choose an action greedily (with noisy weights)
    next_state, reward, done, info = env.step(action)  # Step
    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
    
    # Train and test
    # if T >= args.learn_start:
    mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

    if T % args.replay_frequency == 0:
      if args.reptile_k > 0:
        frac = T/args.T_max
        dqn.learn_reptile(mems=mems, frac_done=frac, inner_steps=args.reptile_k)
      else:
        dqn.learn(mems)
      # elif args.separate_buffer:
      #   dqn.learn_multi_buffer(mems)
      # else:
      #   dqn.learn_single_buffer(mem)  # Train with n-step distributional double-Q learning

    if T % args.evaluation_interval == 0:
      dqn.eval()  # Set DQN (online network) to evaluation mode
      test_all_games(games, cfg, args, T, dqn, val_mems, metrics, results_dir)  # Test
      log_metrics(T)

      # log(log_string, args)
      dqn.train()  # Set DQN (online network) back to training mode

      # If memory path provided, save it
      if args.memory is not None:
        assert not args.separate_buffer, 'Separate buffer not supported with memory!'
        save_memory(mem, args.memory, args.disable_bzip_memory)

    # Update target network
    if T % args.target_update == 0:
      dqn.update_target_net()

    # Checkpoint the network
    if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
      dqn.save(results_dir, f'checkpoint_{T}.pth')

    state = next_state

env.close()
