# -*- coding: utf-8 -*-
""" 
Save data from agent checkpoint
Pong: --games pong --save_dataset --model Pong-Scratch-Seed321/checkpoint_5000000.pth 
MsPacman: --games ms_pacman --model MsPacman-Scratch-Seed213/checkpoint_5000000.pth 
Qbert --games qbert --model Scratch-Qbert-Seed3/checkpoint_5000000.pth
"""
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

from agent import Agent
from agent_mt import MultiTaskAgent
from env import Env
from memory import ReplayMemory
from test import test, test_all_games

from env_mt import MultiTaskEnv
from omegaconf import DictConfig, OmegaConf
import pickle 

from os.path import join 

class Transition(object):

    def __init__(self, state, action,reward, done, info):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done 
        self.info = info


DATA_DIR = [
  '/home/mandi/Rainbow/results',
  '/shared/mandi/rainbow_data'
]

def main():

  

  # Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
  parser = argparse.ArgumentParser(description='Offline')
  
  parser.add_argument('--games', nargs='+', default=['pong'], help='Environment names')
  parser.add_argument('--model', default='model.pth', help='model file path')
  parser.add_argument('--save_dataset', action='store_true', help='save folder path')
  parser.add_argument('--seed', type=int, default=123, help='Random seed')
  # agent load
  parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
  parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient', 'data-effx2'], metavar='ARCH', help='Network architecture')
  parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
  parser.add_argument('--noisy_std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers') 
  parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
  parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
  parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
  parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='Batch size')
  parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
  parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
  parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
  parser.add_argument('--load_conv_only', action='store_true', help='Load convolutional layers only')
  parser.add_argument('--load_conv_fc_h', action='store_true', help='Load convolutional layers and hidden FC layers')
  parser.add_argument('--learning_rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
  parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
  parser.add_argument('--num_games_per_batch', type=int, default=1, help='Number of games in one update batch, if >1 games are used the default is num_games')
  # replay buffer 
  parser.add_argument('--greedy_eps', type=float, default=0.1, help='Act greedily every n steps')
  parser.add_argument('--reset_sigmas', action='store_true', help='Reset sigmas in Noisy Linear nets')
  parser.add_argument('--noiseless', action='store_true', help='Disable sigmas in Noisy Linear nets')

  # dataset save!
  parser.add_argument('--num_steps', default=int(1e5), type=int, help='dataset size')
  parser.add_argument('--save_path', default='/home/mandi/rainbow_data/', type=str)
  parser.add_argument('--save_name', default='rollouts', type=str)
  parser.add_argument('--inspect', action='store_true')

  # Setup
  args = parser.parse_args()
  if len(args.model.split('/')) == 2:
      for append_dir in DATA_DIR:
          full_path = os.path.join(append_dir, args.model) 
          if os.path.exists(full_path):
              print('Prepending model path to: {}'.format(full_path))
              args.model = full_path
              break
  np.random.seed(args.seed)
  torch.manual_seed(np.random.randint(1, 10000))
  assert torch.cuda.is_available(), 'Need GPU'
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(np.random.randint(1, 10000))
  torch.backends.cudnn.enabled = False


  cfg = OmegaConf.load('conf/config.yaml').env
  cfg.games = list(args.games)
  # cfg.modify_action_size = dqn.action_space
  games = sorted(cfg.games)

  env = MultiTaskEnv(cfg)
  env.eval()

  print('Loading agent checkpoint!', args.model)
  dqn = MultiTaskAgent(args, env)
  dqn.eval()
  if args.save_dataset:
    assert len(args.games) == 1
    save_name = str(args.games[0]) + '_' + args.save_name
    save_path = os.path.join(args.save_path, save_name)
    print('Saving dataset to path:', save_path)
    os.makedirs(save_path, exist_ok=True)

    done = True 
    step_count, episode_count = 0, 0
    while step_count < args.num_steps:
      if done:
        state, reward_sum, done = env.reset(resample_game=False), 0, False
        episode = [] 
      action = dqn.act_e_greedy(state, epsilon=0)
      next_state, reward, done, info = env.step(action)
      transition = Transition(state, action, reward, done, info)
      reward_sum += reward
      episode.append(transition)

      state = next_state
      if done:
        print('Episode {} finished after {} timesteps, reward {}'.format(
          episode_count, len(episode), reward_sum)
          )
        save_folder = join(save_path, f'episode{episode_count}-rew{int(reward_sum)}')
        os.makedirs(save_folder, exist_ok=True)
        for j, step in enumerate(episode):
          pickle.dump(step, open(join(save_folder, f'{j}.pkl'), 'wb'))
        episode_count += 1
        step_count += len(episode)
        print('Total step count:', step_count)

  elif args.inspect:
    done = True 
    step_count, episode_count, live_count = 0, 0
    state, reward_sum, done = env.reset(resample_game=False), 0, False
    while True:
      action = dqn.act_e_greedy(state, epsilon=0)
      next_state, reward, done, info = env.step(action) 
      reward_sum += reward 

    # done = True
    # for i in range(args.num_steps):
    #   episode = []
    #   while True:
    #     if done:
    #       state, reward_sum, done = env.reset(resample_game=False), 0, False
    #     action = dqn.act_e_greedy(state, epsilon=0)  # Choose an action ε-greedily
    #     state, reward, done, info = env.step(action)  # Step
    #     episode.append(
    #     Transition(state, action, reward, done, info)
    #     )
    #     reward_sum += reward
    #     if done:
    #       print('Episode {} finished after {} timesteps, reward {}'.format(i, len(episode), reward_sum))
    #       save_folder = join(save_path, f'episode{i}-rew{int(reward_sum)}')
    #       os.makedirs(save_folder, exist_ok=True)
    #       for j, step in enumerate(episode):
    #         pickle.dump(step, open(join(save_folder, f'{j}.pkl'), 'wb'))
    #       break
  else:
    print('Loading saved data back!')
    mem = ReplayMemory(args, int(1e6))

  env.close()


if __name__ == '__main__':
  main()