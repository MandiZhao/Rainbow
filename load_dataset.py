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
from glob import glob
from natsort import natsorted
  

class Transition(object):

    def __init__(self, state, action,reward, done, info):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done 
        self.info = info
def main(): 
  parser = argparse.ArgumentParser(description='Offline')
  parser.add_argument('--game', default='pong', type=str)
  parser.add_argument('--data_path', default='', type=str)

  args = parser.parse_args()
  dataset_path = join(args.data_path, f'{args.game}_rollouts')
  print('trying to load from data set dir: ', dataset_path)

  dataset = []
  episode_paths = glob( join(dataset_path, 'episode*'))
  print(f'Loading {len(episode_paths)} episodes')
  for i, eps in enumerate(episode_paths):
    episode = []
    steps = natsorted(glob(join(eps + '/*.pkl')))
    for step in steps:
      # pickle load step
      with open(step, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        episode.append(data.state, data.action, data.reward, data.done)
    if i % 10 == 0:
      print('Done loading episode {}'.format(i))
    dataset.append(episode)
  print('Done loading dataset')

if __name__ == '__main__':
  main()