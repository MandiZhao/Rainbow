# -*- coding: utf-8 -*-
"""
['pong', 'breakout', 'asteroids', 'qbert']
"""
import gym 
from collections import deque
import numpy as np
import random
import atari_py
import cv2
import torch
import copy 
import hydra 
from omegaconf import DictConfig, OmegaConf, ListConfig

ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}
class MultiTaskEnv():
  def __init__(self, cfg):
    self.device = cfg.device
    self.ale = atari_py.ALEInterface()
    self.ale.setInt('random_seed', cfg.seed)
    self.ale.setInt('max_num_frames_per_episode', cfg.max_episode_length)
    self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
    self.ale.setInt('frame_skip', 0)
    self.ale.setBool('color_averaging', False)
    
    self.games = sorted(list(cfg.games))
    self.num_games = len(self.games)

    invalid_default_action = int(cfg.get('invalid_default_action', 0))
    game_to_actions = []
    tmp_ale = self.ale # copy.deepcopy(self.ale)
    for g in self.games: 
      assert g in atari_py.list_games(), 'Unknown game: {}'.format(g)
      tmp_ale.loadROM(atari_py.get_game_path(g))
      game_to_actions.append(tmp_ale.getMinimalActionSet())
 
    max_action_set = max([len(a) for a in game_to_actions] + [int(cfg.modify_action_size)])
    # if cfg.modify_action_size > 0:
    #   assert cfg.modify_action_size >= max_action_set, 'Modify action size must be at least as big as max action size'
    #   max_action_set = maxcfg.modify_action_size
    # print('Padding multi-task env with max action set size: {}'.format(max_action_set))
    self.game_to_actions = []
    for a in game_to_actions:
      actions = dict([i, e] for i, e in zip(range(len(a)), a))
      for j in range(len(a), max_action_set):
        actions[j] = invalid_default_action
      self.game_to_actions.append(actions)
    self.max_action_size = max_action_set
      
    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = cfg.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=cfg.history_length)
    self._resample_game()
    self.training = True  # Consistent with model training mode

  def _resample_game(self):
    assert len(self.games) > 0, 'No games to sample from'
    self.current_id = np.random.choice(self.num_games)
    self.current_game = self.games[self.current_id]
    # print('sampling game: {}'.format(self.current_game), atari_py.get_game_path(self.current_game))
    # print(self.ale, dir(self.ale))
    self.ale.loadROM(atari_py.get_game_path(self.current_game))  # ROM loading must be done after setting options
    
    # print('sampling game: {}'.format(self.current_game))
    self.actions = self.game_to_actions[self.current_id]
    for a in self.ale.getMinimalActionSet():
      assert a in self.actions.values(), 'Action from game env. not stored: {}'.format(a)
    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.state_buffer = deque([], maxlen=self.window)

  def _set_game(self, game_id):
    assert game_id < self.num_games, f'Game id {game_id} out of range {self.num_games}'
    self.current_id = game_id
    self.current_game = self.games[game_id]
    self.ale.loadROM(atari_py.get_game_path(self.current_game))  # ROM loading must be done after setting options
    self.actions = self.game_to_actions[self.current_id]
    for a in self.ale.getMinimalActionSet():
      assert a in self.actions.values(), 'Action from game env. not stored: {}'.format(a)
    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.state_buffer = deque([], maxlen=self.window)

  def _get_state(self):
    state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84, device=self.device))

  def reset(self, resample_game=True):
    if self.life_termination:
      self.life_termination = False  # Reset flag
      self.ale.act(0)  # Use a no-op after loss of life
    else:
      # Reset internals
      if resample_game:
        self._resample_game()
      self._reset_buffer() 
      self.ale.reset_game()
      # Perform up to 30 random no-ops before starting
      for _ in range(random.randrange(30)):
        self.ale.act(0)  # Assumes raw action 0 is always no-op
        if self.ale.game_over():
          self.ale.reset_game()
    # Process and return "initial" state
    observation = self._get_state()
    self.state_buffer.append(observation)
    self.lives = self.ale.lives()
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, 84, 84, device=self.device)
    reward, done = 0, False
    for t in range(4):
      reward += self.ale.act(self.actions.get(action))
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      done = self.ale.game_over()
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    # Detect loss of life as terminal in training mode
    if self.training:
      lives = self.ale.lives()
      if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
        self.life_termination = not done  # Only set flag when not truly done
        done = True
      self.lives = lives
    # Return state, reward, done
    info = {
      'game_id': self.current_id,
      'game_name': self.current_game,
      'lives': self.lives,
    }
    return torch.stack(list(self.state_buffer), 0), reward, done, info

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return self.max_action_size
    # return len(self.actions)

  def get_current_game_id(self):
    return self.current_id

  def render(self):
    cv2.imshow('screen', self.ale)
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()


if __name__ == '__main__':
  # cfg = OmegaConf.load('conf/config.yaml')
  # print(cfg)
  # print(cfg.env.history_length)
  # env = MultiTaskEnv(cfg.env)
  # print(env.action_space())
  # for _ in range(10):
  #   obs = env.reset()

  #   for _ in range(10):
  #     env.step(1)
  #     print(env.current_game, env.ale.lives())
  import matplotlib.pyplot as plt
  games = ['asteroids', 'alien', 'beam_rider', 'breakout', 'frostbite', 'krull', 'road_runner', 'seaquest'] + \
          ['amidar', 'assault', 'asterix', 'bank_heist', 'boxing', 'chopper_command', 'road_runner', 'crazy_climber'] + \
          ['demon_attack', 'freeway', 'gopher', 'hero']

  # fig, axs = plt.subplots(5, 4, figsize=(50, 50))
  games = ['asteroids', 'alien', 'beam_rider', 'breakout', 'frostbite'] 
  games = ['krull', 'road_runner', 'seaquest', 'amidar', 'assault'] 
  games = ['asterix', 'bank_heist', 'boxing', 'chopper_command', 'private_eye']
  games = ['crazy_climber', 'kung_fu_master', 'freeway', 'gopher', 'hero']

  # test demon_attack (similar to assault), kangaroo, similar to amidar
  games = ['jamesbond', 'kangaroo', 'demon_attack']

  games = ['robotank', 'enduro', 'battle_zone', 'solaris', 'beam_rider']
 
 
  games = ['pong', 'assault', 'battle_zone', 'ms_pacman', 'beam_rider']
  # need training game to pad: 
  # pong: 6, assault: 7, battle_zone: 18, ms_pacman: 9, beam_rider: 9
  
  games = games + ['breakout', 'demon_attack', 'robotank', 'bank_heist', 'solaris']
  obs_all = []
  for i, name in enumerate(games):
    env = atari_py.ALEInterface()
    env.loadROM(atari_py.get_game_path(name))
    env.reset_game() # getScreenRGB()[:, :, ::-1] # (210, 160, 3)
    print(name, env.getMinimalActionSet())
    obs = env.getScreenGrayscale()[:,:,0]
    # plt.imsave('pngs/{}_gray.png'.format(name), obs, cmap='gray')
    obs_all.append(np.ones((210, 40)) * 255)
    obs_all.append(obs)

  obs_all = np.concatenate(obs_all, axis=1)
  #plt.imsave(f"latest_1task_train-{'-'.join(games)}.png", obs_all, cmap='gray')

