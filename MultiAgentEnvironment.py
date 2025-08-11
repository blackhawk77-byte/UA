import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import List, Tuple, Dict, Optional
import copy

class MultiAgentEnvironment:
   def __init__(self, num_ues=9, num_sbs=3, max_beams_per_sbs=None):
       self.num_ues = num_ues
       self.num_sbs = num_sbs
       self.num_bs = num_sbs + 1
       if max_beams_per_sbs is None:
           max_beams_per_sbs = [2, 3, 3]
       self.max_beams = [float('inf')] + max_beams_per_sbs
       self.agents = []
       for i in range(num_ues):
           agent = HystereticDRQNAgent(agent_id=i, input_size=6, action_size=self.num_bs, beta=0.5 if num_ues <= 9 else 0.3)
           self.agents.append(agent)
   def check_collision(self, actions):
       bs_requests = [0] * self.num_bs
       for action in actions:
           bs_requests[action] += 1
       for bs_id in range(1, self.num_bs):
           if bs_requests[bs_id] > self.max_beams[bs_id]:
               return True
       return False
   def compute_reward(self, network_sum_rate, collision):
       if collision:
           return 0.0
       return network_sum_rate
   def distributed_training_step(self, observations, available_actions=None):
       actions = []
       for i, agent in enumerate(self.agents):
           action = agent.select_action(observations[i], available_actions[i] if available_actions else None)
           actions.append(action)
       return actions
   def store_experiences(self, observations, actions, rewards, next_observations, acks):
       for i, agent in enumerate(self.agents):
           agent.store_experience(observations[i], actions[i], rewards[i], next_observations[i], acks[i])
   def train_agents(self, batch_size=32):
       losses = []
       for agent in self.agents:
           loss = agent.train(batch_size)
           losses.append(loss)
       return losses