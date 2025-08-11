import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import List, Tuple, Dict, Optional
import copy

class HystereticDRQNAgent:
   def __init__(self, agent_id, input_size=6, action_size=4, learning_rate=0.001, alpha=1.0, beta=0.5, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, target_update_freq=10, device='cpu'):
       self.agent_id = agent_id
       self.action_size = action_size
       self.device = torch.device(device)
       self.alpha = alpha
       self.beta = beta
       self.gamma = gamma
       self.learning_rate = learning_rate
       self.epsilon = epsilon_start
       self.epsilon_end = epsilon_end
       self.q_network = DuelingDRQN(input_size, action_size=action_size).to(self.device)
       self.target_network = DuelingDRQN(input_size, action_size=action_size).to(self.device)
       self.target_network.load_state_dict(self.q_network.state_dict())
       self.target_update_freq = target_update_freq
       self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
       self.memory = CERTReplayBuffer(capacity=500)
       self.hidden_state = None
       self.step_count = 0
   def get_observation_vector(self, prev_action, prev_rate, prev_network_rate, prev_ack, current_demand, current_rssi):
       return np.array([prev_action, prev_rate, prev_network_rate, prev_ack, current_demand, current_rssi], dtype=np.float32)
   def select_action(self, observation, available_actions=None):
       self._update_epsilon()
       if available_actions is None:
           available_actions = list(range(self.action_size))
       if random.random() < self.epsilon:
           return random.choice(available_actions)
       else:
           obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
           with torch.no_grad():
               q_values, self.hidden_state = self.q_network(obs_tensor, self.hidden_state)
           q_values_np = q_values.cpu().numpy()[0]
           masked_q_values = np.full(self.action_size, -np.inf)
           for action in available_actions:
               masked_q_values[action] = q_values_np[action]
           return np.argmax(masked_q_values)
   def _update_epsilon(self):
       a, b, c = 0.9, 0.001, 800
       t = self.step_count
       self.epsilon = max(self.epsilon_end, 1 - a * np.exp(-b * max(0, t - c)))
   def store_experience(self, observation, action, reward, next_observation, ack):
       experience = Experience(observation, action, reward, next_observation, ack)
       self.memory.push(experience)
   def train(self, batch_size=32):
       if len(self.memory) < batch_size:
           return 0.0
       batch = self.memory.sample(batch_size)
       observations = torch.FloatTensor([e.observation for e in batch]).to(self.device)
       actions = torch.LongTensor([e.action for e in batch]).to(self.device)
       rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
       next_observations = torch.FloatTensor([e.next_observation for e in batch]).to(self.device)
       current_q_values, _ = self.q_network(observations)
       current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
       with torch.no_grad():
           next_q_values, _ = self.target_network(next_observations)
           max_next_q_values = next_q_values.max(1)[0]
           target_q_values = rewards + self.gamma * max_next_q_values
       td_errors = target_q_values - current_q_values
       weights = torch.where(td_errors >= 0, torch.tensor(self.alpha).to(self.device), torch.tensor(self.beta).to(self.device))
       loss = torch.mean((weights * td_errors) ** 2)
       self.optimizer.zero_grad()
       loss.backward()
       self.optimizer.step()
       self.step_count += 1
       if self.step_count % self.target_update_freq == 0:
           self.target_network.load_state_dict(self.q_network.state_dict())
       return loss.item()
   def reset_hidden_state(self):
       self.hidden_state = None