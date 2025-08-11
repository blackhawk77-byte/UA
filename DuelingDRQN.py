import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import List, Tuple, Dict, Optional
import copy

Experience = namedtuple('Experience', ['observation', 'action', 'reward', 'next_observation', 'ack'])

class DuelingDRQN(nn.Module):
   def __init__(self, input_size=6, hidden_size=32, lstm_hidden_size=64, action_size=4, dueling_hidden_size=16):
       super(DuelingDRQN, self).__init__()
       self.fc1 = nn.Linear(input_size, hidden_size)
       self.fc2 = nn.Linear(hidden_size, hidden_size)
       self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, batch_first=True)
       self.lstm_hidden_size = lstm_hidden_size
       self.fc3 = nn.Linear(lstm_hidden_size, hidden_size)
       self.fc4 = nn.Linear(hidden_size, hidden_size)
       self.value_fc = nn.Linear(hidden_size, dueling_hidden_size)
       self.value_out = nn.Linear(dueling_hidden_size, 1)
       self.advantage_fc = nn.Linear(hidden_size, dueling_hidden_size)
       self.advantage_out = nn.Linear(dueling_hidden_size, action_size)
       self.action_size = action_size
       
   def forward(self, x, hidden_state=None):
       if len(x.shape) == 2:
           x = x.unsqueeze(1)
       batch_size = x.shape[0]
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       if hidden_state is None:
           h0 = torch.zeros(1, batch_size, self.lstm_hidden_size).to(x.device)
           c0 = torch.zeros(1, batch_size, self.lstm_hidden_size).to(x.device)
           hidden_state = (h0, c0)
       lstm_out, new_hidden = self.lstm(x, hidden_state)
       x = F.relu(self.fc3(lstm_out))
       x = F.relu(self.fc4(x))
       value = F.relu(self.value_fc(x))
       value = self.value_out(value)
       advantage = F.relu(self.advantage_fc(x))
       advantage = self.advantage_out(advantage)
       q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
       if x.shape[1] == 1:
           q_values = q_values.squeeze(1)
       return q_values, new_hidden