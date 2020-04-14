import gym
gym.logger.set_level(40) ## suppress warning
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
troch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import troch.optim as optim
from torch.distributions import Categorical

# Define the Architecture of the Policy
env = gym.make('CartPole-v0')
env.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
  def __init__(self, s_size = 4, h_size = 16, a_size=2):
    super().__init__()
    self.fc1  = nn.Linear(s_size,h_size)
    self.fc2 = nn.Linear(h_size,a_size)
  def forward(self,x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.softmax(x,dim=1)
  def act(self,state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = self.forward(state).cpu()
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)
  
# Train the agent with Reinforce
policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr = 1e-2)

def reinforce(n_episode=1000, max_t=1000, gamma = 1.0, print_every=100):
  scores_deque = deque(maxlen=100)
  scores = []
  for i_episode in range(1,n_episodes+1):
    saved_log_probs = []
    rewards = []
    state =  env.reset()
    for t in range(max_t):
      action,log_prob = policy.act(state)
      saved_log_probs.append(log_prob)
      state,reward,done,_ = env.step(action)
      rewards.append(reward* (gamma**t))
      if done:
        break
    scores_deque.append(sum(rewards))
    scores.append(sum(rewards))
    R = sum(rewards)
    policy_loss = []
    for log_prob in saved_log_probs:
        policy_loss.append(-log_prob[0]*R)
    policy_loss = sum(policy_loss)

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    if i_episode % print_every == 0:
        print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    if np.mean(scores_deque)>=195.0:
        print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
        break

  return scores
scores = reinforce()