import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym

class Policy(nn.Module):

    def __init__(self, num_obs, hidden_dim, num_act):
        super(Policy, self).__init__()
        self.input_layer = nn.Linear(num_obs, hidden_dim)
        self.tanh = nn.Tanh()
        self.output_layer = nn.Linear(hidden_dim, num_act)

    def forward(self, obs):
        h = self.input_layer(obs)
        h = self.tanh(h)
        logits = self.output_layer(h)
        return logits

    def get_policy(self, obs):
        logits = self(obs)
        return Categorical(logits = logits)

    def get_action(self, obs):
        return self.get_policy(obs).sample().item()

def train(epochs, agent, env):

    for epoch in range(epochs):
        batch_obs, batch_act, batch_weight, eps_rew = [], [], [], []
        while len(batch_obs) < 5000:
            obs = env.reset()
            while True:
                batch_obs.append(torch.as_tensor(obs))
                act = agent.get_action(obs)
                obs_next, rew, done, truncated, _ = env.step(act)
                batch_act.append(torch.as_tensor(act))
                eps_rew.append(rew)
                if done:
                    eps_return, eps_len = sum(eps_rew), len(eps_rew)
                    

