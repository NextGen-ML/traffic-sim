import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from environment import IntersectionEnv, four_way
from config import Config
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, output_size)
        self.log_std = nn.Parameter(torch.zeros(output_size))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.elu(self.ln1(self.fc1(x)))
        x = F.elu(self.ln2(self.fc2(x)))
        mu = F.tanh(self.fc3(x))
        std = F.softplus(self.log_std).expand_as(mu)
        return mu, std

class PolicyGradientAgent:
    def __init__(self, env, entropy_coeff=0.01):
        self.env = env
        self.policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003)
        self.episode_log_probs = []
        self.episode_rewards = []
        self.entropy_coeff = entropy_coeff

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mu, std = self.policy_net(state)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        self.episode_log_probs.append((log_prob, dist.entropy().sum(dim=-1)))
        
        # Scale the action from [-1, 1] to the action space range
        action = action.squeeze(0).detach().numpy()
        low = self.env.action_space.low
        high = self.env.action_space.high
        scaled_action = low + (0.5 * (action + 1.0) * (high - low))
        
        # Clip the scaled action to ensure it's within the action space
        clipped_action = np.clip(scaled_action, low, high)
        
        return clipped_action

    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []

        for r in self.episode_rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        for (log_prob, entropy), R in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * R - self.entropy_coeff * entropy)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.episode_log_probs = []
        self.episode_rewards = []