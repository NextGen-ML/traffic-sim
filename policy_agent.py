import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from environment import IntersectionEnv, four_way
from config import Config

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PolicyGradientAgent:
    def __init__(self, env, entropy_coeff=0.01):
        self.env = env
        self.policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.episode_log_probs = []
        self.episode_rewards = []
        self.entropy_coeff = entropy_coeff

    def select_action(self, state):
        state = torch.FloatTensor(state)
        mu = self.policy_net(state)
        sigma = torch.ones_like(mu) * 0.1  # Fixed standard deviation
        dist = Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        self.episode_log_probs.append((log_prob, dist.entropy().sum(dim=-1)))
        return action.detach().numpy()

    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []

        for r in self.episode_rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        for (log_prob, entropy), R in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * R - self.entropy_coeff * entropy)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.episode_log_probs = []
        self.episode_rewards = []