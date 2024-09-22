import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.log_std = nn.Parameter(torch.full((output_size,), -1.0))

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        mu = F.tanh(self.fc2(x) * 0.035)
        std = F.softplus(self.log_std).expand_as(mu)
        return mu, std

class PolicyGradientAgent:
    def __init__(self, env, entropy_coeff=0.01, gamma=0.65, learning_rate=0.002, entropy_decay=0.975):
        self.env = env
        self.policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.65)
        self.episode_log_probs = []
        self.entropy_coeff = entropy_coeff
        self.entropy_decay = entropy_decay
        self.gamma = gamma
        self.rewards = []
        self.update_count = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mu, std = self.policy_net(state)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        self.episode_log_probs.append((log_prob, entropy))
        action = action.squeeze(0).detach().numpy()

        low = self.env.action_space.low
        high = self.env.action_space.high
        scaled_action = low + (action + 1.0) * 0.5 * (high - low)
        clipped_action = np.clip(scaled_action, low, high)
        return clipped_action

    def update_policy(self):
        R = 0
        returns = []

        # Calculate the discounted returns
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        # Calculate policy loss
        policy_losses = []
        for (log_prob, entropy), R in zip(self.episode_log_probs, returns):
            policy_losses.append(-log_prob * R - self.entropy_coeff * entropy)
        policy_loss = torch.mean(torch.stack(policy_losses))

        print(policy_loss)

        # Backpropagation and optimization step
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.5)
        self.optimizer.step()
        self.scheduler.step()

        # Decay the entropy coefficient
        self.decay_entropy_coeff()

        # Clear interval data after update
        self.episode_log_probs = []
        self.rewards = []

    def store_reward(self, reward):
        self.rewards.append(reward)

    def decay_entropy_coeff(self):
        self.update_count += 1
        self.entropy_coeff *= self.entropy_decay