import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
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
    def __init__(self, env, entropy_coeff=0.01): # Higher entropy coefficient means more exploration
        self.env = env
        self.policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.episode_log_probs = []
        self.episode_rewards = []
        self.entropy_coeff = entropy_coeff

    def select_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.policy_net(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.episode_log_probs.append((log_prob, dist.entropy()))  # store both log_prob and entropy
        return action.cpu().numpy()

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
            policy_loss.append(-log_prob * R - self.entropy_coeff * entropy)  # include entropy in loss

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.episode_log_probs = []
        self.episode_rewards = []

# Training code remains the same
import gym

def train_agent(agent, env, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.episode_rewards.append(reward)
            state = next_state
            total_reward += reward

        agent.update_policy()
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

if __name__ == "__main__":
    config = Config()
    env = IntersectionEnv(config, four_way)
    agent = PolicyGradientAgent(env)
    train_agent(agent, env, num_episodes=1000)
    env.render()