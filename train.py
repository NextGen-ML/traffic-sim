from policy_agent import PolicyGradientAgent
from environment import IntersectionEnv, four_way
from config import Config

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
    env = IntersectionEnv(config, four_way, None)  # Temporarily pass None for agent
    agent = PolicyGradientAgent(env)  
    env.agent = agent 

    train_agent(agent, env, num_episodes=1000)
    env.render()