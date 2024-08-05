from policy_agent import PolicyGradientAgent
from environment import IntersectionEnv, four_way
from config import Config
from sim import initialize_plot, update_plot, save_and_plot_data
import matplotlib.pyplot as plt
from helper_functions import set_seed

seed = 42
set_seed(seed)

def train_agent(agent, env, num_episodes):
    initialize_plot()

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

        # Update the plot after each episode
        save_and_plot_data()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    config = Config()
    env = IntersectionEnv(config, four_way, None)
    agent = PolicyGradientAgent(env)
    env.agent = agent

    train_agent(agent, env, num_episodes=1000)