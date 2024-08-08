from policy_agent import PolicyGradientAgent
from environment import IntersectionEnv, four_way
from config import Config
from sim import run_simulation, initialize_plot, update_plot, save_and_plot_data
import matplotlib.pyplot as plt
from helper_functions import set_seed
import torch

seed = 43
set_seed(seed)

def train_agent(agent, env, num_episodes):
    fig, ax1, ax2, ax3 = initialize_plot()

    collision_records = []
    intersection_records = []
    reward_records = []
    parameter_records = []  # List to store parameters for each interval
    interval_count = 0

    for episode in range(num_episodes):
        # agent.log_parameters(episode)
        interval_results, bottom_top_next_interval, left_right_next_interval, total_reward, collision_records, intersection_records, reward_records, interval_count, parameter_records = run_simulation(env.config, agent, interval_count, collision_records, intersection_records, reward_records, parameter_records)

        # Store rewards collected during simulation
        for _, _, _, _, _, reward in interval_results:
            agent.store_reward(reward)
        # Update the policy at the end of each episode
        agent.update_policy()

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

        # Update the plot after each episode
        update_plot(collision_records, intersection_records, reward_records, fig, ax1, ax2, ax3)
        save_and_plot_data(collision_records, intersection_records, reward_records, parameter_records)

        # Save the model state dict every 33 episodes
        if (episode + 1) % 33 == 0:
            torch.save(agent.policy_net.state_dict(), f"policy_net_episode_{episode + 1}.pth")
            print(f"Saved model state dict at episode {episode + 1}")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    config = Config()
    env = IntersectionEnv(config, four_way, None)
    agent = PolicyGradientAgent(env)
    env.agent = agent

    train_agent(agent, env, num_episodes=1000)