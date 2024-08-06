from policy_agent import PolicyGradientAgent
from environment import IntersectionEnv, four_way
from config import Config
from sim import run_simulation, initialize_plot, update_plot, save_and_plot_data
import matplotlib.pyplot as plt
from helper_functions import set_seed

seed = 1
set_seed(seed)

def train_agent(agent, env, num_episodes):
    fig, ax1, ax2, ax3 = initialize_plot()

    collision_records = []
    intersection_records = []
    reward_records = []
    interval_count = 0

    for episode in range(num_episodes):
        interval_results, total_reward, collision_records, intersection_records, reward_records, interval_count = run_simulation(
            env.config, agent, interval_count, collision_records, intersection_records, reward_records)

        # Store rewards collected during simulation
        for _, _, _, _, _, reward in interval_results:
            agent.store_reward(reward)
        
        # Update the policy at the end of each episode
        agent.update_policy()

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

        # Update the plot after each episode
        update_plot(collision_records, intersection_records, reward_records, fig, ax1, ax2, ax3)
        save_and_plot_data(collision_records, intersection_records, reward_records)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    config = Config()
    env = IntersectionEnv(config, four_way, None)
    agent = PolicyGradientAgent(env)
    env.agent = agent

    train_agent(agent, env, num_episodes=1000)