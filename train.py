from policy_agent import PolicyGradientAgent
from environment import IntersectionEnv, four_way
from config import Config
from sim import initialize_plot, update_plot, save_and_plot_data
import matplotlib.pyplot as plt

def train_agent(agent, env, num_episodes):
    initialize_plot()

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        try:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.episode_rewards.append(reward)
            state = next_state
            total_reward += reward

            agent.update_policy()  # Update policy after the interval
            print("updated")

            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

            # Update the plot after each episode
            save_and_plot_data()

        except ValueError as e:
            print(f"Error in episode {episode + 1}: {str(e)}")
            print(f"Current state: {state}")
            break

    plt.ioff()  
    plt.show()

if __name__ == "__main__":
    config = Config()
    env = IntersectionEnv(config, four_way, None)
    agent = PolicyGradientAgent(env)  
    env.agent = agent 

    train_agent(agent, env, num_episodes=1000)