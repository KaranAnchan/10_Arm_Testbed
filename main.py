import numpy as np
from agent import Agent
from visualization import plot_rewards

def main():
    
    """
    Main function to run the ε-greedy 10-Arm Testbed simulation.
    Sets up the environment, runs the simulation, and invokes plotting functions.
    """
    
    # Parameters
    n_arms = 10
    episodes = 1000
    epsilon = 0.1
    alpha = 0.1
    
    # Initialization
    true_means = np.random.randn(n_arms)
    agent = Agent(n_arms, epsilon, alpha, true_means)
    
    # Simulation
    rewards = []
    for _ in range(episodes):
        arm = agent.choose_arm()
        reward = agent.get_reward(arm)
        agent.update_estimates(arm, reward)
        rewards.append(reward)
        
    # Visualization
    plot_rewards(rewards)
        
if __name__ == "__main__":
    main()