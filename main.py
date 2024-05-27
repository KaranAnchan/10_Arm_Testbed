import numpy as np
from agent import Agent
from visualization import plot_rewards, plot_comparison, plot_selections

def run_simulation(n_arms, epsilon_values, alpha, episodes, true_means):
    """
    Runs a simulation for multiple ε-greedy agents with varying epsilon values.
    
    Parameters:
        n_arms (int): Number of arms in the bandit.
        epsilon_values (list of float): Epsilon values for each agent to test.
        alpha (float): Step size for updating estimates.
        episodes (int): Number of episodes to run.
        true_means (np.array): True mean rewards for each arm.
        
    Returns:
        np.array: Rewards matrix with each row representing an agent's rewards over episodes.
    """
    agents = [Agent(n_arms, epsilon, alpha, true_means) for epsilon in epsilon_values]
    rewards = np.zeros((len(agents), episodes))
    selections = np.zeros((len(agents), n_arms))

    for episode in range(episodes):
        for index, agent in enumerate(agents):
            arm = agent.choose_arm()
            selections[index, arm] += 1
            reward = agent.get_reward(arm)
            agent.update_estimates(arm, reward)
            rewards[index, episode] = reward

    return rewards, selections

def main():
    # Parameters
    n_arms = 10
    epsilon_values = [0.01, 0.1, 0.2]  # Different epsilon values for comparison
    alpha = 0.1
    episodes = 1000
    true_means = np.random.randn(n_arms)

    # Run simulation
    rewards, selections = run_simulation(n_arms, epsilon_values, alpha, episodes, true_means)

    # Visualization
    plot_rewards(rewards[-1])  # Plot rewards for the last agent for consistency
    plot_comparison(rewards, epsilon_values)
    plot_selections(selections[-1])  # Plot selections for the last agent for consistency

if __name__ == "__main__":
    main()