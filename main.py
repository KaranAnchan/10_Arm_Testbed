import numpy as np
from agent import Agent
from visualization import plot_comparison_seaborn, plot_color_adjusted_grouped_selections_seaborn

def run_simulation(n_arms, epsilon_values, alpha, episodes, true_means):
    
    """
    Run the multi-armed bandit simulation with different epsilon-greedy agents.

    Parameters:
        n_arms (int): Number of arms in the bandit.
        epsilon_values (list): List of epsilon values for different agents.
        alpha (float): Step size for updating estimates.
        episodes (int): Number of episodes to run the simulation.
        true_means (np.array): True reward means for each arm.

    Returns:
        tuple: Tuple containing arrays of rewards and selections for each agent.
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

if __name__ == "__main__":
    
    n_arms = 10
    epsilon_values = [0.01, 0.1, 0.2]
    alpha = 0.1
    episodes = 1000
    true_means = np.random.randn(n_arms)

    rewards, selections = run_simulation(n_arms, epsilon_values, alpha, episodes, true_means)
    plot_comparison_seaborn(rewards, epsilon_values)
    plot_color_adjusted_grouped_selections_seaborn(selections, epsilon_values)
