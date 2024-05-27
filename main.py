import numpy as np
from agent import Agent, UCB_Agent
from visualization import plot_comparison_seaborn, plot_color_adjusted_grouped_selections_seaborn, plot_optimistic_vs_ucb

def run_simulation(n_arms, epsilon_values, alpha, episodes, true_means, optimistic_initial=5.0, c=2):
    
    """
    Run the multi-armed bandit simulation with different agents including ε-greedy agents, 
    an optimistic initial values agent, and a UCB agent.

    Parameters:
        n_arms (int): Number of arms in the bandit.
        epsilon_values (list): List of epsilon values for ε-greedy agents.
        alpha (float): Step size for updating estimates.
        episodes (int): Number of episodes to run the simulation.
        true_means (np.array): True reward means for each arm.
        optimistic_initial (float): Initial value for optimistic agent's estimates.
        c (float): Confidence level for the UCB agent.

    Returns:
        tuple: A tuple containing arrays of rewards and selections for each agent.
    """
    
    agents = [Agent(n_arms, epsilon, alpha, true_means) for epsilon in epsilon_values]
    agents.append(Agent(n_arms, 0.1, alpha, true_means, optimistic_initial))  # Optimistic agent
    agents.append(UCB_Agent(n_arms, true_means, c))  # UCB agent
    rewards = np.zeros((len(agents), episodes))
    selections = np.zeros((len(agents), n_arms))

    for episode in range(episodes):
        for index, agent in enumerate(agents):
            arm = agent.choose_arm()
            reward = agent.get_reward(arm)
            agent.update_estimates(arm, reward)
            rewards[index, episode] = reward
            selections[index, arm] += 1

    return rewards, selections

if __name__ == "__main__":
    
    n_arms = 10
    epsilon_values = [0.01, 0.1, 0.2]
    alpha = 0.1
    episodes = 1000
    true_means = np.random.normal(0, 1, n_arms)
    optimistic_initial = 5.0

    # Running the simulation
    rewards, selections = run_simulation(n_arms, epsilon_values, alpha, episodes, true_means, optimistic_initial, c=2)

    # Labels for agents
    epsilon_labels = epsilon_values + ['Optimistic ε=0.1', 'UCB']

    # Plotting the results
    plot_comparison_seaborn(rewards, epsilon_labels)
    plot_color_adjusted_grouped_selections_seaborn(selections, epsilon_labels)
    plot_optimistic_vs_ucb(rewards[-2:], ['Optimistic ε=0.1', 'UCB'])
