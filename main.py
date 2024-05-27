import numpy as np
from agent import Agent
from visualization import plot_comparison_seaborn, plot_color_adjusted_grouped_selections_seaborn

def run_simulation(n_arms, epsilon_values, alpha, episodes, true_means, optimistic_initial=5.0):
    
    """
    Run the multi-armed bandit simulation with different epsilon-greedy agents, including an agent with optimistic initial values.

    Parameters:
        n_arms (int): Number of arms in the bandit.
        epsilon_values (list): List of epsilon values for different agents.
        alpha (float): Step size for updating estimates.
        episodes (int): Number of episodes to run the simulation.
        true_means (np.array): True reward means for each arm.
        optimistic_initial (float): Initial high estimate for the optimistic agent.

    Returns:
        tuple: Tuple containing arrays of rewards and selections for each agent.
    """
    
    agents = [Agent(n_arms, epsilon, alpha, true_means) for epsilon in epsilon_values]
    agents.append(Agent(n_arms, 0.1, alpha, true_means, optimistic_initial))  # Adding an optimistic agent
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
    optimistic_initial = 5.0  # High initial value for optimistic exploration

    rewards, selections = run_simulation(n_arms, epsilon_values, alpha, episodes, true_means, optimistic_initial)
    
    epsilon_labels = epsilon_values + [0.1]  # Including the optimistic agent's ε in the labels
    plot_comparison_seaborn(rewards, epsilon_labels)
    plot_color_adjusted_grouped_selections_seaborn(selections, epsilon_labels)
