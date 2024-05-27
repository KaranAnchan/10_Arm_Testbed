import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_comparison_seaborn(rewards, epsilon_values):
    
    """
    Plot the comparison of average rewards over episodes for different epsilon values using Seaborn, including any agents with optimistic initial values.

    Parameters:
        rewards (np.array): Array containing rewards for each episode for each agent.
        epsilon_values (list): List of epsilon values used for different agents, including an extra label for the optimistic agent.
    """
    
    plt.figure(figsize=(12, 6))
    episodes = rewards.shape[1]
    for i in range(rewards.shape[0]):
        label = f'ε={epsilon_values[i]}' if i < len(epsilon_values) - 1 else 'Optimistic ε=0.1'
        sns.lineplot(np.arange(1, episodes + 1), np.cumsum(rewards[i]) / np.arange(1, episodes + 1), label=label)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.title('Comparison of ε-Greedy Agents Using Seaborn', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_color_adjusted_grouped_selections_seaborn(selections, epsilon_values):
    
    """
    Plot the number of times each arm was selected by each agent in a grouped bar chart using Seaborn, including agents with optimistic initial values.

    Parameters:
        selections (np.array): Array containing the count of selections for each arm for all agents.
        epsilon_values (list): List of ε-values used in the simulation to label the agents, including an extra label for the optimistic agent.
    """
    
    plt.figure(figsize=(14, 8))
    arms = np.arange(selections.shape[1])
    width = 0.25  # Width of each bar, optimally adjusted for visibility and contrast
    palette = sns.color_palette("Set2", len(epsilon_values))
    
    for i, agent_selections in enumerate(selections):
        label = f'ε={epsilon_values[i]}' if i < len(epsilon_values) - 1 else 'Optimistic ε=0.1'
        plt.bar(arms + i * width, agent_selections, width=width, label=label, color=palette[i])
    
    plt.xlabel('Arms', fontsize=14)
    plt.ylabel('Number of times selected', fontsize=14)
    plt.title('Grouped Bar Chart of Arm Selections by ε-Greedy Agents', fontsize=16)
    plt.xticks(arms + width * len(epsilon_values) / 2, arms)
    plt.legend(title='Agent ε-value')
    plt.show()
