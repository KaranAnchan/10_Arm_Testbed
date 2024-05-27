import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparison_seaborn(rewards, epsilon_values):
    
    """
    Plot the comparison of average rewards over episodes for different epsilon values using Seaborn.

    Parameters:
        rewards (np.array): Array containing rewards for each episode for each agent.
        epsilon_values (list): List of epsilon values used for different agents.
    """
    
    plt.figure(figsize=(12, 6))
    for i in range(rewards.shape[0]):
        sns.lineplot(np.arange(1, rewards.shape[1] + 1), np.cumsum(rewards[i]) / np.arange(1, rewards.shape[1] + 1), label=f'ε={epsilon_values[i]}')
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.title('Comparison of ε-Greedy Agents Using Seaborn', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_color_adjusted_grouped_selections_seaborn(selections, epsilon_values):
    
    """
    Plot the number of times each arm was selected by each agent in a grouped bar chart with adjusted colors for better distinction using Seaborn.

    Parameters:
        selections (np.array): Array containing the count of selections for each arm for all agents.
        epsilon_values (list): List of ε-values used in the simulation to label the agents.
    """
    
    plt.figure(figsize=(14, 8))
    arms = np.arange(selections.shape[1])  # Arms are the base for the x-axis
    width = 0.25  # Width of each bar, optimally adjusted for visibility and contrast
    palette = sns.color_palette("Set2", len(epsilon_values))
    
    for i, agent_selections in enumerate(selections):
        plt.bar(arms + i * width, agent_selections, width=width, label=f'ε={epsilon_values[i]}', color=palette[i])
    
    plt.xlabel('Arms', fontsize=14)
    plt.ylabel('Number of times selected', fontsize=14)
    plt.title('Color Adjusted Grouped Bar Chart of Arm Selections by ε-Greedy Agents', fontsize=16)
    plt.xticks(arms + width, np.arange(selections.shape[1]))  # Centering the x-ticks between the grouped bars
    plt.legend(title='Agent ε-value')
    plt.show()