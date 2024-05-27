import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_comparison_seaborn(rewards, labels):
    
    """
    Plot the comparison of average rewards over episodes for different agents using Seaborn.

    Parameters:
        rewards (np.array): Array containing rewards for each episode for each agent.
        labels (list): List of labels for different agents.
    """
    
    plt.figure(figsize=(12, 6))
    episodes = rewards.shape[1]
    for i, label in enumerate(labels):
        sns.lineplot(np.arange(1, episodes + 1), np.cumsum(rewards[i]) / np.arange(1, episodes + 1), label=label)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.title('Average Reward vs. Episodes', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_color_adjusted_grouped_selections_seaborn(selections, labels):
    
    """
    Plot the number of times each arm was selected by each agent in a grouped bar chart using Seaborn.

    Parameters:
        selections (np.array): Array containing the count of selections for each arm for all agents.
        labels (list): List of labels for different agents.
    """
    
    n_agents, n_arms = selections.shape
    indices = np.arange(n_arms)
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(n_agents):
        ax.bar(indices + i * bar_width, selections[i], bar_width, label=labels[i])
    
    ax.set_xlabel('Arms', fontsize=14)
    ax.set_ylabel('Number of Selections', fontsize=14)
    ax.set_title('Selections of Each Arm', fontsize=16)
    ax.set_xticks(indices + bar_width * (n_agents - 1) / 2)
    ax.set_xticklabels([f'Arm {i}' for i in range(n_arms)])
    ax.legend()
    plt.grid(True)
    plt.show()

def plot_optimistic_vs_ucb(rewards, labels):
    
    """
    Plot the comparison of average rewards over episodes between Optimistic and UCB agents using Seaborn.

    Parameters:
        rewards (np.array): Array containing rewards for each episode for the Optimistic and UCB agents.
        labels (list): List of labels for the Optimistic and UCB agents.
    """
    
    plt.figure(figsize=(12, 6))
    episodes = rewards.shape[1]
    for i, label in enumerate(labels):
        sns.lineplot(np.arange(1, episodes + 1), np.cumsum(rewards[i]) / np.arange(1, episodes + 1), label=label)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.title('Comparison between Optimistic and UCB Agents', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()
