import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_rewards(rewards):
    
    """
    Plot the average reward over episodes using Seaborn.
    
    Parameters:
        rewards (np.array): Rewards array obtained over all episodes for an agent.
    """
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=np.arange(1, len(rewards) + 1), y=np.cumsum(rewards) / np.arange(1, len(rewards) + 1))
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.title('Average Reward vs. Episodes for ε-Greedy Agent', fontsize=16)
    plt.grid(True)
    plt.show()

def plot_comparison(rewards, epsilon_values):
    
    """
    Plot the average rewards for agents with different ε-values to compare their performance using Seaborn.
    
    Parameters:
        rewards (np.array): Matrix of rewards, where each row corresponds to a different ε-agent.
        epsilon_values (list): List of ε-values used in the simulation.
    """
    
    plt.figure(figsize=(12, 6))
    for i in range(rewards.shape[0]):
        sns.lineplot(x=np.arange(1, rewards.shape[1] + 1), y=np.cumsum(rewards[i]) / np.arange(1, rewards.shape[1] + 1), label=f'ε={epsilon_values[i]}')
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.title('Comparison of ε-Greedy Agents', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_selections(selections):
    
    """
    Plot the number of times each arm was selected by an agent using Seaborn.
    
    Parameters:
        selections (np.array): Array containing the count of selections for each arm.
    """
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=np.arange(len(selections)), y=selections, palette="viridis")
    plt.xlabel('Arms', fontsize=14)
    plt.ylabel('Number of times selected', fontsize=14)
    plt.title('Selections of Each Arm', fontsize=16)
    plt.show()
