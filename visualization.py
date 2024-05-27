import matplotlib.pyplot as plt

def plot_rewards(rewards):
    
    """
    Plot the average reward over episodes.
    
    Parameters:
        rewards (np.array): Rewards array obtained over all episodes for an agent.
    """
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(rewards) / np.arange(1, len(rewards)+1), color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs. Episodes for ε-Greedy Agent')
    plt.grid(True)
    plt.show()

def plot_comparison(rewards, epsilon_values):
    
    """
    Plot the average rewards for agents with different ε-values to compare their performance.
    
    Parameters:
        rewards (np.array): Matrix of rewards, where each row corresponds to a different ε-agent.
        epsilon_values (list): List of ε-values used in the simulation.
    """
    
    plt.figure(figsize=(10, 5))
    for i in range(rewards.shape[0]):
        plt.plot(np.cumsum(rewards[i]) / np.arange(1, rewards.shape[1] + 1), label=f'ε={epsilon_values[i]}')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Comparison of ε-Greedy Agents')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_selections(selections):
    
    """
    Plot the number of times each arm was selected by an agent.
    
    Parameters:
        selections (np.array): Array containing the count of selections for each arm.
    """
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(selections)), selections, color='green')
    plt.xlabel('Arms')
    plt.ylabel('Number of times selected')
    plt.title('Selections of Each Arm')
    plt.show()