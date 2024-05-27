import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards):
    
    """
    Plot the average reward over episodes.
    
    Parameters:
        rewards (list): List of rewards obtained per episode.
    """
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(rewards) / np.arange(1, len(rewards)+1), color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs. Episodes for 10-Arm Testbed')
    plt.grid(True)
    plt.savefig('10_arm_testbed_performance.png')
    plt.show()
    
def plot_selections(selections):
    
    """
    Plot the number of times each arm was selected.
    
    Parameters:
        selections (ndarray): Array containing the count of selections for each arm.
    """
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(selections)), selections, color='green')
    plt.xlabel('Arms')
    plt.ylabel('Number of times selected')
    plt.title('Selections of Each Arm')
    plt.savefig('10_arm_testbed_selections.png')
    plt.show()