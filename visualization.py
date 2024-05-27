import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(rewards) / np.arange(1, len(rewards)+1), color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs. Episodes for 10-Arm Testbed')
    plt.grid(True)
    plt.savefig('10_arm_testbed_performance.png')
    plt.show()