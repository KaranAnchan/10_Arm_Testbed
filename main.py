import numpy as np
from agent import Agent

def main():
    # Parameters
    n_arms = 10
    episodes = 1000
    epsilon = 0.1
    alpha = 0.1
    
    # Initialization
    true_means = np.random.randn(n_arms)
    agent = Agent(n_arms, epsilon, alpha, true_means)