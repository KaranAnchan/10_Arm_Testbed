import numpy as np

class Agent:
    
    """
    Represents an agent using the ε-greedy strategy on a 10-armed bandit problem.
    
    Attributes:
        n_arms (int): Number of arms in the bandit.
        epsilon (float): Probability of selecting a random arm (explore).
        alpha (float): Step size for updating estimates.
        true_means (ndarray): True mean rewards of the arms.
        estimates (ndarray): Estimated values of the arms.
        counts (ndarray): Number of times each arm has been chosen.
    """
    
    def __init__(self, 
                 n_arms, 
                 epsilon, 
                 alpha, 
                 true_means):
        
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.true_means = true_means
        self.estimates = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        
    def choose_arm(self):
        
        """
        Choose an arm using ε-greedy strategy.
        
        Returns:
            int: The selected arm index.
        """
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)  # Explore
        else:
            return np.argmax(self.estimates)  # Exploit
        
    def get_reward(self, 
                   arm):
        
        
        
        reward = np.random.normal(self.true_means[arm], 1)
        return reward
    
    def update_estimates(self, 
                         arm, 
                         reward):
        
        
        
        self.counts[arm] += 1
        self.estimates[arm] += self.alpha * (reward - self.estimates[arm])  # Incremental update