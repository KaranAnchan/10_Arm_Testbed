import numpy as np

class Agent:
    
    """
    Agent for the ε-greedy strategy in a multi-armed bandit problem.
    
    Attributes:
        n_arms (int): Number of arms in the bandit.
        epsilon (float): Exploration probability.
        alpha (float): Step size for incremental average.
        true_means (np.array): True mean rewards of the arms.
        estimates (np.array): Estimated values of the arms.
        counts (np.array): How many times each arm was chosen.
    """
    
    def __init__(self, n_arms, epsilon, alpha, true_means):
        
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.true_means = true_means
        self.estimates = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)

    def choose_arm(self):
        
        """
        Selects an arm based on the ε-greedy strategy.
        
        Returns:
            int: Index of the selected arm.
        """
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)  # Explore
        else:
            return np.argmax(self.estimates)  # Exploit

    def get_reward(self, arm):
        
        """
        Gets a reward for a chosen arm, simulated as a draw from a normal distribution.
        
        Parameters:
            arm (int): Index of the chosen arm.
        
        Returns:
            float: Simulated reward.
        """
        
        return np.random.normal(self.true_means[arm], 1)

    def update_estimates(self, arm, reward):
        
        """
        Updates the estimated value of the selected arm using incremental averaging.
        
        Parameters:
            arm (int): Index of the arm.
            reward (float): Reward received.
        """
        
        self.counts[arm] += 1
        self.estimates[arm] += self.alpha * (reward - self.estimates[arm])