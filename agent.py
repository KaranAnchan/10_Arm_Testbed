import numpy as np

class Agent:
    
    """
    Represents an ε-greedy agent for the multi-armed bandit problem.

    Attributes:
        n_arms (int): Number of arms in the bandit.
        epsilon (float): Probability of exploring.
        alpha (float): Step size for updating estimates.
        true_means (np.array): True reward means for each arm.
        estimates (np.array): Estimated values for each arm.
        counts (np.array): Count of selections for each arm.
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
        Select which arm to pull based on ε-greedy strategy.

        Returns:
            int: Index of the chosen arm.
        """
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)  # Explore
        else:
            return np.argmax(self.estimates)  # Exploit

    def get_reward(self, arm):
        
        """
        Simulate getting a reward for a chosen arm.

        Parameters:
            arm (int): The chosen arm.

        Returns:
            float: The reward received.
        """
        
        return np.random.normal(self.true_means[arm], 1)

    def update_estimates(self, arm, reward):
        
        """
        Update the estimated values for a chosen arm based on the received reward.

        Parameters:
            arm (int): The chosen arm.
            reward (float): The reward received.
        """
        
        self.counts[arm] += 1
        self.estimates[arm] += self.alpha * (reward - self.estimates[arm])  # Incremental update
