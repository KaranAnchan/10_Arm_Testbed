import numpy as np

class Agent:
    def __init__(self, n_arms, epsilon, alpha, true_means):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.true_means = true_means
        self.estimates = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)