import numpy as np
import numpy.matlib as nm

class Gaussian_dist_1D:
    def __init__(self, mu, var):
        self.mu = mu
        self.var = var
    
    def logprob(self, x):
        return -0.5 * (np.log(2 * np.pi * self.var) + (x - self.mu)**2 / self.var)
    
    def dlogprob(self, x):
        # derivative of log probability with respect to x
        return - (x - self.mu) / self.var

class Mix_Gaussian_dist_1D:
    def __init__(self, gau1, gau2, prob1, prob2):
        self.gau1 = gau1
        self.gau2 = gau2
        self.prob1 = prob1
        self.prob2 = prob2
    
    def logprob(self, x):
        return np.log(self.prob1 * np.exp(self.gau1.logprob(x)) + self.prob2 * np.exp(self.gau2.logprob(x)))
    
    def dlogprob(self, x):
        # derivative of log probability with respect to x
        # log p(x) = log (prob1*p1 + prob2*p2) = log (explog prob1*p1 + explog prob2*p2)
        p = np.exp(self.logprob(x))
        p1, p2 = np.exp(self.gau1.logprob(x)), np.exp(self.gau2.logprob(x))
        dlogp1, dlogp2 = self.gau1.dlogprob(x), self.gau2.dlogprob(x)
        return 1 / p * (self.prob1 * p1 * dlogp1 + self.prob2 * p2 * dlogp2)

class Gaussian_dist_2D:
    def __init__(self, mu, pre):
        assert len(mu) == 2
        assert len(pre) == 2
        self.mu = np.array(mu)
        self.pre = np.array([[pre[0],0],[0,pre[1]]])  # precision matrix (diagonal)
    
    def logprob(self, x):
        temp = np.matmul(x - nm.repmat(self.mu, x.shape[0], 1), self.pre)
        temp = np.diagonal(np.matmul(temp, (x - nm.repmat(self.mu, x.shape[0], 1)).T))
        return 0.5 * (np.sum(np.log(np.diagonal(self.pre))) - 2 * np.log(2*np.pi) - temp)
    
    def dlogprob(self, x):
        # derivative of log probability with respect to x
        return -1 * np.matmul(x - nm.repmat(self.mu, x.shape[0], 1), self.pre)

class Mix_Gaussian_dist_2D:
    def __init__(self, gau1, gau2, prob1, prob2):
        self.gau1 = gau1
        self.gau2 = gau2
        self.prob1 = prob1
        self.prob2 = prob2
    
    def logprob(self, x):
        return np.log(self.prob1 * np.exp(self.gau1.logprob(x)) + self.prob2 * np.exp(self.gau2.logprob(x)))
    
    def dlogprob(self, x):
        # derivative of log probability with respect to x
        # log p(x) = log (prob1*p1 + prob2*p2) = log (explog prob1*p1 + explog prob2*p2)
        p = np.exp(self.logprob(x)).reshape(-1, 1)
        p1, p2 = np.exp(self.gau1.logprob(x)).reshape(-1, 1), np.exp(self.gau2.logprob(x)).reshape(-1, 1)
        dlogp1, dlogp2 = self.gau1.dlogprob(x), self.gau2.dlogprob(x)
        return 1 / p * (self.prob1 * p1 * dlogp1 + self.prob2 * p2 * dlogp2)