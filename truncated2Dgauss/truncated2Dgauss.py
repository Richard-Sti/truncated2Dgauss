from numpy import exp, log
from scipy.stats import multivariate_normal

from .norm import BoxProbability


class Truncated2DGaussian:
    lower = None
    upper = None

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.box_prob = BoxProbability(lower[0], upper[0], lower[1], upper[1])

    def _in_bounds(self, x):
        return ((x[0] >= self.lower[0]) & (x[1] >= self.lower[1])
                & (x[0] <= self.upper[0]) & (x[1] <= self.upper[1]))
    
    def logpdf(self, x, mean, cov, allow_singular=False):
        logprob = multivariate_normal.logpdf(
            x, mean=mean, cov=cov, allow_singular=allow_singular)
        logprob -= log(self.box_prob(cov[0,0], cov[1,1], cov[0,1]))
        return logprob

    def pdf(self, x, mean, cov, allow_singular=False):
        return exp(self.logpdf(x, mean, cov, allow_singular))

    def rvs(self, mean, cov):
        while True:
            x = multivariate_normal.rvs(mean, cov)
            if self._in_bounds(x):
                return x