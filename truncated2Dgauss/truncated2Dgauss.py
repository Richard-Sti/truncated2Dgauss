from numpy import exp, log
from scipy.stats import multivariate_normal

from .norm import BoxProbability, in_bounds


class Truncated2DGaussian:
    lower = None
    upper = None

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.box_prob = BoxProbability(lower, upper)

    def logpdf(self, x, mean, cov, allow_singular=False):
        logprob = multivariate_normal.logpdf(
            x, mean=mean, cov=cov, allow_singular=allow_singular)
        logprob -= log(self.box_prob(mean, cov))
        return logprob

    def pdf(self, x, mean, cov, allow_singular=False):
        return exp(self.logpdf(x, mean, cov, allow_singular))

    def rvs(self, mean, cov):
        while True:
            x = multivariate_normal.rvs(mean, cov)
            if in_bounds(x, self.lower, self.upper):
                return x