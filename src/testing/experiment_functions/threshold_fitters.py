from typing import List, Union

import numpy as np
from scipy import stats


class GammaFitter(object):

    def __init__(self, obs: Union[List, np.ndarray]):
        """
        Initialises object

        :param obs:      Observations
        """

        self.n_rvs            = len(obs)
        self.mu1, self.mu2    = np.mean(obs), np.var(obs)
        self.alpha, self.beta = None, None

        self.calculate_gamma_approximation()

    def calculate_gamma_approximation(self) -> None:
        """
        Calculates alpha, beta estimate values

        :return: None
        """
        self.alpha = (np.power(self.mu1, 2)) / self.mu2
        self.beta = (self.n_rvs * self.mu2) / self.mu1

    def cdf(self, x: float):
        """
        Calculates cdf of corresponding gamma distribution at a given point x

        :param x:       Point of evaluation
        :return:        Value of cdf at x
        """
        return stats.gamma.cdf(x*self.n_rvs, a=self.alpha, scale=self.beta)

    def ppf(self, q: float):
        """
        Calculates quantile function of corresponding distribution for a given level q

        :param q:   Quantile to calculate inverse cdf of
        :return:    Value of quantile function at q
        """
        return stats.gamma.ppf(q, a=self.alpha, scale=self.beta)/self.n_rvs
