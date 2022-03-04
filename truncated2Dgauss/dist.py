# Copyright (C) 2022 Richard Stiskalek
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


import numpy
from scipy.stats import multivariate_normal

from .prob_mass import CDFIntegral, is_higher_equal, is_in_bounds


class Truncated2DGaussian:
    _lower = None
    _upper = None
    _cdf = None
    _cov_hash = numpy.nan
    _dist = None
    _allow_singular = False

    def __init__(self, lower, upper, allow_singular=False):
        if not is_higher_equal(upper, lower):
            raise ValueError("`upper={}` must be larger than `lower={}`."
                             .format(lower, upper))
        self._lower = lower
        self._upper = upper
        self._allw_singular = allow_singular
        self._cdf = CDFIntegral(lower, upper)

    @property
    def lower(self):
        """
        Lower box limits.

        Returns
        -------
        lower : array
        """
        return self._lower

    @property
    def upper(self):
        """
        Upper box limits.

        Returns
        -------
        upper : array
        """
        return self._upper

    def _check_new_cov(self, cov):
        """
        Check whether the covariance matrix has changed. If yes recalculate
        the internal multivariate normal distribution.

        Arguments
        ---------
        cov : 2-dimensional array
            The new covariance matrix.
        """
        new_hash = hash(cov.data.tobytes())
        if new_hash != self._cov_hash:
            self._cov_hash = new_hash
            self._dist = multivariate_normal(
                cov=cov, allow_singular=self._allow_singular)

    def logpdf(self, x, mean, cov, allow_singular=False):
        """
        Log probability density at `x`.
        
        TODO: finish up the docs
        """
        # Check the mean is in bounds
        if not is_in_bounds(x, self.lower, self.upper):
            return ValueError("`mean={}` is out of bounds.".format(mean))
        # Check current position is in bounds
        if not is_in_bounds(x, self.lower, self.upper):
            return -numpy.infty
        # Optionally update the distribution
        self._check_new_cov(cov)
        return self._dist.logpdf(x - mean) - numpy.log(self._cdf(mean, cov))

    def pdf(self, x, mean, cov, allow_singular=False):
        return numpy.exp(self.logpdf(x, mean, cov, allow_singular))

    def rvs(self, mean, cov):
        """
        Rejection sampling of a single observation from the trucanted 2D
        distribution.
        
        Arguments
        ---------
        mean : 1-dimensional array
            Mean of the non-truncated Gaussian.
        cov : 2-dimensional array
            Covariance of the non-truncated Gaussian.

        Returns
        -------
        sample : 1-dimensional array
            The new observation.
        """
        # Check the mean is in bounds
        if not is_in_bounds(x, self.lower, self.upper):
            return ValueError("`mean={}` is out of bounds.".format(mean))
        self._check_new_cov(cov)
        while True:
            x = mean + self._dist.rvs()
            if is_in_bounds(x, self.lower, self.upper):
                return x