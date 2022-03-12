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


def list_to_array(x):
    if isinstance(x, numpy.ndarray):
        return x
    else:
        return numpy.asarray(x)

class Truncated2DGauss:
    _lower = None
    _upper = None
    _cdf = None
    _cov_hash = numpy.nan
    _dist = None
    _random_generator = None

    def __init__(self, lower, upper, random_generator=None):
        # Ensure we have numpy arrays
        self._lower = list_to_array(lower)
        self._upper = list_to_array(upper)
        # Which have correct ordering 
        if not is_higher_equal(self._upper, self._lower):
            raise ValueError("`upper={}` must be larger than `lower={}`."
                             .format(self._lower, self._upper))
        # Create the random generator
        if random_generator is None:
            self._random_generator = numpy.random.default_rng()
        else:
            self._random_generator = random_generator

        self._cdf = CDFIntegral(self._lower, self._upper)

    @property
    def lower(self):
        """
        Lower box limits.

        Returns
        -------
        lower : 1-dimensional array
        """
        return self._lower

    @property
    def upper(self):
        """
        Upper box limits.

        Returns
        -------
        upper : 1-dimensional array
        """
        return self._upper

    def _check_new_cov(self, cov, allow_singular):
        """
        Check whether the covariance matrix has changed. If yes recalculate
        the internal multivariate normal distribution.

        Arguments
        ---------
        cov : 2-dimensional array
            The new covariance matrix.
        allow_singular : bool, optional
            Whether to allow a singular covariance matrix.
        """
        new_hash = hash(cov.data.tobytes())
        if new_hash != self._cov_hash:
            self._cov_hash = new_hash
            self._dist = multivariate_normal(
                cov=cov, allow_singular=allow_singular)

    def _is_mean_in_bounds(self, mean):
        if not is_in_bounds(mean, self.lower, self.upper):
            return ValueError("`mean={}` is out of bounds.".format(mean))

    def logpdf(self, x, mean, cov, allow_singular=False):
        """
        Log probability density at `x`.
        
        TODO: finish up the docs
        """
        # Optionally convert to arrays
        x = list_to_array(x)
        mean = list_to_array(x)
        cov = list_to_array(cov)
        
        self._is_mean_in_bounds(mean)
        # Check current position is in bounds
        if not is_in_bounds(x, self.lower, self.upper):
            return -numpy.infty

        # Optionally update the distribution
        self._check_new_cov(cov, allow_singular)
        return self._dist.logpdf(x - mean) - numpy.log(self._cdf(mean, cov))

    def pdf(self, x, mean, cov, allow_singular=False):
        return numpy.exp(self.logpdf(x, mean, cov, allow_singular))

    def rvs(self, mean, cov):
        """
        Rejection sampling of a single observation from the truncated 2D
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
        mean = list_to_array(mean)
        self._is_mean_in_bounds(mean)
        while True:
            x = self._random_generator.multivariate_normal(mean, cov)
            if is_in_bounds(x, self.lower, self.upper):
                return x