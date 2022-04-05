#!/usr/bin/env python

# Copyright (C) 2020  Collin Capano
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
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
"""Performs unit tests on the truncated 2D Gaussian."""

import pytest
import numpy
from scipy import stats
from scipy.integrate import dblquad

from truncated2Dgauss import Truncated2DGauss


SEED = 42
MU = [0.0, 0.0]
COV = [[1, 0], [0, 1]]
NSAMPLES = 64
# set the scipy seed for comparison
numpy.random.seed(SEED)


@pytest.mark.parametrize("lower", [[0, 0],
                                   [-1, 0],
                                   [-4, -3]])
@pytest.mark.parametrize("upper", [[1, 0.1],
                                   [2, 1],
                                   [3, 15]])
@pytest.mark.parametrize("cov", [[[1, 0.5], [0.5, 1]],
                                 [[1, 0.9], [0.9, 1]]])
def test_normalisation(lower, upper, cov):
    """
    Test that the PDF is correctly normalised and 0 outside boundaries.
    """
    dist = Truncated2DGauss(lower, upper, SEED)

    f = lambda y, x: dist.pdf([x, y], MU, cov)
    intg, err = dblquad(f, lower[0], upper[0], lower[1], upper[1])
    assert numpy.isclose(intg, 1.0)


@pytest.mark.parametrize("lower", [[0, 0],
                                   [-1, 0],
                                   [-4, -3]])
@pytest.mark.parametrize("upper", [[1, 0.1],
                                   [2, 1],
                                   [3, 15]])
@pytest.mark.parametrize("cov", [COV,
                                 [[1, 0.5], [0.5, 1]],
                                 [[1, 0.9], [0.9, 1]]])
def test_boundaries(lower, upper, cov):
    """
    Test that the PDF is 0 outside boundaries.
    """
    dist = Truncated2DGauss(lower, upper, SEED)
    lower = numpy.array(lower)
    upper = numpy.array(upper)

    # Create a wide uniform distribution and sample it
    uniform_dist = stats.uniform(lower - 2, upper + 2)
    for __ in range(NSAMPLES):
        x = uniform_dist.rvs()

        isinbounds = ((lower[0] < x[0] < upper[0])
                      & (lower[1] < x[1] < upper[1]))

        assert numpy.isfinite(dist.logpdf(x, MU, cov)) == isinbounds
        

@pytest.mark.parametrize("lower", [[0, 0],
                                   [-1, 0],
                                   [-4, -3]])
@pytest.mark.parametrize("upper", [[1, 0.1],
                                   [2, 1],
                                   [3, 15]])
@pytest.mark.parametrize("cov", [COV,
                                 [[1, 0.5], [0.5, 1]],
                                 [[1, 0.9], [0.9, 1]]])
@pytest.mark.parametrize("seed", [SEED, 2022])
def test_rvs(lower, upper, cov, seed):
    """
    Test drawing samples.
    """
    dist = Truncated2DGauss(lower, upper, seed)

    for i in range(NSAMPLES):
        x = dist.rvs(MU, cov)
        isinbounds = ((lower[0] < x[0] < upper[0])
                      & (lower[1] < x[1] < upper[1]))

        assert isinbounds
        assert dist.pdf(x, MU, cov) > 0
