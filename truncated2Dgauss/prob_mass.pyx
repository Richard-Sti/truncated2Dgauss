import sys
import numpy
cimport numpy
from cython import cdivision, boundscheck, wraparound
from libc.math cimport erf as _erf
from scipy.special.cython_special import erfinv as _erfinv

from scipy.integrate import quad
from scipy import LowLevelCallable


# Setup the numpy arrays behaviour
numpy.import_array()
DTYPE = numpy.float64
ctypedef numpy.float64_t DTYPE_t


MODULE = sys.modules[__name__]


################################################################################
#####                        Error function                                #####
################################################################################
cpdef double erf(double z):
    r"""
    Calculate the error function specified as
    
    ..math::
        \Phi(z) = 1 / \sqrt(2\pi) \int_{-\infty}^z e^{-t^2 / 2} \mathrm{d}t

    Parameters
    ----------
    z : float
        Point to evaluate the error function such
        that :math:`z \in (-\infty, + \infty)`.

    Returns
    -------
    result : float
        Evaluated error function.
    """
    cdef double sign = 1.0 if z > 0 else -1.0
    if z < 0:
        z *= -1
    return 0.5 * (1 + sign * _erf(z / 2**0.5))


################################################################################
#####                Inverse error function                                #####
################################################################################
cpdef double erfinv(double x):
    r"""
    Inverse error function as specified in ``erf``.

    Parameters
    ----------
    x : float
        Error function value to be inverted. Must be :math:`x \in [0, 1]`.
    
    Returns
    -------
    result : float
        Error function inverse.
    """
    cdef double sign = 1.0 if (x - 0.5) > 0 else -1.0
    return sign * 2**0.5 * _erfinv(sign * (2 * x - 1))


################################################################################
#####                       CDF calculation                                #####
################################################################################
cdef double integrand(int n, double[6] args):
    """
    The integral to be calculated within `CFDIntegral`.
    """
    cdef double inv_norm = args[3] * erfinv(args[1]+ args[0] * args[2])
    return erf(args[4]- inv_norm) - erf(args[5]- inv_norm)


cdef class CDFIntegral:
    cdef numpy.ndarray lower
    cdef numpy.ndarray upper
    cdef object integrand
  
    def __init__(
        self,
        numpy.ndarray[DTYPE_t, ndim=1] lower,
        numpy.ndarray[DTYPE_t, ndim=1] upper
        ):
        self.lower = lower
        self.upper = upper
        self.integrand = LowLevelCallable.from_cython(MODULE, 'integrand')

    @cdivision(True)
    @boundscheck(False)
    @wraparound(False)
    def precalculate(
        self,
        numpy.ndarray[DTYPE_t, ndim=1] mean,
        numpy.ndarray[DTYPE_t, ndim=2] cov
        ):
        # Cholesky decomposition of the covariance matrix
        cdef double c11 = cov[0, 0]**0.5
        cdef double c22 = (cov[1, 1] - cov[0, 1]**2 / cov[0, 0])**0.5
        cdef double c12 = cov[0, 1] / c11

        cdef double erf_xmin = erf((self.lower[0] - mean[0]) / c11)
        cdef double erf_xmax = erf((self.upper[0] - mean[0]) / c11)
        cdef double d_erf = erf_xmax - erf_xmin
        cdef double ymin_norm = (self.lower[1] - mean[1])/ c22
        cdef double ymax_norm = (self.upper[1] - mean[1])/ c22
        cdef double cratio = c12 / c22

        return (erf_xmin, d_erf, cratio, ymax_norm, ymin_norm) 

    def __call__(self, mean, cov):
        args = self.precalculate(mean, cov)

        intg, err = quad(self.integrand, 0.0, 1.0, args=args)
        return args[1] * intg 
      

################################################################################
#####                       Check boundaries                               #####
################################################################################
@boundscheck(False)
@wraparound(False)
def is_higher_equal(
    numpy.ndarray[DTYPE_t, ndim=1] x,
    numpy.ndarray[DTYPE_t, ndim=1] y
    ):
    """
    Checks that `x` is higher or equal to `y`, where both are 1-dimensional
    arrays of length 2.
    
    Parameters
    ----------
    x : array
        The first array checked to be larger.
    y : array
        The second array.

    Returns
    -------
        : bool
    """
    return (x[0] >= y[0]) & (x[1] >= y[1])


@boundscheck(False)
@wraparound(False)
def is_in_bounds(
    numpy.ndarray[DTYPE_t, ndim=1] x,
    numpy.ndarray[DTYPE_t, ndim=1] lower,
    numpy.ndarray[DTYPE_t, ndim=1] upper
    ):
    """
    Checks that `x` is within `lower` and `upper`. All must be 1-dimensional
    arrays of length 2.

    Parameters
    ----------
    x : array
        Point checked to be within the boundaries.
    lower : array
        Lower bound.
    upper : array
        Upper bound.
    
    Returns
    -------
        : bool
    """
    return ((x[0] >= lower[0]) & (x[1] >= lower[1])
            & (x[0] <= upper[0]) & (x[1] <= upper[1]))