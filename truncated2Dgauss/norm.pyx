from scipy.special import erfinv as erfinv_scipy
from scipy.integrate import quad

import numpy
cimport numpy
from libc.math cimport erf

numpy.import_array()
DTYPE = numpy.float64
ctypedef numpy.float64_t DTYPE_t


def _erfunc(double z):
    return erf(z)


def erfunc(double z):
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
    return 0.5 * (1 + sign * _erfunc(z / 2**0.5))


def erfinv(double x):
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
    return sign * 2**0.5 * erfinv_scipy(sign * (2 * x - 1))


def cholesky2x2(numpy.ndarray[DTYPE_t, ndim=2] S):
    r"""
    Calculate the Cholesky decomposition of a symmetric and positive
    semi-definite 2x2 covariance matrix `S`. The Cholesky decomposition is
    returned in the form:
    
    ..math::
        \begin{bmatrix}
            a & 0 \\
            c & b 
        \end{bmatrix}

    Parameters
    ---------
    S : array
        Covariance matrix
    
    Returns
    -------
    a, b, c : floats
        Cholesky decomposition as specified above.
    """
    # Cholesky decomposition solved analytically for a 2x2 matrix 
    cdef float a = S[0, 0]**0.5
    cdef float b = (S[1, 1] - S[0, 1]**2 / S[0, 0])**0.5
    cdef float c = S[0, 1] / a

    return a, b, c


def in_bounds(
    numpy.ndarray[DTYPE_t, ndim=1] x,
    numpy.ndarray[DTYPE_t, ndim=1] lower,
    numpy.ndarray[DTYPE_t, ndim=1] upper
):
    return ((x[0] >= lower[0]) & (x[1] >= lower[1])
            & (x[0] <= upper[0]) & (x[1] <= upper[1]))


cdef class BoxProbability:
    cdef numpy.ndarray _lower
    cdef numpy.ndarray _upper
    cdef double c11, c22, c12
    cdef double erf_xmin, erf_xmax
    cdef double ymin_norm, ymax_norm
    cdef dict _quad_kwargs

    def __init__(
        self,
        numpy.ndarray[DTYPE_t, ndim=1] lower,
        numpy.ndarray[DTYPE_t, ndim=1] upper,
        dict quad_kwargs={}
    ):
        self._lower = lower
        self._upper = upper
        self._quad_kwargs = quad_kwargs

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper
    
    def _integrand(self, double w):

        inv_norm = erfinv(self.erf_xmin
                          + w * (self.erf_xmax - self.erf_xmin))
        inv_norm *= self.c12 / self.c22
        return erfunc(self.ymax_norm - inv_norm) - erfunc(self.ymin_norm - inv_norm)

    def __call__(
        self,
        numpy.ndarray[DTYPE_t, ndim=1] mu,
        numpy.ndarray[DTYPE_t, ndim=2] cov
    ):

        cdef double intg, err

        self.c11, self.c22, self.c12 = cholesky2x2(cov)
        self.erf_xmin = erfunc((self.lower[0] - mu[0]) / self.c11)
        self.erf_xmax = erfunc((self.upper[0] - mu[0]) / self.c11)

        self.ymin_norm = (self.lower[1] - mu[1])/ self.c22
        self.ymax_norm = (self.upper[1] - mu[1])/ self.c22

        intg, err = quad(self._integrand, 0.0, 1.0, **self._quad_kwargs)

        if err > 1e-6:
            raise ValueError("Integration error of {} > 1e-6.".format(err))

        return (self.erf_xmax - self.erf_xmin) * intg