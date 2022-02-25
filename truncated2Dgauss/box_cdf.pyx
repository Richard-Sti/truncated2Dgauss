import numpy
cimport numpy
from libc.math cimport erf

# Can these Python calls be avoided?
from scipy.special import erfinv as erfinv_scipy
from scipy.integrate import quad


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


def is_higher_equal(
    numpy.ndarray[DTYPE_t, ndim=1] x,
    numpy.ndarray[DTYPE_t, ndim=1] y
    ):
    return (x[0] >= y[0]) & (x[1] >= y[1])


def in_bounds(
    numpy.ndarray[DTYPE_t, ndim=1] x,
    numpy.ndarray[DTYPE_t, ndim=1] lower,
    numpy.ndarray[DTYPE_t, ndim=1] upper
    ):
    return ((x[0] >= lower[0]) & (x[1] >= lower[1])
            & (x[0] <= upper[0]) & (x[1] <= upper[1]))


cdef class BoxCDF:
    """
    Cumulative density function of a 2D Gaussian distribution integrated over
    a box. 

    Parameters
    ----------
    lower : numpy.ndarray
        1D array of box lower bounds.
    upper : numpy.ndarray
        1D array of box upper bounds.
    quad_kwargs : dict, optional
        Optional keyword arguments passed into `scipy.integrate.quad`.
    """
    cdef numpy.ndarray _lower
    cdef numpy.ndarray _upper
    cdef double _c11, _c22, _c12
    cdef double _erf_xmin, _erf_xmax
    cdef double _ymin_norm, _ymax_norm
    cdef dict _quad_kwargs

    def __init__(
        self,
        numpy.ndarray[DTYPE_t, ndim=1] lower,
        numpy.ndarray[DTYPE_t, ndim=1] upper,
        dict quad_kwargs={}
        ):
        # Check boundaries
        if not is_higher_equal(upper, lower):
            raise ValueError(
                "``upper: {}`` must be higher or equal to ``lower: {}``."
                .format(upper, lower))

        self._lower = lower
        self._upper = upper
        self._quad_kwargs = quad_kwargs

    @property
    def lower(self):
        """The box lower bounds."""
        return self._lower

    @property
    def upper(self):
        """The box upper bounds."""
        return self._upper
    
    def _integrand(self, double w):
        """Expression to be integrated. For more information see [1]."""
        cdef double inv_norm = erfinv(
            self._erf_xmin + w * (self._erf_xmax - self._erf_xmin)
            )
        inv_norm *= self._c12 / self._c22
        return (erfunc(self._ymax_norm - inv_norm)
                - erfunc(self._ymin_norm - inv_norm)
                )

    def __call__(
        self,
        numpy.ndarray[DTYPE_t, ndim=1] mu,
        numpy.ndarray[DTYPE_t, ndim=2] cov,
    ):
        """
        Cumulative density function of a 2D Gaussian integrated over a box
        specified by `self.lower` and `self.upper`.

        Parameters
        ----------
        mu : numpy.ndarray
            Mean of the unbounded Gaussian.
        cov : numpy.ndarray
            Covariance of the unbounded Gaussian.

        Returns
        CDF : float
            CDF of an unconstrained Gaussian evaluated over the box.
        """
        cdef double intg, err

        self._c11, self._c22, self._c12 = cholesky2x2(cov)
        self._erf_xmin = erfunc((self.lower[0] - mu[0]) / self._c11)
        self._erf_xmax = erfunc((self.upper[0] - mu[0]) / self._c11)

        self._ymin_norm = (self.lower[1] - mu[1])/ self._c22
        self._ymax_norm = (self.upper[1] - mu[1])/ self._c22

        intg, err = quad(self._integrand, 0.0, 1.0, **self._quad_kwargs)

        if err > 1e-6:
            raise ValueError("Integration error of {} > 1e-6.".format(err))

        return (self._erf_xmax - self._erf_xmin) * intg