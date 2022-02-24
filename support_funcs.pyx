from scipy.special import erf as erf_scipy
from scipy.special import erfinv as erfinv_scipy
from scipy.integrate import quad


def erf(float z):
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
    cdef float sign
    sign = 1.0 if z > 0 else -1.0
    if z < 0:
        z *= -1
    return 0.5 * (1 + sign * erf_scipy(z / 2**0.5))


def erfinv(float x):
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
    cdef float sign
    sign = 1.0 if (x - 0.5) > 0 else -1.0
    return sign * 2**0.5 * erfinv_scipy(sign * (2 * x - 1))


def cholesky2x2(float var1, float var2, float corr):
    r"""
    TODO: rewrite the documentation

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
    cdef float a, b, c
    # Cholesky decomposition solved analytically for a 2x2 matrix 
    a = var1**0.5
    b = (var2 - corr**2 / var1)**0.5
    c = corr / a

    return a, b, c


cdef class BoxProbability:
    cdef float a1, b1
    cdef float a2, b2
    cdef float c11, c22, c12
    cdef float erf_a1, erf_b1
    cdef float a2_norm, b2_norm
    cdef dict _quad_kwargs

    def __init__(self, float a1, float b1, float a2, float b2, dict quad_kwargs={}):
        self.a1, self.b1 = a1, b1
        self.a2, self.b2 = a2, b2
        self._quad_kwargs = quad_kwargs
    
    def _integrand(self, float w):
        cdef float inv_norm

        inv_norm = erfinv(self.erf_a1 + w * (self.erf_b1 - self.erf_a1))
        inv_norm *= self.c12 / self.c22
        return erf(self.b2_norm - inv_norm) - erf(self.a2_norm - inv_norm)

    def __call__(self, float var1, float var2, float corr):
        cdef float intg, err

        self.c11, self.c22, self.c12 = cholesky2x2(var1, var2, corr)
        self.erf_a1 = erf(self.a1 / self.c11)
        self.erf_b1 = erf(self.b1 / self.c11)

        self.a2_norm = self.a2 / self.c22
        self.b2_norm = self.b2 / self.c22

        intg, err = quad(self._integrand, 0.0, 1.0, **self._quad_kwargs)

        return (self.erf_b1 - self.erf_a1) * intg, err