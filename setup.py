from matplotlib.pyplot import annotate
from setuptools import setup
from Cython.Build import cythonize
import numpy


setup(
    ext_modules = cythonize(["truncated2Dgauss/erf.pyx",
                             "truncated2Dgauss/prob_mass.pyx"],
                            annotate=True),
    include_dirs=[numpy.get_include()]
)