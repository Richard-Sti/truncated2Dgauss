from matplotlib.pyplot import annotate
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("truncated2Dgauss/norm.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)