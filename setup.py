from matplotlib.pyplot import annotate
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("truncated2Dgauss/norm.pyx", annotate=True)
)