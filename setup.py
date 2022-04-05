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

from setuptools import setup
from Cython.Build import cythonize
from numpy import get_include


setup(
    name='truncated2Dgauss',
    version='1.0',
    description='A truncated 2D Gaussian distribution.',
    url='https://github.com/Richard-Sti/truncated2Dgauss',
    author='Richard Stiskalek',
    author_email='richard.stiskalek@protonmail.com',
    license='GPL-3.0',
    packages=['truncated2Dgauss'],
    install_requires=['scipy>=0.16.0',
                      'numpy>=1.17.0',
                      'Cython>=0.29'],
    python_requires=">=3.6",
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'],
    ext_modules = cythonize("truncated2Dgauss/prob_mass.pyx",
                            annotate=True),
    include_dirs=[get_include(),],
    zip_safe=False,
)