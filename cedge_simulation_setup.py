# python cedge_simulation_setup.py build_ext --inplace
__author__ = 'tillhoffmann'

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("cedge_simulation", ["cedge_simulation.pyx"],
              include_dirs=[np.get_include()],
              libraries=['gsl', 'gslcblas'])
]

setup(ext_modules=cythonize(extensions, annotate=True))