#!/usr/bin/env python

from setuptools import setup, find_packages, Extension
import numpy as np


setup(name='alienworlds',
      version='0.1',
      description='Searching technosignatures in Kepler and TESS data',
      author='Ushasi Bhowmick, Vikram Khaire',
      author_email='ushasibhowmick@gmail.com, vickykhaire@gmail.com',
      url='https://github.com/ushasi-bhowmick/alienworlds',
      include_dirs=np.get_include(),
      install_requires=['astropy',
                        'numpy',
                        'matplotlib'],
     )
