#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018  Collin Capano
#
# This file is part of GWIn
#
# GWIn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWIn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWIn.  If not, see <http://www.gnu.org/licenses/>.

"""Setup the GWIn package
"""

import os.path
import re
import sys

from setuptools import (setup, find_packages)

import versioneer

# -- versioning ---------------------------------------------------------------

cmdclass = versioneer.get_cmdclass()
__version__ = versioneer.get_version()

# -- dependencies -------------------------------------------------------------

setup_requires = []
if set(('test',)).intersection(sys.argv):
    setup_requires.extend(['pytest_runner'])

install_requires = [
    'numpy',
    'pycbc>=1.11.1',
    'matplotlib',
    'scipy',
    'h5py',
]

try:  # try to detect system install of lalinference
    import lalinference
except ImportError:  # fall-back to installing lalsuite wheels
    install_requires.append('lalsuite')

extras_require = {
    'kombine': ['kombine'],
    'emcee': ['emcee'],
    'docs': ['sphinx >= 1.7.1', 'sphinx_rtd_theme', 'numpydoc',
             'sphinxcontrib_programoutput'],
}

tests_require = [
    'pytest>=2.8',
]

if sys.version < '3':
    tests_require.append('mock')


# -- files --------------------------------------------------------------------

def find_scripts(scripts_dir='bin'):
    """Get relative file paths for all files under the ``scripts_dir``
    """
    scripts = []
    for (dirname, _, filenames) in os.walk(scripts_dir):
        scripts.extend([os.path.join(dirname, fn) for fn in filenames])
    return scripts


# -- setup --------------------------------------------------------------------

# get long description from README
with open('README.rst', 'rb') as f:
    longdesc = f.read().decode().strip()

# run setup
setup(
    name='gwin',
    version=__version__,
    description="A python package for Bayesian inference of "
                "gravitational-wave data",
    long_description=longdesc,
    author='Collin Capano',
    author_email='collin.capano@ligo.org',
    url='https://github.com/gwastro/gwin',
    license='GPLv3',
    cmdclass=cmdclass,
    packages=find_packages(),
    scripts=find_scripts(),
    setup_requires=setup_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    tests_require=tests_require,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
)
