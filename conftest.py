# Copyright (C) 2018 Duncan Macleod
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
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

"""Pytest configuration module

This just adds an extra command line option to specify the waveform
approximants to use in the tests
"""

import pytest

from pycbc.waveform import fd_approximants

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

DEFAULT_APPROXIMANTS = [
    'TaylorF2',
    'IMRPhenomPv2',
]


def pytest_addoption(parser):
    """Add --approximants command-line option
    """
    parser.addoption('--approximants', type=str, nargs='*', metavar='APPROX',
                     choices=sorted(fd_approximants()),
                     default=DEFAULT_APPROXIMANTS,
                     help='one or more approximant names to use in tests, '
                          'default: %(default)s, choices: %(choices)s')


def pytest_generate_tests(metafunc):
    """Enable approximant parametrisation if requested
    """
    if 'approximant' in metafunc.fixturenames:
        metafunc.parametrize("approximant",
                             metafunc.config.getoption('approximants'))
