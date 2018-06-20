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

"""Unit tests for gwin.results
"""

import itertools

import pytest

import numpy
from numpy.testing import assert_array_equal

from scipy.stats import gaussian_kde

from matplotlib import use
use('agg')  # nopep8
from matplotlib import pyplot
from matplotlib.figure import Figure
from matplotlib.axes import SubplotBase

from gwin.results import scatter_histograms

try:
    from kombine.clustered_kde import KDE
except ImportError:
    kde_types = (gaussian_kde, )
    HAS_KOMBINE = False
else:
    kde_types = (gaussian_kde, KDE)
    HAS_KOMBINE = True

from utils import mock

skip_no_kombine = pytest.mark.skipif(not HAS_KOMBINE,
                                     reason='Kombine not installed')


# -- scatter_histograms.py ----------------------------------------------------

@pytest.mark.parametrize('nodiag', [False, True])
def test_create_axes_grid(nodiag):
    parameters = ['mass1', 'mass2', 'tc']
    if nodiag:
        combinations = itertools.combinations
        ndim = len(parameters) - 1
    else:
        combinations = itertools.combinations_with_replacement
        ndim = len(parameters)

    # create figure
    fig, axes = scatter_histograms.create_axes_grid(
        parameters, no_diagonals=nodiag)

    # test
    assert isinstance(fig, Figure)
    for p1, p2 in combinations(parameters, 2):
        ax, row, col = axes[p1, p2]
        print(row, col, p1, p2)
        print(ax.get_xlabel(), ax.get_ylabel())
        assert isinstance(ax, SubplotBase)
        if row + 1 == ndim:
            assert ax.get_xlabel() == p1
        else:
            assert ax.get_xlabel() == ''
        if col == 0:
            assert ax.get_ylabel() == p2
        else:
            assert ax.get_ylabel() == ''


@pytest.mark.parametrize('width, height, scale', [
    (8, 7, 1),
    (16, 14, 2),
    (6.4, 4.8, .7406560798180412),
])
def test_get_scale_factor(width, height, scale):
    fig = pyplot.figure(figsize=(width, height))
    assert scatter_histograms.get_scale_fac(fig) == scale


@pytest.mark.parametrize('kombine', [
    None,  # test ImportError
    False,
    pytest.param(True, marks=skip_no_kombine),
])
def test_construct_kde(kombine):
    numpy.random.seed(0)
    samples = numpy.random.rand(5, 5)

    if kombine is None:
        with mock.patch.dict('sys.modules', {'kombine': None}):
            with pytest.raises(ImportError):
                scatter_histograms.construct_kde(samples, use_kombine=True)
        return

    kde = scatter_histograms.construct_kde(samples, use_kombine=kombine)
    assert isinstance(kde, kde_types)


@pytest.mark.parametrize('data, offset', [
    (numpy.ones(4), 0),  # simple array
    (1e4 * (numpy.arange(10) - 5), 0),  # large array with mixed sign
    (numpy.arange(10) + 1001, 1001),  # large positive-only array
    (-(numpy.arange(10) + 1001), -1001),  # large negative-only array
])
def test_remove_common_offset(data, offset):
    new, off = scatter_histograms.remove_common_offset(data)
    assert_array_equal(new + off, data)
    assert off == offset


def test_reduce_ticks():
    fig = pyplot.figure()
    ax = fig.gca()
    ax.set_xticks(numpy.linspace(0, 1, num=6))

    ticks = scatter_histograms.reduce_ticks(ax, 'x', maxticks=3)
    assert_array_equal(ticks, ax.get_xticks()[1:5:2])

    ticks = scatter_histograms.reduce_ticks(ax, 'x', maxticks=7)
    assert_array_equal(ticks, ax.get_xticks())
