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

"""Tests for :mod:`gwin.option_utils`
"""

import argparse

import mock

import pytest

from gwin import option_utils
from gwin.io.hdf import InferenceFile
from gwin.io.txt import InferenceTXTFile
from gwin.sampler import MCMCSampler

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


@pytest.mark.parametrize('input_, output', [
    ('test', 'test'),
    ('a, b, c', 'a, b, c'),
    ('[foo, bar, cat]', ['foo', 'bar', 'cat']),
    ('[foo bar, cat]', ['foo bar', 'cat']),
])
def test_convert_liststring_to_list(input_, output):
    assert option_utils.convert_liststring_to_list(input_) == output


def test_add_config_opts_to_parser():
    parser = argparse.ArgumentParser()
    option_utils.add_config_opts_to_parser(parser)
    args = parser.parse_args(['--config-files', 'test', 'test2',
                              '--config-overrides', 'test3'])
    assert args.config_files == ['test', 'test2']
    assert args.config_overrides == ['test3']
    with pytest.raises(SystemExit):
        parser.parse_args(['--blah'])


def test_add_sampler_option_group(capsys):
    parser = argparse.ArgumentParser()
    option_utils.add_sampler_option_group(parser)

    # check no arguments raises error
    with pytest.raises(SystemExit):
        parser.parse_args()
    assert '--sampler is required' in capsys.readouterr()[1]

    # check type casting
    args = parser.parse_args([
        '--sampler', 'mcmc',
        '--niterations', '1',
        '--n-independent-samples', '2',
        '--nwalkers', '3',
        '--ntemps', '4',
        '--burn-in-function', 'half_chain',
        '--min-burn-in', '5',
        '--update-interval', '6',
        '--nprocesses', '7',
        '--use-mpi',
    ])
    for arg in ('niterations', 'n_independent_samples', 'nwalkers', 'ntemps',
                'min_burn_in', 'update_interval', 'nprocesses'):
        assert isinstance(getattr(args, arg), int)

    # check choices
    with pytest.raises(SystemExit):
        parser.parse_args(['--sampler', 'foo'])
    assert 'invalid choice: \'foo\'' in capsys.readouterr()[1]
    with pytest.raises(SystemExit):
        parser.parse_args(['--burn-in-function', 'bar'])
    assert 'invalid choice: \'bar\'' in capsys.readouterr()[1]


@mock.patch('gwin.likelihood.GaussianLikelihood')
def test_sampler_from_cli(Likelihood):
    parser = argparse.ArgumentParser()
    option_utils.add_sampler_option_group(parser)
    args = parser.parse_args([
        '--sampler', 'mcmc',
    ])
    sampler = option_utils.sampler_from_cli(args, Likelihood())
    assert isinstance(sampler, MCMCSampler)


def test_add_low_frequency_cutoff_opt():
    parser = argparse.ArgumentParser()
    option_utils.add_low_frequency_cutoff_opt(parser)
    args = parser.parse_args(['--low-frequency-cutoff', '123.456'])
    assert args.low_frequency_cutoff == 123.456


def test_low_frequency_cutoff_from_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', nargs='+')
    option_utils.add_low_frequency_cutoff_opt(parser)
    args = parser.parse_args([
        '--instruments', 'H1', 'L1',
        '--low-frequency-cutoff', '123.456',
    ])
    assert option_utils.low_frequency_cutoff_from_cli(args) == {
        'H1': 123.456, 'L1': 123.456}


def test_add_inference_results_option_group():
    parser = argparse.ArgumentParser()
    option_utils.add_inference_results_option_group(parser)
    args = parser.parse_args([
        '--input-file', 'test.h5', 'test2.h5',
        '--parameters', 'a:1', 'b:2',
        '--thin-start', '1',
        '--thin-interval', '2',
        '--thin-end', '3',
        '--iteration', '4',
        '--parameters-group', 'samples',
    ])
    assert args.input_file == ['test.h5', 'test2.h5']
    for arg in ('thin_start', 'thin_interval', 'thin_end'):
        assert isinstance(getattr(args, arg), int)

    # try again without --parameters-group
    parser = argparse.ArgumentParser()
    option_utils.add_inference_results_option_group(
        parser, include_parameters_group=False)
    with pytest.raises(SystemExit):
        args = parser.parse_args([
            '--input-file', 'test.h5',
            '--parameters-group', 'samples',
        ])


@pytest.mark.parametrize('input_, output', [
    (None, (None, dict())),
    (['a:1', 'b:2'], (['a', 'b'], {'a': '1', 'b': '2'})),
])
def test_parse_parameters_opt(input_, output):
    assert option_utils.parse_parameters_opt(input_) == output


@pytest.mark.parametrize('filename, type_', [
    ('test.txt', InferenceTXTFile),
    ('test.h5', InferenceFile),
    ('test.blah', TypeError),
])
def test_get_file_type(filename, type_):
    if issubclass(type_, Exception):
        with pytest.raises(type_):
            option_utils.get_file_type(filename)
    else:
        assert option_utils.get_file_type(filename) is type_


def test_add_plot_posterior_option_group():
    parser = argparse.ArgumentParser()
    option_utils.add_plot_posterior_option_group(parser)
    args = parser.parse_args([
        '--plot-marginal',
        '--marginal-percentiles', '5', '95',
        '--plot-scatter',
        '--plot-density',
        '--plot-contours',
        '--contour-percentiles', '5', '95',
        '--mins', 'a:1', 'b:2',
        '--maxs', 'a:2', 'b:3',
        '--expected-parameters', 'a:1', 'b:2',
        '--expected-parameters-color', 'b',
        '--plot-injection-parameters',
        '--injection-hdf-group', 'blah',
    ])
    for arg in ('marginal_percentiles', 'contour_percentiles'):
        assert list(map(type, getattr(args, arg))) == [float, float]


def test_plot_ranges_from_cli():
    parser = argparse.ArgumentParser()
    option_utils.add_plot_posterior_option_group(parser)
    args = parser.parse_args([
        '--mins', 'a:1', 'b:2',
        '--maxs', 'a:2', 'b:3',
    ])
    assert option_utils.plot_ranges_from_cli(args) == (
        {'a': 1, 'b': 2}, {'a': 2, 'b': 3})


def test_expected_parameters_from_cli():
    parser = argparse.ArgumentParser()
    option_utils.add_plot_posterior_option_group(parser)
    args = parser.parse_args([
        '--expected-parameters', 'a:1', 'b:2',
    ])
    assert option_utils.expected_parameters_from_cli(args) == {
        'a': 1, 'b': 2}


def test_add_scatter_option_group():
    parser = argparse.ArgumentParser()
    option_utils.add_scatter_option_group(parser)
    args = parser.parse_args([
        '--z-arg', 'snr',
        '--vmin', '10.1',
        '--vmax', '20.2',
        '--scatter-cmap', 'hot',
    ])
    assert args.vmin == 10.1
    assert args.vmax == 20.2
    with pytest.raises(SystemExit):
        parser.parse_args(['--z-arg', 'blah'])


def test_add_density_option_group():
    parser = argparse.ArgumentParser()
    option_utils.add_density_option_group(parser)
    parser.parse_args([
        '--density-cmap', 'hot',
        '--contour-color', 'blah',
        '--use-kombine-kde',
    ])
    assert parser.parse_args([]).density_cmap == 'viridis'
