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

"""Unit tests for gwin.io
"""

import pytest

import numpy
from numpy.testing import assert_array_equal

from gwin.io.hdf import (InferenceFile, _PosteriorOnlyParser)
from gwin.sampler import MCMCSampler


# -- gwin.io.hdf --------------------------------------------------------------

class TestInferenceFile(object):
    TEST_CLASS = InferenceFile

    @classmethod
    @pytest.fixture
    def empty(cls):
        """Create and return a new, empty, in-memory `InferenceFile`
        """
        with cls.TEST_CLASS('testfile', driver='core',
                            backing_store=False) as h5f:
            yield h5f

    @pytest.mark.parametrize('prop', [
        'variable_args',
        'sampling_args',
        'lognl',
        'niterations',
        'burn_in_iterations',
        'is_burned_in',
        'nwalkers',
        'acl',
        'resume_points',
    ])
    def test_property(self, empty, prop):
        empty.attrs[prop] = 'test value'
        assert getattr(empty, prop) == 'test value'

    def test_posterior_only(self, empty):
        assert empty.posterior_only is False
        empty.attrs['posterior_only'] = True
        assert empty.posterior_only

    def test_sampler_name(self, empty):
        empty.attrs['sampler'] = 'mcmc'
        assert empty.sampler_name == 'mcmc'

    def test_sampler_class(self, empty):
        assert empty.sampler_class is None
        empty.attrs['sampler'] = 'mcmc'
        assert empty.sampler_class is MCMCSampler

    @pytest.mark.parametrize('postonly, parser', [
        (False, None),
        (True, _PosteriorOnlyParser),
    ])
    def test_samples_parser(self, empty, postonly, parser):
        empty.attrs['posterior_only'] = postonly
        assert empty.samples_parser is parser

    def likelihood_eval_name(self, empty):
        empty.attrs['likelihood_evaluator'] = 'anything'
        assert empty.likelihood_eval_name == 'anything'

    def test_static_args(self, empty):
        empty.attrs['arg1'] = 1
        empty.attrs['arg2'] = 2
        empty.attrs['static_args'] = ('arg1', 'arg2')
        assert empty.static_args == {'arg1': 1, 'arg2': 2}

    def test_cmd(self, empty):
        empty.attrs['cmd'] = 'test command'
        assert empty.cmd == 'test command'
        empty.attrs['cmd'] = ['test command 1', 'test command 2']
        assert empty.cmd == 'test command 2'

    def test_log_evidence(self, empty):
        empty.attrs.update({
            'log_evidence': 1,
            'dlog_evidence': 2,
        })
        assert empty.log_evidence == (1, 2)

    def _test_read(self, h5file, reader, group, *args, **kwargs):
        # create data
        h5file.attrs.update({
            'sampler': 'mcmc',
            'burn_in_iterations': 2,
            'niterations': 5,
        })
        h5g = h5file.create_group(group)
        data = numpy.random.rand(5, 5, 5)
        h5g.create_dataset('x', data=data)

        # read data
        out = getattr(h5file, reader)(*args, **kwargs)
        assert out.fieldnames == ('x', )
        assert out.shape == (5 * 5 * 3, )
        assert_array_equal(out.view(out.dtype[0]),
                           data[:, 2:].flatten())

    @pytest.mark.parametrize('group, samples_group', [
        (TEST_CLASS.samples_group, None),
        ('test', 'test'),
    ])
    def test_read_samples(self, empty, group, samples_group):
        return self._test_read(empty, 'read_samples', group,
                               ['x'], samples_group=samples_group)

    def test_read_likelihood_stats(self, empty):
        return self._test_read(empty, 'read_likelihood_stats',
                               empty.stats_group)
