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

import pytest

import numpy
from numpy import isclose

from gwin import likelihood as gwin_likelihood

from utils import _TestBase


class TestNoPrior(object):
    TEST_CLASS = gwin_likelihood._NoPrior

    def test_apply_boundary_conditions(self):
        p = self.TEST_CLASS()
        assert p.apply_boundary_conditions(a=1, b=2) == {'a': 1, 'b': 2}

    def test_call(self):
        p = self.TEST_CLASS()
        assert p() == 0.


class TestBaseLikelihoodEvaluator(_TestBase):
    TEST_CLASS = gwin_likelihood.BaseLikelihoodEvaluator

    @classmethod
    def setup_class(cls):
        super(TestBaseLikelihoodEvaluator, cls).setup_class()

        cls.data = range(10)

    @pytest.fixture(scope='function')
    def simple(self):
        return self.TEST_CLASS([])

    def test_defaults(self, simple):
        assert simple.variable_args is tuple()
        assert isinstance(simple._prior, gwin_likelihood._NoPrior)

    @pytest.mark.parametrize('transforms, params, result', [
        (None, {}, 0.),  # defaults
    ])
    def test_logjacobian(self, simple, transforms, params, result):
        _st = simple.sampling_transforms
        simple._sampling_transforms = transforms
        try:
            assert simple.logjacobian(**params) == result

        finally:
            simple._sampling_transforms = _st

    @pytest.mark.parametrize('return_meta, result', [
        (False, 1),
        (True, (1, (2, 3, 4))),
    ])
    def test_formatreturn(self, simple, return_meta, result):
        _rm = simple.return_meta
        simple.return_meta = return_meta
        try:
            assert simple._formatreturn(1, 2, 3, 4) == result
        finally:
            simple.return_meta = _rm

    def test_set_callfunc(self):
        _callfunc = self.TEST_CLASS._callfunc
        try:
            self.TEST_CLASS.set_callfunc('logplr')
            assert self.TEST_CLASS._callfunc == self.TEST_CLASS.logplr

        finally:
            self.TEST_CLASS._callfunc = _callfunc


# -- GaussianLikelihood -------------------------------------------------------

class TestGaussianLikelihood(TestBaseLikelihoodEvaluator):
    TEST_CLASS = gwin_likelihood.GaussianLikelihood

    @pytest.fixture(scope='function')
    def simple(self, random_data, fd_waveform_generator):
        data = {ifo: random_data for ifo in self.ifos}
        return self.TEST_CLASS([], data, fd_waveform_generator,
                               f_lower=self.fmin)

    @pytest.fixture(scope='function')
    def full(self, fd_waveform, fd_waveform_generator, zdhp_psd):
        return self.TEST_CLASS(
            ['tc'], fd_waveform, fd_waveform_generator, self.fmin,
            psds={ifo: zdhp_psd for ifo in self.ifos}, return_meta=False)

    @pytest.mark.parametrize('callfunc', ['logplr', 'logposterior'])
    def test_call_1d_noprior(self, full, approximant, callfunc):
        # set the calling function
        full.set_callfunc(callfunc)

        # create times to evaluate over
        target = self.parameters['tc']
        tstart = self.parameters['tc'] - self.data_length / 2.
        times = tstart + numpy.arange(self.nsamp) / self.sample_rate

        # evaluate likelihood and check recovery
        likelihoods = [full([t]) for t in times]
        assert isclose(times[numpy.argmax(likelihoods)], target)
