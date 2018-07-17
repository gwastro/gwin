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

from gwin import models

from utils import _TestBase


class TestNoPrior(object):
    TEST_CLASS = models.base._NoPrior

    def test_apply_boundary_conditions(self):
        p = self.TEST_CLASS()
        assert p.apply_boundary_conditions(a=1, b=2) == {'a': 1, 'b': 2}

    def test_call(self):
        p = self.TEST_CLASS()
        assert p() == 0.


class TestBaseModel(_TestBase):
    """Tests BaseModel."""

    class DummyBase(models.base.BaseModel):
        """BaseModel cannot be initialized because it is an abstract base
        class. It should only require ``_loglikelihood`` to be defined. This
        tests that.
        """
        def _loglikelihood(self):
            return 0.

    TEST_CLASS = DummyBase
    CALL_CLASS = models.CallModel
    DEFAULT_CALLSTAT = 'logposterior'

    @classmethod
    def setup_class(cls):
        super(TestBaseModel, cls).setup_class()

        cls.data = range(10)

    @pytest.fixture(scope='function')
    def simple(self):
        model = self.TEST_CLASS([])
        return self.CALL_CLASS(model, self.DEFAULT_CALLSTAT)

    def test_defaults(self, simple):
        assert simple.variable_params is tuple()
        assert isinstance(simple.prior_distribution, models.base._NoPrior)

    @pytest.mark.parametrize('transforms, params, result', [
        (None, {}, 0.),  # defaults
    ])
    def test_logjacobian(self, simple, transforms, params, result):
        _st = simple.sampling_transforms
        simple.sampling_transforms = transforms
        try:
            simple.update(**params)
            assert simple.logjacobian == result

        finally:
            simple._sampling_transforms = _st


# -- GaussianNoise -------------------------------------------------------

class TestGaussianNoise(TestBaseModel):
    TEST_CLASS = models.GaussianNoise
    DEFAULT_CALLSTAT = 'logplr'

    @pytest.fixture(scope='function')
    def simple(self, random_data, fd_waveform_generator):
        data = {ifo: random_data for ifo in self.ifos}
        model = self.TEST_CLASS([], data, fd_waveform_generator,
                                f_lower=self.fmin)
        return self.CALL_CLASS(model, self.DEFAULT_CALLSTAT)

    @pytest.fixture(scope='function')
    def full(self, fd_waveform, fd_waveform_generator, zdhp_psd):
        model = self.TEST_CLASS(
            ['tc'], fd_waveform, fd_waveform_generator, self.fmin,
            psds={ifo: zdhp_psd for ifo in self.ifos})
        return self.CALL_CLASS(model, self.DEFAULT_CALLSTAT,
                               return_all_stats=False)

    @pytest.mark.parametrize('callstat', ['logplr', 'logposterior'])
    def test_call_1d_noprior(self, full, approximant, callstat):
        # set the calling function
        full.callstat = callstat

        # create times to evaluate over
        target = self.parameters['tc']
        tstart = self.parameters['tc'] - self.data_length / 2.
        times = tstart + numpy.arange(self.nsamp) / self.sample_rate

        # evaluate model and check recovery
        stats = [full([t]) for t in times]
        assert isclose(times[numpy.argmax(stats)], target)
