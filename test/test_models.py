# Copyright (C) 2018 Duncan Macleod, Charlie Hoy
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

from pycbc.workflow import WorkflowConfigParser
from pycbc import distributions

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


class _TestBaseModel(_TestBase):
    """Tests BaseModel."""

    TEST_CLASS = models.base.BaseModel
    CALL_CLASS = models.CallModel
    DEFAULT_CALLSTAT = 'logposterior'

    @classmethod
    def setup_class(cls):
        super(_TestBaseModel, cls).setup_class()

        cls.data = range(10)

    @pytest.fixture(scope='function')
    def simple(self, request):
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

    def test_default_stats(self, simple):
        # tests that the default stats always has at least logjacobian,
        # logprior, and loglikelihooe
        expected = set(['logjacobian', 'logprior', 'loglikelihood'])
        assert expected.issubset(set(simple.default_stats))


# -- GaussianNoise -------------------------------------------------------

class TestGaussianNoise(_TestBaseModel):
    TEST_CLASS = models.GaussianNoise
    DEFAULT_CALLSTAT = 'logplr'

    @pytest.fixture(scope='function')
    def simple(self, random_data, fd_waveform_generator, request):
        data = {ifo: random_data for ifo in self.ifos}
        model = self.TEST_CLASS([], data, fd_waveform_generator,
                                f_lower=self.fmin)
        return self.CALL_CLASS(model, self.DEFAULT_CALLSTAT)

    @pytest.fixture(scope='function')
    def full(self, fd_waveform, fd_waveform_generator, zdhp_psd, request):
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

# -- MarginalizedGaussianNoise --------------------------------------------


class TestMarginalizedGaussianNoise(TestGaussianNoise):
    """Tests MarginalizedGaussianNoise."""
    TEST_CLASS = models.MarginalizedGaussianNoise
    DEFAULT_CALLSTAT = 'logplr'

    @pytest.fixture(scope='function')
    def simple(self, random_data, fd_waveform_generator):
        marg_prior = [distributions.Uniform(distance=(50, 5000))]
        data = {ifo: random_data for ifo in self.ifos}
        model = self.TEST_CLASS([], data, fd_waveform_generator,
                                f_lower=self.fmin,
                                distance_marginalization=True,
                                marg_prior=marg_prior)
        return self.CALL_CLASS(model, self.DEFAULT_CALLSTAT)

    @pytest.fixture(scope='function')
    def full(self, fd_waveform, fd_waveform_generator, zdhp_psd, request):
        marg_prior = [distributions.Uniform(distance=(50, 5000))]
        model = self.TEST_CLASS(
            ['tc'], fd_waveform, fd_waveform_generator, self.fmin,
            psds={ifo: zdhp_psd for ifo in self.ifos},
            distance_marginalization=True, marg_prior=marg_prior)
        return self.CALL_CLASS(model, self.DEFAULT_CALLSTAT,
                               return_all_stats=False)

    def test_from_config(self, random_data, request):
        """Test the function which loads data from a configuration file. Here
        we assume we are just marginalizing over distance with a uniform prior
        [50, 5000)
        """
        param = {"approximant": "IMRPhenomPv2", "f_lower": "20", "f_ref": "20",
                 "ra": "1.5", "dec": "-0.5", "polarization": "0.5"}

        cp = WorkflowConfigParser()
        cp.add_section("model")
        cp.set("model", "name", "marginalized_gaussian_noise")
        cp.set("model", "distance_marginalization", "")
        cp.add_section("marginalized_prior-distance")
        cp.set("marginalized_prior-distance", "name", "uniform")
        cp.set("marginalized_prior-distance", "min-distance", "50")
        cp.set("marginalized_prior-distance", "max-distance", "5000")
        cp.add_section("variable_params")
        cp.set("variable_params", "tc", "")
        cp.add_section("static_params")
        for key in param.keys():
            cp.set("static_params", key, param[key])
        cp.add_section("prior-tc")
        cp.set("prior-tc", "name", "uniform")
        cp.set("prior-tc", "min-tc", "1126259462.32")
        cp.set("prior-tc", "max-tc", "1126259462.52")

        data = {ifo: random_data for ifo in self.ifos}
        model = models.MarginalizedGaussianNoise.from_config(cp, data)
        marg_priors = model._marg_prior
        keys = list(marg_priors.keys())
        assert keys[0] == "distance"
        assert model._margdist
        assert marg_priors["distance"].bounds["distance"].min == 50.0
        assert marg_priors["distance"].bounds["distance"].max == 5000.0
