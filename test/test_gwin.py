# Copyright (C) 2017 Christopher M. Biwer
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

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""
These are the unittests for samplers in the gwin subpackage.
"""

from collections import OrderedDict

import numpy

import pytest

from pycbc import distributions
from pycbc.psd import analytical
from pycbc.waveform import generator

from gwin import likelihood
from gwin import sampler

APPROXIMANTS = [
    'IMRPhenomPv2',
]
CBC_TEST_PARAMETERS = OrderedDict([
    ("mass1", 30.0),
    ("mass2", 30.0),
    ("tc", 100.0),
    ("coa_phase", 1.1),
    ("spin1x", 0.0),
    ("spin1y", 0.0),
    ("spin1z", 0.0),
    ("spin2x", 0.0),
    ("spin2y", 0.0),
    ("spin2z", 0.0),
    ("ra", 0.1),
    ("dec", 0.1),
    ("polarization", 0.1),
    ("inclination", 0.1),
    ("distance", 300.0),
])
LIKELIHOOD_EVALUATORS = [n for n in likelihood.likelihood_evaluators if
                         not n.startswith('test_')]


class TestSamplers(object):
    # -- setup ----------------------------------

    @classmethod
    def setup_class(cls):
        cls.ifos = ['H1', 'L1', 'V1']
        cls.epoch = CBC_TEST_PARAMETERS['tc']

        # PSD params
        cls.data_length = 4 # in seconds
        cls.sample_rate = 2048 # in Hertz
        cls.fdomain_samples = cls.data_length * cls.sample_rate / 2 + 1
        cls.delta_f = 1.0 / cls.data_length
        cls.fmin = 30.0

        # fake command-line options
        class Arguments(object):
            ntemps = 2
            nwalkers = 30
            niterations = 4
            update_interval = 2
            nprocesses = 2

        cls.opts = Arguments()

    @pytest.fixture(scope='class')
    def psd(self):
        return analytical.aLIGOZeroDetHighPower(
            self.fdomain_samples, self.delta_f, self.fmin)

    @pytest.fixture(scope='class', params=APPROXIMANTS)
    def waveform_generator(self, request):
        return generator.FDomainDetFrameGenerator(
            generator.FDomainCBCGenerator, self.epoch,
            variable_args=list(CBC_TEST_PARAMETERS.keys()),
            detectors=self.ifos,
            delta_f=self.delta_f,
            f_lower=self.fmin,
            approximant=request.param,
        )

    @pytest.fixture
    def waveform(self, waveform_generator):
        return waveform_generator.generate(**CBC_TEST_PARAMETERS)

    @pytest.fixture
    def prior_eval(self):
        """ Returns the prior evaluator class initialized with a set of
        pre-defined distributions for each parameters.
        """
        parameters, values = zip(*CBC_TEST_PARAMETERS.items())
        prior_dists = []
        for param, val in zip(parameters, values):
            if param in ["mass1", "mass2"]:
                dist = distributions.Uniform(**{param : (6, 50)})
            elif param in ["inclination", "dec"]:
                dist = distributions.SinAngle(**{param : None})
            elif param in ["polarization", "ra", "coa_phase"]:
                dist = distributions.Uniform(**{param : (0, 2 * 3.1415)})
            elif param in ["distance"]:
                dist = distributions.UniformRadius(distance=(val - 100,
                                                             val + 300))
            elif param in ["spin1x", "spin1y", "spin1z",
                           "spin2x", "spin2y", "spin2z",]:
                dist = distributions.Uniform(**{param : (-0.1, 0.1)})
            elif param in ["tc"]:
                dist = distributions.Uniform(tc=(val - 0.2, val + 0.2))
            else:
                raise KeyError("Do not recognize parameter %s" % param)
            prior_dists.append(dist)
        return distributions.JointDistribution(parameters, *prior_dists)

    @pytest.fixture(params=LIKELIHOOD_EVALUATORS)
    def likelihood_eval(self, waveform_generator, waveform, prior_eval, psd,
                        request):
        eval_class = likelihood.likelihood_evaluators[request.param]
        return eval_class(
            waveform_generator.variable_args, waveform_generator, waveform,
            self.fmin, psds={ifo: psd for ifo in self.ifos},
            prior=prior_eval, return_meta=False)

    def test_likelihood_eval(self, likelihood_eval):
        out = likelihood_eval(list(CBC_TEST_PARAMETERS.values()))
        assert out.dtype.type is numpy.float64

    @pytest.mark.parametrize('sampler_class', sampler.samplers.values())
    def test_sampler(self, sampler_class, likelihood_eval):
        """Runs each sampler for 4 iterations.
        """
        # init sampler
        s = sampler_class.from_cli(self.opts, likelihood_eval)
        s.set_p0()

        # run
        s.run(self.opts.niterations)
