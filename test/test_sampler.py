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

from gwin import models
from gwin import sampler

from utils import _TestBase

MODELS = [n for n in models.models
          if not n.startswith('test_')]


# -- parametrisation ----------------------------------------------------------

def with_model():
    return pytest.mark.parametrize('model', MODELS,
                                   indirect=['model'])


# -- tests --------------------------------------------------------------------

class TestSamplers(_TestBase):
    # -- setup ----------------------------------

    @classmethod
    def setup_class(cls):
        super(TestSamplers, cls).setup_class()

        # fake command-line options
        class Arguments(object):
            ntemps = 2
            nwalkers = 30
            niterations = 4
            update_interval = 2
            nprocesses = 2

        cls.opts = Arguments()

    @pytest.fixture
    def prior_eval(self):
        """ Returns the joint distribution class initialized with a set of
        pre-defined distributions for each parameters.
        """
        parameters, values = zip(*self.parameters.items())
        prior_dists = []
        for param, val in zip(parameters, values):
            if param in ["mass1", "mass2"]:
                dist = distributions.Uniform(**{param: (6, 50)})
            elif param in ["inclination", "dec"]:
                dist = distributions.SinAngle(**{param: None})
            elif param in ["polarization", "ra", "coa_phase"]:
                dist = distributions.Uniform(**{param: (0, 2 * 3.1415)})
            elif param in ["distance"]:
                dist = distributions.UniformRadius(distance=(val - 100,
                                                             val + 300))
            elif param in ["spin1x", "spin1y", "spin1z",
                           "spin2x", "spin2y", "spin2z"]:
                dist = distributions.Uniform(**{param: (-0.1, 0.1)})
            elif param in ["tc"]:
                dist = distributions.Uniform(tc=(val - 0.2, val + 0.2))
            else:
                raise KeyError("Do not recognize parameter %s" % param)
            prior_dists.append(dist)
        return distributions.JointDistribution(parameters, *prior_dists)

    @pytest.fixture
    def model(self, fd_waveform, fd_waveform_generator, prior_eval,
              zdhp_psd, request):
        eval_class = models.models[request.param]
        model = eval_class(
            fd_waveform_generator.variable_args,
            fd_waveform, fd_waveform_generator,
            self.fmin, psds={ifo: zdhp_psd for ifo in self.ifos},
            prior=prior_eval)
        return models.CallModel(model, "logposterior", return_all_stats=False)

    # -- actual tests ---------------------------

    @with_model()
    @pytest.mark.parametrize('sampler_class', sampler.samplers.values())
    def test_sampler(self, model, approximant, sampler_class):
        """Runs each sampler for 4 iterations.
        """
        # init sampler
        s = sampler_class.from_cli(self.opts, model)
        s.set_p0()

        # run
        s.run(self.opts.niterations)
