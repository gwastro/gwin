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

from collections import OrderedDict

import pytest

from pycbc.waveform import generator

from core import get_parametrized_value

DEFAULT_CBC_PARAMETERS = OrderedDict([
    ("mass1", 30.0),
    ("mass2", 30.0),
    ("tc", 3.1),
    ("coa_phase", 1.1),
    ("ra", 0.1),
    ("dec", 0.1),
    ("polarization", 0.1),
    ("inclination", 0.1),
    ("distance", 300.0),
])


class WaveformMixin(object):
    """Mixin class to simplify waveform generation

    Test classes should declare the following variables before the test
    fixtures run (i.e. as class variables, or via `setup_class`):

    - ifos : List[str] = None
    - epoch : float = 0.
    - delta_f : float = 0.25
    - fmin : float = 30.
    """
    @pytest.fixture(scope='function')
    def fd_waveform_generator(self, request):
        wfgen_kw = {}

        parameters = getattr(self, 'parameters', DEFAULT_CBC_PARAMETERS)
        wfgen_kw['variable_args'] = list(parameters.keys())

        wfgen_kw['approximant'] = get_parametrized_value(
            request, 'approximant', request.config.getoption('approximants'))

        wfgen_kw['detectors'] = getattr(self, 'ifos', None)
        wfgen_kw['delta_f'] = getattr(self, 'delta_f', .25)
        wfgen_kw['f_lower'] = getattr(self, 'fmin', 30.)
        wfgen_kw['epoch'] = getattr(self, 'epoch', parameters.get('tc', 0.))

        return generator.FDomainDetFrameGenerator(
            generator.FDomainCBCGenerator, wfgen_kw.pop('epoch'), **wfgen_kw)

    @pytest.fixture(scope='function')
    def fd_waveform(self, request, fd_waveform_generator):
        parameters = {}
        for key in fd_waveform_generator.variable_args:
            try:
                parameters[key] = get_parametrized_value(
                    request, key, DEFAULT_CBC_PARAMETERS[key])
            except KeyError:
                raise ValueError(
                    "no parameter {0!r} provided to fd_waveform fixture. "
                    "This must be given either as a parametrized value, "
                    "or in the default {1!r}.parameters dict".format(
                        key, type(self).__name__))
        return fd_waveform_generator.generate(**parameters)
