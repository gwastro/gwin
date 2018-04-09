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

from .waveform import (WaveformMixin, DEFAULT_CBC_PARAMETERS)


class _TestBase(WaveformMixin):

    @classmethod
    def setup_class(cls):
        # basic data parameters
        cls.epoch = 0.
        cls.data_length = 4 # in seconds
        cls.sample_rate = 2048. # in Hertz
        cls.fmin = 30.0

        # derived data parameters
        cls.nsamp = int(cls.data_length * cls.sample_rate)
        cls.nsampf = int(cls.nsamp // 2) + 1
        cls.delta_f = 1.0 / cls.data_length

        # waveform parameters
        cls.ifos = ['H1', 'L1', 'V1']
        cls.parameters = DEFAULT_CBC_PARAMETERS.copy()
        cls.parameters.update({
            'tc': 3.1,  # some time in [epoch, data_length)
        })

    @pytest.fixture(scope='class')
    def random_data(self):
        from numpy.random import normal
        from pycbc.types import FrequencySeries
        return FrequencySeries(normal(size=self.nsamp).astype(complex),
                               epoch=self.epoch, delta_f=self.delta_f)

    def generate_psd(self, psd_func):
        return psd_func(self.nsampf, self.delta_f, self.fmin)

    @pytest.fixture(scope='class')
    def zdhp_psd(self):
        from pycbc.psd import aLIGOZeroDetHighPower
        return self.generate_psd(aLIGOZeroDetHighPower)
