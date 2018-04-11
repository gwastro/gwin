#!/usr/bin/env python

"""Print the usable waveform parameters
"""

# NOTE: the manual call to OrdereDict can be removed in favour of
# `ParameterList.description_dict` when gwastro/pycbc#2125 is merged
# and released

from collections import OrderedDict

from pycbc import waveform
from gwin.utils.sphinx import rst_dict_table

allparams = (waveform.fd_waveform_params +
             waveform.location_params +
             waveform.calibration_params)
paramdict = OrderedDict(allparams.descriptions)

print(rst_dict_table(
    paramdict,
    key_format='``\'{0}\'``'.format,
    header=('Parameter', 'Description'),
))
