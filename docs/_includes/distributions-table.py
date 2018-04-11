#!/usr/bin/env python

"""Print an RST table of available distributions from
"""

from pycbc.distributions import distribs
from gwin.utils.sphinx import rst_dict_table


def val_format(class_):
    return ':py:class:`{0}.{1}`'.format(class_.__module__, class_.__name__)


print(rst_dict_table(distribs, key_format='``\'{0}\'``'.format,
                     val_format=val_format))
