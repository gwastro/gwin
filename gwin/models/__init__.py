# Copyright (C) 2018  Collin Capano
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


"""
This package provides classes and functions for evaluating Bayesian statistics
assuming various noise models.
"""


from .analytic import (TestEggbox, TestNormal, TestRosenbrock, TestVolcano)
from .gaussian_noise import (GaussianNoise,
                             MarginalizedPhaseGaussianNoise)


# Used to manage a model instance across multiple cores or MPI
_global_instance = None


def _call_global_model(*args, **kwds):
    """Private function for global model (needed for parallelization)."""
    return _global_instance(*args, **kwds)  # pylint:disable=not-callable


def read_from_config(cp, section="model", **kwargs):
    """Initializes a model from the given config file.

    The section must have a ``name`` argument. The name argument corresponds to
    the name of the class to initialize.

    Parameters
    ----------
    cp : WorkflowConfigParser
        Config file parser to read.
    section : str, optional
        The section to read. Default is "model".
    \**kwargs :
        All other keyword arguments are passed to the ``from_config`` method
        of the class specified by the name argument.

    Returns
    -------
    cls
        The initialized model.
    """
    # use the name to get the distribution
    name = cp.get(section, "name")
    return models[name].from_config(
        cp, section=section, **kwargs)


models = {_cls.name: _cls for _cls in (
    TestEggbox,
    TestNormal,
    TestRosenbrock,
    TestVolcano,
    GaussianNoise,
    MarginalizedPhaseGaussianNoise,
)}
