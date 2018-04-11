# Copyright (C) 2018  Collin Capano
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

"""
====================================================
HDF output file handler (``gwin.io.InferenceFile``)
====================================================

The executable ``gwin`` will write a HDF file with all the samples from
each walker along with the PSDs and some meta-data about the sampler.
To read the output file::

    from gwin.io import InferenceFile
    fp = InferenceFile("example.h5", "r")

To get all samples for ``distance`` from the first walker you can do::

    samples = fp.read_samples("distance", walkers=0)
    print(samples.distance)

The :meth:`InferenceFile.read_samples` method includes the options to thin
the samples.
By default the function will return samples beginning at the end of the
burn-in to the last written sample, and will use the autocorrelation
length (ACL) calculated by ``gwin`` to select the indepdedent samples.
You can supply ``thin_start``, ``thin_end``, and ``thin_interval`` to
override this.
To read all samples you would do::

    samples = fp.read_samples("distance", walkers=0, thin_start=0, thin_end=-1, thin_interval=1)
    print(samples.distance)

Some standard parameters that are derived from the variable arguments
(listed via :attr:`fp.variable_args <InferenceFile.variable_args>`) can also be retrieved.
For example, if ``fp.variable_args`` includes ``'mass1'`` and ``'mass2'``,
then you can retrieve the chirp mass with::

   samples = fp.read_samples("mchirp")
   print(samples.mchirp)

In this case, :meth:`fp.read_samples <InferenceFile.read_samples>` will
retrieve ``mass1`` and ``mass2`` (since they are needed to compute chirp mass);
``samples.mchirp`` then returns an array of the chirp mass computed from
``mass1`` and ``mass2``.

For more information, including the list of predefined derived parameters,
see the class refernce for :class:`gwin.io.InferenceFile`.

=============
API Reference
=============

.. autosummary::
   :toctree: ../api/

   InferenceFile
   InferenceTXTFile
"""

from .hdf import InferenceFile
from .txt import InferenceTXTFile
