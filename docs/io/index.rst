.. _gwin-io:

.. currentmodule:: gwin.io

#################
File input/output
#################

====================================================
HDF output file handler (``gwin.io.InferenceFile``)
====================================================

.. currentmodule:: gwin.io.hdf

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

    samples = fp.read_samples("distance", walkers=0,
                              thin_start=0, thin_end=-1, thin_interval=1)
    print(samples.distance)

Some standard parameters that are derived from the variable arguments
(listed via :attr:`fp.variable_params <InferenceFile.variable_params>`) can also
be retrieved.
For example, if ``fp.variable_params`` includes ``'mass1'`` and ``'mass2'``,
then you can retrieve the chirp mass with::

   samples = fp.read_samples("mchirp")
   print(samples.mchirp)

Some standard parameters that are derived from the variable arguments
(listed via :attr:`fp.variable_params <InferenceFile.variable_params>`) can also
be retrieved.
For example, if ``fp.variable_params`` includes ``'mass1'`` and ``'mass2'``,
then you can retrieve the chirp mass with::

   samples = fp.read_samples("mchirp")
   print(samples.mchirp)

In this case, :meth:`fp.read_samples <InferenceFile.read_samples>` will
retrieve ``mass1`` and ``mass2`` (since they are needed to compute chirp mass);
``samples.mchirp`` then returns an array of the chirp mass computed from
``mass1`` and ``mass2``.

For more information, including the list of predefined derived parameters,
see the class reference for :class:`InferenceFile`.

=============
API Reference
=============

.. autosummary::

   ~gwin.io.hdf.InferenceFile
   ~gwin.io.txt.InferenceTXTFile
