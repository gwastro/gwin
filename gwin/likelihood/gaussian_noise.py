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
This modules provides likelihood classes that assume the noise is Gaussian.
"""

import numpy
from scipy import special

from pycbc import filter
from pycbc.waveform import NoWaveformError
from pycbc.types import Array

from .base import DataBasedLikelihoodEvaluator


class GaussianLikelihood(DataBasedLikelihoodEvaluator):
    r"""Computes log likelihoods assuming the detectors' noise is Gaussian.

    With Gaussian noise the log likelihood functions for signal
    :math:`\log p(d|\Theta)` and for noise :math:`\log p(d|n)` are given by:

    .. math::

        \log p(d|\Theta) &=  -\frac{1}{2} \sum_i
            \left< h_i(\Theta) - d_i | h_i(\Theta) - d_i \right> \\
        \log p(d|n) &= -\frac{1}{2} \sum_i \left<d_i | d_i\right>

    where the sum is over the number of detectors, :math:`d_i` is the data in
    each detector, and :math:`h_i(\Theta)` is the model signal in each
    detector. The inner product is given by:

    .. math::

        \left<a | b\right> = 4\Re \int_{0}^{\infty}
            \frac{\tilde{a}(f) \tilde{b}(f)}{S_n(f)} \mathrm{d}f,

    where :math:`S_n(f)` is the PSD in the given detector.

    Note that the log prior-weighted likelihood ratio has one less term
    than the log posterior, since the :math:`\left<d_i|d_i\right>` term cancels
    in the likelihood ratio:

    .. math::

        \log \hat{\mathcal{L}} = \log p(\Theta) + \sum_i \left[
            \left<h_i(\Theta)|d_i\right> -
            \frac{1}{2} \left<h_i(\Theta)|h_i(\Theta)\right> \right]

    For this reason, by default this class returns ``logplr`` when called as a
    function instead of ``logposterior``. This can be changed via the
    ``set_callfunc`` method.

    Upon initialization, the data is whitened using the given PSDs. If no PSDs
    are given the data and waveforms returned by the waveform generator are
    assumed to be whitened. The likelihood function of the noise,

    .. math::

        p(d|n) = \frac{1}{2} \sum_i \left<d_i|d_i\right>,

    is computed on initialization and stored as the `lognl` attribute.

    By default, the data is assumed to be equally sampled in frequency, but
    unequally sampled data can be supported by passing the appropriate
    normalization using the ``norm`` keyword argument.

    For more details on initialization parameters and definition of terms, see
    ``BaseLikelihoodEvaluator``.

    Parameters
    ----------
    variable_args : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    waveform_generator : generator class
        A generator class that creates waveforms. This must have a ``generate``
        function which takes parameter values as keyword arguments, a
        detectors attribute which is a dictionary of detectors keyed by their
        names, and an epoch which specifies the start time of the generated
        waveform.
    data : dict
        A dictionary of data, in which the keys are the detector names and the
        values are the data (assumed to be unwhitened). The list of keys must
        match the waveform generator's detectors keys, and the epoch of every
        data set must be the same as the waveform generator's epoch.
    f_lower : float
        The starting frequency to use for computing inner products.
    psds : {None, dict}
        A dictionary of FrequencySeries keyed by the detector names. The
        dictionary must have a psd for each detector specified in the data
        dictionary. If provided, the inner products in each detector will be
        weighted by 1/psd of that detector.
    f_upper : {None, float}
        The ending frequency to use for computing inner products. If not
        provided, the minimum of the largest frequency stored in the data
        and a given waveform will be used.
    norm : {None, float or array}
        An extra normalization weight to apply to the inner products. Can be
        either a float or an array. If ``None``, ``4*data.values()[0].delta_f``
        will be used.
    **kwargs :
        All other keyword arguments are passed to ``BaseLikelihoodEvaluator``.

    Examples
    --------
    Create a signal, and set up the likelihood evaluator on that signal:

    >>> from pycbc import psd as pypsd
    >>> from pycbc.waveform.generator import (FDomainDetFrameGenerator,
                                              FDomainCBCGenerator)
    >>> import gwin
    >>> seglen = 4
    >>> sample_rate = 2048
    >>> N = seglen*sample_rate/2+1
    >>> fmin = 30.
    >>> m1, m2, s1z, s2z, tsig, ra, dec, pol, dist = (
            38.6, 29.3, 0., 0., 3.1, 1.37, -1.26, 2.76, 3*500.)
    >>> generator = FDomainDetFrameGenerator(
            FDomainCBCGenerator, 0.,
            variable_args=['tc'], detectors=['H1', 'L1'],
            delta_f=1./seglen, f_lower=fmin,
            approximant='SEOBNRv2_ROM_DoubleSpin',
            mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z,
            ra=ra, dec=dec, polarization=pol, distance=dist)
    >>> signal = generator.generate(tc=tsig)
    >>> psd = pypsd.aLIGOZeroDetHighPower(N, 1./seglen, 20.)
    >>> psds = {'H1': psd, 'L1': psd}
    >>> likelihood_eval = gwin.GaussianLikelihood(
            ['tc'], signal, generator, fmin, psds=psds, return_meta=False)

    Now compute the log likelihood ratio and prior-weighted likelihood ratio;
    since we have not provided a prior, these should be equal to each other:

    >>> likelihood_eval.loglr(tc=tsig)
    ArrayWithAligned(277.92945279883855)
    >>> likelihood_eval.logplr(tc=tsig)
    ArrayWithAligned(277.92945279883855)

    Compute the log likelihood and log posterior; since we have not
    provided a prior, these should both be equal to zero:

    >>> likelihood_eval.loglikelihood(tc=tsig)
    ArrayWithAligned(0.0)
    >>> likelihood_eval.logposterior(tc=tsig)
    ArrayWithAligned(0.0)

    Compute the SNR; for this system and PSD, this should be approximately 24:

    >>> likelihood_eval.snr(tc=tsig)
    ArrayWithAligned(23.576660187517593)

    Using the same likelihood evaluator, evaluate the log prior-weighted
    likelihood ratio at several points in time, check that the max is at tsig,
    and plot (note that we use the class as a function here, which defaults
    to calling ``logplr``):

    >>> from matplotlib import pyplot
    >>> times = numpy.arange(seglen*sample_rate)/float(sample_rate)
    >>> lls = numpy.array([likelihood_eval([t]) for t in times])
    >>> times[lls.argmax()]
    3.10009765625
    >>> fig = pyplot.figure(); ax = fig.add_subplot(111)
    >>> ax.plot(times, lls)
    [<matplotlib.lines.Line2D at 0x1274b5c50>]
    >>> fig.show()

    Create a prior and use it (see distributions module for more details):

    >>> from pycbc import distributions
    >>> uniform_prior = distributions.Uniform(tc=(tsig-0.2,tsig+0.2))
    >>> prior = distributions.JointDistribution(['tc'], uniform_prior)
    >>> likelihood_eval = gwin.GaussianLikelihood(['tc'],
            signal, generator, 20., psds=psds, prior=prior,
            return_meta=False)
    >>> likelihood_eval.logplr(tc=tsig)
    ArrayWithAligned(278.84574353071264)
    >>> likelihood_eval.logposterior(tc=tsig)
    ArrayWithAligned(0.9162907318741418)
    """
    name = 'gaussian'

    def __init__(self, variable_args, data, waveform_generator,
                 f_lower, psds=None, f_upper=None, norm=None,
                 **kwargs):
        # set up the boiler-plate attributes; note: we'll compute the
        # log evidence later
        super(GaussianLikelihood, self).__init__(variable_args, data,
                                                 waveform_generator, **kwargs)
        # check that the data and waveform generator have the same detectors
        if (sorted(waveform_generator.detectors.keys()) !=
                sorted(self._data.keys())):
            raise ValueError(
                "waveform generator's detectors ({0}) does not "
                "match data ({1})".format(
                    ','.join(sorted(waveform_generator.detector_names)),
                    ','.join(sorted(self._data.keys()))))
        # check that the data and waveform generator have the same epoch
        if any(waveform_generator.epoch != d.epoch
               for d in self._data.values()):
            raise ValueError("waveform generator does not have the same epoch "
                             "as all of the data sets.")
        # check that the data sets all have the same lengths
        dlens = numpy.array([len(d) for d in data.values()])
        if not all(dlens == dlens[0]):
            raise ValueError("all data must be of the same length")
        # we'll use the first data set for setting values
        d = data.values()[0]
        N = len(d)
        # figure out the kmin, kmax to use
        kmin, kmax = filter.get_cutoff_indices(f_lower, f_upper, d.delta_f,
                                               (N-1)*2)
        self._kmin = kmin
        self._kmax = kmax
        if norm is None:
            norm = 4*d.delta_f
        # we'll store the weight to apply to the inner product
        if psds is None:
            w = Array(numpy.sqrt(norm)*numpy.ones(N))
            self._weight = {det: w for det in data}
        else:
            # temporarily suppress numpy divide by 0 warning
            numpysettings = numpy.seterr(divide='ignore')
            self._weight = {det: Array(numpy.sqrt(norm/psds[det]))
                            for det in data}
            numpy.seterr(**numpysettings)
        # whiten the data
        for det in self._data:
            self._data[det][kmin:kmax] *= self._weight[det][kmin:kmax]
        # compute the log likelihood function of the noise and save it
        self.set_lognl(-0.5*sum([
            d[kmin:kmax].inner(d[kmin:kmax]).real
            for d in self._data.values()]))
        # set default call function to logplor
        self.set_callfunc('logplr')

    def loglr(self, **params):
        r"""Computes the log likelihood ratio,

        .. math::

            \log \mathcal{L}(\Theta) = \sum_i
                \left<h_i(\Theta)|d_i\right> -
                \frac{1}{2}\left<h_i(\Theta)|h_i(\Theta)\right>,

        at the given point in parameter space :math:`\Theta`.

        Parameters
        ----------
        \**params :
            The keyword arguments should give the values of each parameter to
            evaluate.

        Returns
        -------
        numpy.float64
            The value of the log likelihood ratio evaluated at the given point.
        """
        lr = 0.
        try:
            wfs = self._waveform_generator.generate(**params)
        except NoWaveformError:
            # if no waveform was generated, just return 0
            return lr
        for det, h in wfs.items():
            # the kmax of the waveforms may be different than internal kmax
            kmax = min(len(h), self._kmax)
            # whiten the waveform
            if self._kmin >= kmax:
                # if the waveform terminates before the filtering low frequency
                # cutoff, there is nothing to filter, so just go onto the next
                continue
            slc = slice(self._kmin, kmax)
            h[self._kmin:kmax] *= self._weight[det][slc]
            lr += (
                # <h, d>
                self.data[det][slc].inner(h[slc]).real -
                # <h, h>/2.
                0.5*h[slc].inner(h[slc]).real
            )
        return numpy.float64(lr)

    def loglikelihood(self, **params):
        r"""Computes the log likelihood of the paramaters,

        .. math::

            p(d|\Theta) = -\frac{1}{2}\sum_i
                \left<h_i(\Theta) - d_i | h_i(\Theta) - d_i\right>

        Parameters
        ----------
        \**params :
            The keyword arguments should give the values of each parameter to
            evaluate.

        Returns
        -------
        float
            The value of the log likelihood evaluated at the given point.
        """
        # since the loglr has fewer terms, we'll call that, then just add
        # back the noise term that canceled in the log likelihood ratio
        return self.loglr(**params) + self._lognl

    def logposterior(self, **params):
        """Computes the log-posterior probability at the given point in
        parameter space.

        Parameters
        ----------
        \**params :
            The keyword arguments should give the values of each parameter to
            evaluate.

        Returns
        -------
        float
            The value of the log-posterior evaluated at the given point in
            parameter space.
        metadata : tuple
            If ``return_meta``, the prior and likelihood ratio as a tuple.
            Otherwise, just returns the log-posterior.
        """
        # since the logplr has fewer terms, we'll call that, then just add
        # back the noise term that canceled in the log likelihood ratio
        logplr = self.logplr(**params)
        if self.return_meta:
            logplr, (pr, lr, lj) = logplr
        else:
            pr = lr = lj = None
        return self._formatreturn(logplr + self._lognl, prior=pr, loglr=lr,
                                  logjacobian=lj)


class MarginalizedPhaseGaussianLikelihood(GaussianLikelihood):
    r"""The likelihood is analytically marginalized over phase.

    This class can be used with signal models that can be written as:

    .. math::

        \tilde{h}(f; \Theta, \phi) = A(f; \Theta)e^{i\Psi(f; \Theta) + i \phi},

    where :math:`\phi` is an arbitrary phase constant. This phase constant
    can be analytically marginalized over with a uniform prior as follows:
    assuming the noise is stationary and Gaussian (see `GaussianLikelihood`
    for details), the posterior is:

    .. math::

        p(\Theta,\phi|d)
            &\propto p(\Theta)p(\phi)p(d|\Theta,\phi) \\
            &\propto p(\Theta)\frac{1}{2\pi}\exp\left[
                -\frac{1}{2}\sum_{i}^{N_D} \left<
                    h_i(\Theta,\phi) - d_i, h_i(\Theta,\phi) - d_i
                \right>\right].

    Here, the sum is over the number of detectors :math:`N_D`, :math:`d_i`
    and :math:`h_i` are the data and signal in detector :math:`i`,
    respectively, and we have assumed a uniform prior on :math:`phi \in [0,
    2\pi)`. With the form of the signal model given above, the inner product
    in the exponent can be written as:

    .. math::

        -\frac{1}{2}\left<h_i - d_i, h_i- d_i\right>
            &= \left<h_i, d_i\right> -
               \frac{1}{2}\left<h_i, h_i\right> -
               \frac{1}{2}\left<d_i, d_i\right> \\
            &= \Re\left\{O(h^0_i, d_i)e^{-i\phi}\right\} -
               \frac{1}{2}\left<h^0_i, h^0_i\right> -
               \frac{1}{2}\left<d_i, d_i\right>,

    where:

    .. math::

        h_i^0 &\equiv \tilde{h}_i(f; \Theta, \phi=0); \\
        O(h^0_i, d_i) &\equiv 4 \int_0^\infty
            \frac{\tilde{h}_i^*(f; \Theta,0)\tilde{d}_i(f)}{S_n(f)}\mathrm{d}f.

    Gathering all of the terms that are not dependent on :math:`\phi` together:

    .. math::

        \alpha(\Theta, d) \equiv \exp\left[-\frac{1}{2}\sum_i
            \left<h^0_i, h^0_i\right> + \left<d_i, d_i\right>\right],

    we can marginalize the posterior over :math:`\phi`:

    .. math::

        p(\Theta|d)
            &\propto p(\Theta)\alpha(\Theta,d)\frac{1}{2\pi}
                     \int_{0}^{2\pi}\exp\left[\Re \left\{
                         e^{-i\phi} \sum_i O(h^0_i, d_i)
                     \right\}\right]\mathrm{d}\phi \\
            &\propto p(\Theta)\alpha(\Theta, d)\frac{1}{2\pi}
                     \int_{0}^{2\pi}\exp\left[
                         x(\Theta,d)\cos(\phi) + y(\Theta, d)\sin(\phi)
                     \right]\mathrm{d}\phi.

    The integral in the last line is equal to :math:`2\pi I_0(\sqrt{x^2+y^2})`,
    where :math:`I_0` is the modified Bessel function of the first kind. Thus
    the marginalized log posterior is:

    .. math::

        \log p(\Theta|d) \propto \log p(\Theta) +
            I_0\left(\left|\sum_i O(h^0_i, d_i)\right|\right) -
            \frac{1}{2}\sum_i\left[ \left<h^0_i, h^0_i\right> -
                                    \left<d_i, d_i\right> \right]

    This class computes the above expression for the log likelihood.
    """
    name = 'marginalized_phase'

    def loglr(self, **params):
        r"""Computes the log likelihood ratio,

        .. math::

            \log \mathcal{L}(\Theta) =
                I_0 \left(\left|\sum_i O(h^0_i, d_i)\right|\right) -
                \frac{1}{2}\left<h^0_i, h^0_i\right>,

        at the given point in parameter space :math:`\Theta`.

        Parameters
        ----------
        \**params :
            The keyword arguments should give the values of each parameter to
            evaluate.

        Returns
        -------
        numpy.float64
            The value of the log likelihood ratio evaluated at the given point.
        """
        try:
            wfs = self._waveform_generator.generate(**params)
        except NoWaveformError:
            # if no waveform was generated, just return 0
            return 0.
        hh = 0.
        hd = 0j
        for det, h in wfs.items():
            # the kmax of the waveforms may be different than internal kmax
            kmax = min(len(h), self._kmax)
            # whiten the waveform
            if self._kmin >= kmax:
                # if the waveform terminates before the filtering low frequency
                # cutoff, there is nothing to filter, so just go onto the next
                continue
            h[self._kmin:kmax] *= self._weight[det][self._kmin:kmax]
            hh += h[self._kmin:kmax].inner(h[self._kmin:kmax]).real
            hd += self.data[det][self._kmin:kmax].inner(h[self._kmin:kmax])
        hd = abs(hd)
        return numpy.log(special.i0e(hd)) + hd - 0.5*hh
