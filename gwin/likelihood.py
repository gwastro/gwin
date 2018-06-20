# Copyright (C) 2016  Collin Capano
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
This modules provides classes and functions for evaluating the log likelihood
for parameter estimation.
"""

import numpy
import logging

from scipy import (stats, special)

from pycbc import (conversions, filter, transforms, distributions)
from pycbc.waveform import (generator, NoWaveformError)
from pycbc.types import Array
from pycbc.io import FieldArray
from pycbc.workflow import ConfigParser

# Used to manage a likelihood instance across multiple cores or MPI
_global_instance = None


def _call_global_likelihood(*args, **kwds):
    """Private function for global likelihood (needed for parallelization)."""
    return _global_instance(*args, **kwds)  # pylint:disable=not-callable


class _NoPrior(object):
    """Dummy class to just return 0 if no prior is provided in a
    likelihood generator.
    """
    @staticmethod
    def apply_boundary_conditions(**params):
        return params

    def __call__(self, **params):
        return 0.


class BaseLikelihoodEvaluator(object):
    r"""Base container class for computing posteriors.

    The nomenclature used by this class and those that inherit from it is as
    follows: Given some model parameters :math:`\Theta` and some data
    :math:`d` with noise model :math:`n`, we define:

     * the **likelihood function**: :math:`p(d|\Theta)`

     * the **noise likelihood**: :math:`p(d|n)`

     * the **likelihood ratio**:
       :math:`\mathcal{L}(\Theta) = \frac{p(d|\Theta)}{p(d|n)}`

     * the **prior**: :math:`p(\Theta)`

     * the **posterior**: :math:`p(\Theta|d) \propto p(d|\Theta)p(\Theta)`

     * the **prior-weighted likelihood ratio**:
       :math:`\hat{\mathcal{L}}(\Theta) = \frac{p(d|\Theta)p(\Theta)}{p(d|n)}`

     * the **SNR**: :math:`\rho(\Theta) = \sqrt{2\log\mathcal{L}(\Theta)}`;
       for two detectors, this is approximately the same quantity as the
       coincident SNR used in the CBC search.

    .. note::

        Although the posterior probability is only proportional to
        :math:`p(d|\Theta)p(\Theta)`, here we refer to this quantity as the
        posterior. Also note that for a given noise model, the prior-weighted
        likelihood ratio is proportional to the posterior, and so the two can
        usually be swapped for each other.

    When performing parameter estimation we work with the log of these values
    since we are mostly concerned with their values around the maxima. If
    we have multiple detectors, each with data :math:`d_i`, then these values
    simply sum over the detectors. For example, the log likelihood ratio is:

    .. math::

        \log \mathcal{L}(\Theta) =
            \sum_i \left[\log p(\Theta|d_i) - \log p(n|d_i)\right]

    This class provides boiler-plate methods and attributes for evaluating the
    log likelihood ratio, log prior, and log likelihood. This class
    makes no assumption about the detectors' noise model :math:`n`. As such,
    the methods for computing these values raise ``NotImplementedErrors``. These
    functions need to be monkey patched, or other classes that inherit from
    this class need to define their own functions.

    Instances of this class can be called like a function. The default is for
    this class to call its ``logposterior`` function, but this can be changed
    with the ``set_callfunc`` method.

    Parameters
    ----------
    variable_args : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    static_args : dict, optional
        A dictionary of parameter names -> values to keep fixed.
    prior : callable, optional
        A callable class or function that computes the log of the prior. If
        None provided, will use ``_noprior``, which returns 0 for all parameter
        values.
    sampling_parameters : list, optional
        Replace one or more of the variable args with the given parameters
        for sampling.
    replace_parameters : list, optional
        The variable args to replace with sampling parameters. Must be the
        same length as ``sampling_parameters``.
    sampling_transforms : list, optional
        List of transforms to use to go between the variable args and the
        sampling parameters. Required if ``sampling_parameters`` is not None.

    Attributes
    ----------
    lognl : {None, float}
        The log of the noise likelihood summed over the number of detectors.
    return_meta : {True, bool}
        If True, ``prior``, ``logposterior``, and ``logplr`` will return the
        value of the prior, the loglikelihood ratio, and the log jacobian,
        along with the posterior/plr.

    Methods
    -------
    logjacobian :
        Returns the log of the jacobian needed to go from the parameter space
        of the variable args to the sampling args.
    prior :
        A function that returns the log of the prior.
    loglikelihood :
        A function that returns the log of the likelihood function.
    logposterior :
        A function that returns the log of the posterior.
    loglr :
        A function that returns the log of the likelihood ratio.
    logplr :
        A function that returns the log of the prior-weighted likelihood ratio.
    snr :
        A function that returns the square root of twice the log likelihood
        ratio. If the log likelihood ratio is < 0, will return 0.
    evaluate :
        Maps a list of values to their parameter names and calls whatever the
        call function is set to.
    set_callfunc :
        Set the function to use when the class is called as a function.
    """
    name = None

    def __init__(self, variable_args, static_args=None, prior=None,
                 sampling_parameters=None, replace_parameters=None,
                 sampling_transforms=None, return_meta=True):
        # store variable and static args
        if isinstance(variable_args, basestring):
            variable_args = (variable_args,)
        if not isinstance(variable_args, tuple):
            variable_args = tuple(variable_args)
        self._variable_args = variable_args
        if static_args is None:
            static_args = {}
        self._static_args = static_args
        # store prior
        if prior is None:
            self._prior = _NoPrior()
        else:
            # check that the variable args of the prior evaluator is the same
            # as the waveform generator
            if prior.variable_args != variable_args:
                raise ValueError("variable args of prior and waveform "
                                 "generator do not match")
            self._prior = prior
        # initialize the log nl to None
        self._lognl = None
        self.return_meta = return_meta
        # store sampling parameters and transforms
        if sampling_parameters is not None:
            if replace_parameters is None or \
                    len(replace_parameters) != len(sampling_parameters):
                raise ValueError("number of sampling parameters must be the "
                                 "same as the number of replace parameters")
            if sampling_transforms is None:
                raise ValueError("must provide sampling transforms for the "
                                 "sampling parameters")
            # pull out the replaced parameters
            self._sampling_args = [arg for arg in self._variable_args if
                                   arg not in replace_parameters]
            # add the samplign parameters
            self._sampling_args += sampling_parameters
            self._sampling_transforms = sampling_transforms
        else:
            self._sampling_args = self._variable_args
            self._sampling_transforms = None

    #
    # Methods for initiating from a config file.
    #
    @staticmethod
    def sampling_transforms_from_config(cp):
        """Gets sampling transforms specified in a config file.

        Sampling parameters and the parameters they replace are read from the
        ``sampling_parameters`` section, if it exists. Sampling transforms are
        read from the ``sampling_transforms`` section(s), using
        ``transforms.read_transforms_from_config``.

        If no ``sampling_parameters`` section exists in the config file, then
        no sampling sampling transforms will be returned, even if
        ``sampling_transforms`` sections do exist in the config file.

        Parameters
        ----------
        cp : WorkflowConfigParser
            Config file parser to read.

        Returns
        -------
        dict
            A dictionary of keyword arguments giving the
            ``sampling_parameters``, ``replace_parameters``, and
            ``sampling_transforms`` that were read from the config file. If
            no ``sampling_parameters`` section exists in the config file, these
            will all map to ``None``.
        """
        # get sampling transformations
        sampling_args = {}
        if cp.has_section('sampling_parameters'):
            sampling_parameters, replace_parameters = \
                read_sampling_args_from_config(cp)
            sampling_transforms = transforms.read_transforms_from_config(
                cp, 'sampling_transforms')
            logging.info("Sampling in {} in place of {}".format(
                ', '.join(sampling_parameters), ', '.join(replace_parameters)))
        else:
            sampling_parameters = None
            replace_parameters = None
            sampling_transforms = None
        sampling_args['sampling_parameters'] = sampling_parameters
        sampling_args['replace_parameters'] = replace_parameters
        sampling_args['sampling_transforms'] = sampling_transforms
        return sampling_args

    @staticmethod
    def extra_args_from_config(cp, section, skip_args=None, dtypes=None):
        """Gets any additional keyword in the given config file.

        Parameters
        ----------
        cp : WorkflowConfigParser
            Config file parser to read.
        section : str
            The name of the section to read.
        skip_args : list of str, optional
            Names of arguments to skip.
        dtypes : dict, optional
            A dictionary of arguments -> data types. If an argument is found
            in the dict, it will be cast to the given datatype. Otherwise, the
            argument's value will just be read from the config file (and thus
            be a string).

        Returns
        -------
        dict
            Dictionary of keyword arguments read from the config file.
        """
        kwargs = {}
        if dtypes is None:
            dtypes = {}
        if skip_args is None:
            skip_args = []
        read_args = [opt for opt in cp.options(section)
                     if opt not in skip_args]
        for opt in read_args:
            val = cp.get(section, opt)
            # try to cast the value if a datatype was specified for this opt
            try:
                val = dtypes[opt](val)
            except KeyError:
                pass
            kwargs[opt] = val
        return kwargs

    @classmethod
    def get_args_from_config(cls, cp, section="likelihood",
                             prior_section="prior"):
        """Gets arguments and keyword arguments from a config file.

        Parameters
        ----------
        cp : WorkflowConfigParser
            Config file parser to read.
        section : str, optional
            Section to read from. Default is 'likelihood'.
        prior_section : str, optional
            Section(s) to read prior(s) from. Default is 'prior'.

        Returns
        -------
        dict
            A dctionary of keyword arguments giving the ``variable_args``,
            ``static_args``, and ``prior`` class.
        """
        # check that the name exists and matches
        name = cp.get(section, 'name')
        if name != cls.name:
            raise ValueError("section's {} name does not match mine {}".format(
                             name, cls.name))

        variable_args, static_args, constraints = \
            distributions.read_args_from_config(cp)
        args = {'variable_args': variable_args, 'static_args': static_args}

        # get prior distribution for each variable parameter
        logging.info("Setting up priors for each parameter")
        dists = distributions.read_distributions_from_config(cp, prior_section)
        prior_eval = distributions.JointDistribution(
            variable_args, *dists, **{"constraints": constraints})
        args['prior'] = prior_eval
        # get sampling transforms and any other keyword arguments provided
        args.update(cls.sampling_transforms_from_config(cp))
        args.update(cls.extra_args_from_config(cp, section,
                                               skip_args=['name']))
        return args

    @classmethod
    def from_config(cls, cp, section="likelihood", prior_section="prior",
                    **kwargs):
        """Initializes an instance of this class from the given config file.

        Parameters
        ----------
        cp : WorkflowConfigParser
            Config file parser to read.
        section : str, optional
            The section to read the arguments to the likelihood class from.
            Default is 'likelihood'.
        prior_section : str, optional
            The section to read the prior arguments from. Default is 'prior'.
        \**kwargs :
            All additional keyword arguments are passed to the class. Any
            provided keyword will over ride what is in the config file.
        """
        args = cls.get_args_from_config(cp, section=section,
                                        prior_section=prior_section)
        args.update(kwargs)
        return cls(**args)

    #
    # Properties and methods
    #
    @property
    def variable_args(self):
        """Returns the variable arguments."""
        return self._variable_args

    @property
    def static_args(self):
        """Returns the static arguments."""
        return self._static_args

    @property
    def sampling_args(self):
        """Returns the sampling arguments."""
        return self._sampling_args

    @property
    def sampling_transforms(self):
        """Returns the sampling transforms."""
        return self._sampling_transforms

    def apply_sampling_transforms(self, samples, inverse=False):
        """Applies the sampling transforms to the given samples.

        If ``sampling_transforms`` is None, just returns the samples.

        Parameters
        ----------
        samples : dict or FieldArray
            The samples to apply the transforms to.
        inverse : bool, optional
            Whether to apply the inverse transforms (i.e., go from the sampling
            args to the variable args). Default is False.

        Returns
        -------
        dict or FieldArray
            The transformed samples, along with the original samples.
        """
        if self._sampling_transforms is None:
            return samples
        return transforms.apply_transforms(samples, self._sampling_transforms,
                                           inverse=inverse)

    @property
    def lognl(self):
        """Returns the log of the noise likelihood."""
        return self._lognl

    def set_lognl(self, lognl):
        """Set the value of the log noise likelihood."""
        self._lognl = lognl

    def logjacobian(self, **params):
        r"""Returns the log of the jacobian needed to transform pdfs in the
        ``variable_args`` parameter space to the ``sampling_args`` parameter
        space.

        Let :math:`\mathbf{x}` be the set of variable parameters,
        :math:`\mathbf{y} = f(\mathbf{x})` the set of sampling parameters, and
        :math:`p_x(\mathbf{x})` a probability density function defined over
        :math:`\mathbf{x}`.
        The corresponding pdf in :math:`\mathbf{y}` is then:

        .. math::

            p_y(\mathbf{y}) =
                p_x(\mathbf{x})\left|\mathrm{det}\,\mathbf{J}_{ij}\right|,

        where :math:`\mathbf{J}_{ij}` is the Jacobian of the inverse transform
        :math:`\mathbf{x} = g(\mathbf{y})`. This has elements:

        .. math::

            \mathbf{J}_{ij} = \frac{\partial g_i}{\partial{y_j}}

        This function returns
        :math:`\log \left|\mathrm{det}\,\mathbf{J}_{ij}\right|`.


        Parameters
        ----------
        \**params :
            The keyword arguments should specify values for all of the variable
            args and all of the sampling args.

        Returns
        -------
        float :
            The value of the jacobian.
        """
        if self._sampling_transforms is None:
            return 0.
        else:
            return numpy.log(abs(transforms.compute_jacobian(
                params, self._sampling_transforms, inverse=True)))

    def prior(self, **params):
        """This function should return the prior of the given params.
        """
        logj = self.logjacobian(**params)
        logp = self._prior(**params) + logj
        if numpy.isnan(logp):
            logp = -numpy.inf
        return self._formatreturn(logp, prior=logp, logjacobian=logj)

    def prior_rvs(self, size=1, prior=None):
        """Returns random variates drawn from the prior.

        If the ``sampling_args`` are different from the ``variable_args``, the
        variates are transformed to the `sampling_args` parameter space before
        being returned.

        Parameters
        ----------
        size : int, optional
            Number of random values to return for each parameter. Default is 1.
        prior : JointDistribution, optional
            Use the given prior to draw values rather than the saved prior.

        Returns
        -------
        FieldArray
            A field array of the random values.
        """
        # draw values from the prior
        if prior is None:
            prior = self._prior
        p0 = prior.rvs(size=size)
        # transform if necessary
        if self._sampling_transforms is not None:
            ptrans = self.apply_sampling_transforms(p0)
            # pull out the sampling args
            p0 = FieldArray.from_arrays([ptrans[arg]
                                         for arg in self._sampling_args],
                                        names=self._sampling_args)
        return p0

    def loglikelihood(self, **params):
        """Returns the natural log of the likelihood function.
        """
        raise NotImplementedError("Likelihood function not set.")

    def loglr(self, **params):
        """Returns the natural log of the likelihood ratio.
        """
        return self.loglikelihood(**params) - self.lognl

    # the names and order of data returned by _formatreturn when
    # return_metadata is True
    metadata_fields = ["prior", "loglr", "logjacobian"]

    def _formatreturn(self, val, prior=None, loglr=None, logjacobian=0.):
        """Adds the prior to the return value if return_meta is True.
        Otherwise, just returns the value.

        Parameters
        ----------
        val : float
            The value to return.
        prior : float, optional
            The value of the prior.
        loglr : float, optional
            The value of the log likelihood-ratio.
        logjacobian : float, optional
            The value of the log jacobian used to go from the variable args
            to the sampling args.

        Returns
        -------
        val : float
            The given value to return.
        *If return_meta is True:*
        metadata : (prior, loglr, logjacobian)
            A tuple of the prior, log likelihood ratio, and logjacobian.
        """
        if self.return_meta:
            return val, (prior, loglr, logjacobian)
        else:
            return val

    def logplr(self, **params):
        """Returns the log of the prior-weighted likelihood ratio.
        """
        if self.return_meta:
            logp, (_, _, logj) = self.prior(**params)
        else:
            logp = self.prior(**params)
            logj = None
        # if the prior returns -inf, just return
        if logp == -numpy.inf:
            return self._formatreturn(logp, prior=logp, logjacobian=logj)
        llr = self.loglr(**params)
        return self._formatreturn(llr + logp, prior=logp, loglr=llr,
                                  logjacobian=logj)

    def logposterior(self, **params):
        """Returns the log of the posterior of the given params.
        """
        if self.return_meta:
            logp, (_, _, logj) = self.prior(**params)
        else:
            logp = self.prior(**params)
            logj = None
        # if the prior returns -inf, just return
        if logp == -numpy.inf:
            return self._formatreturn(logp, prior=logp, logjacobian=logj)
        ll = self.loglikelihood(**params)
        return self._formatreturn(ll + logp, prior=logp, loglr=ll-self._lognl,
                                  logjacobian=logj)

    def snr(self, **params):
        """Returns the "SNR" of the given params. This will return
        imaginary values if the log likelihood ratio is < 0.
        """
        return conversions.snr_from_loglr(self.loglr(**params))

    _callfunc = logposterior

    @classmethod
    def set_callfunc(cls, funcname):
        """Sets the function used when the class is called as a function.

        Parameters
        ----------
        funcname : str
            The name of the function to use; must be the name of an instance
            method.
        """
        cls._callfunc = getattr(cls, funcname)

    def _transform_params(self, params):
        """Applies all transforms to the given list of param values.

        Parameters
        ----------
        params : list
            A list of values. These are assumed to be in the same order as
            ``variable_args``.

        Returns
        -------
        dict
            A dictionary of the transformed parameters.
        """
        params = dict(zip(self._sampling_args, params))
        # apply inverse transforms to go from sampling parameters to
        # variable args
        params = self.apply_sampling_transforms(params, inverse=True)
        # apply boundary conditions
        params = self._prior.apply_boundary_conditions(**params)
        return params

    def evaluate(self, params, callfunc=None):
        """Evaluates the call function at the given list of parameter values.

        Parameters
        ----------
        params : list
            A list of values. These are assumed to be in the same order as
            variable args.
        callfunc : str, optional
            The name of the function to call. If None, will use
            ``self._callfunc``. Default is None.

        Returns
        -------
        float or tuple :
            If ``return_meta`` is False, the output of the call function. If
            ``return_meta`` is True, a tuple of the output of the call function
            and the meta data.
        """
        params = self._transform_params(params)
        # apply any boundary conditions to the parameters before
        # generating/evaluating
        if callfunc is not None:
            f = getattr(self, callfunc)
        else:
            f = self._callfunc
        return f(**params)

    __call__ = evaluate


#
# =============================================================================
#
#                              Test distributions
#
# =============================================================================
#
class TestNormal(BaseLikelihoodEvaluator):
    r"""The test distribution is an multi-variate normal distribution.

    The number of dimensions is set by the number of ``variable_args`` that are
    passed. For details on the distribution used, see
    ``scipy.stats.multivariate_normal``.

    Parameters
    ----------
    variable_args : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    mean : array-like, optional
        The mean values of the parameters. If None provide, will use 0 for all
        parameters.
    cov : array-like, optional
        The covariance matrix of the parameters. If None provided, will use
        unit variance for all parameters, with cross-terms set to 0.
    **kwargs :
        All other keyword arguments are passed to ``BaseLikelihoodEvaluator``.

    """
    name = "test_normal"

    def __init__(self, variable_args, mean=None, cov=None, **kwargs):
        # set up base likelihood parameters
        super(TestNormal, self).__init__(variable_args, **kwargs)
        # set the lognl to 0 since there is no data
        self.set_lognl(0.)
        # store the pdf
        if mean is None:
            mean = [0.]*len(variable_args)
        if cov is None:
            cov = [1.]*len(variable_args)
        self._dist = stats.multivariate_normal(mean=mean, cov=cov)
        # check that the dimension is correct
        if self._dist.dim != len(variable_args):
            raise ValueError("dimension mis-match between variable_args and "
                             "mean and/or cov")

    def loglikelihood(self, **params):
        """Returns the log pdf of the multivariate normal.
        """
        return self._dist.logpdf([params[p] for p in self.variable_args])


class TestEggbox(BaseLikelihoodEvaluator):
    r"""The test distribution is an 'eggbox' function:

    .. math::

        \log \mathcal{L}(\Theta) = \left[
            2+\prod_{i=1}^{n}\cos\left(\frac{\theta_{i}}{2}\right)\right]^{5}

    The number of dimensions is set by the number of ``variable_args`` that are
    passed.

    Parameters
    ----------
    variable_args : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    **kwargs :
        All other keyword arguments are passed to ``BaseLikelihoodEvaluator``.

    """
    name = "test_eggbox"

    def __init__(self, variable_args, **kwargs):
        # set up base likelihood parameters
        super(TestEggbox, self).__init__(variable_args, **kwargs)

        # set the lognl to 0 since there is no data
        self.set_lognl(0.)

    def loglikelihood(self, **params):
        """Returns the log pdf of the eggbox function.
        """
        return (2 + numpy.prod(numpy.cos([
            params[p]/2. for p in self.variable_args]))) ** 5


class TestRosenbrock(BaseLikelihoodEvaluator):
    r"""The test distribution is the Rosenbrock function:

    .. math::

        \log \mathcal{L}(\Theta) = -\sum_{i=1}^{n-1}[
            (1-\theta_{i})^{2}+100(\theta_{i+1} - \theta_{i}^{2})^{2}]

    The number of dimensions is set by the number of ``variable_args`` that are
    passed.

    Parameters
    ----------
    variable_args : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    **kwargs :
        All other keyword arguments are passed to ``BaseLikelihoodEvaluator``.

    """
    name = "test_rosenbrock"

    def __init__(self, variable_args, **kwargs):
        # set up base likelihood parameters
        super(TestRosenbrock, self).__init__(variable_args, **kwargs)

        # set the lognl to 0 since there is no data
        self.set_lognl(0.)

    def loglikelihood(self, **params):
        """Returns the log pdf of the Rosenbrock function.
        """
        l = 0
        p = [params[p] for p in self.variable_args]
        for i in range(len(p) - 1):
            l -= ((1 - p[i])**2 + 100 * (p[i+1] - p[i]**2)**2)
        return l


class TestVolcano(BaseLikelihoodEvaluator):
    r"""The test distribution is a two-dimensional 'volcano' function:

    .. math::
        \Theta =
            \sqrt{\theta_{1}^{2} + \theta_{2}^{2}} \log \mathcal{L}(\Theta) =
            25\left(e^{\frac{-\Theta}{35}} +
                    \frac{1}{2\sqrt{2\pi}} e^{-\frac{(\Theta-5)^{2}}{8}}\right)

    Parameters
    ----------
    variable_args : (tuple of) string(s)
        A tuple of parameter names that will be varied. Must have length 2.
    **kwargs :
        All other keyword arguments are passed to ``BaseLikelihoodEvaluator``.

    """
    name = "test_volcano"

    def __init__(self, variable_args, **kwargs):
        # set up base likelihood parameters
        super(TestVolcano, self).__init__(variable_args, **kwargs)

        # make sure there are exactly two variable args
        if len(self.variable_args) != 2:
            raise ValueError("TestVolcano distribution requires exactly "
                             "two variable args")

        # set the lognl to 0 since there is no data
        self.set_lognl(0.)

    def loglikelihood(self, **params):
        """Returns the log pdf of the 2D volcano function.
        """
        p = [params[p] for p in self.variable_args]
        r = numpy.sqrt(p[0]**2 + p[1]**2)
        mu, sigma = 5.0, 2.0
        return 25 * (
            numpy.exp(-r/35) + 1 / (sigma * numpy.sqrt(2 * numpy.pi)) *
            numpy.exp(-0.5 * ((r - mu) / sigma) ** 2))


#
# =============================================================================
#
#                              Data-based likelihoods
#
# =============================================================================
#

class DataBasedLikelihoodEvaluator(BaseLikelihoodEvaluator):
    r"""A likelihood evaulator that requires data and a waveform generator.

    Like ``BaseLikelihoodEvaluator``, this class only provides boiler-plate
    attributes and methods for evaluating likelihoods. Classes that make use
    of data and a waveform generator should inherit from this.

    Parameters
    ----------
    variable_args : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    data : dict
        A dictionary of data, in which the keys are the detector names and the
        values are the data.
    waveform_generator : generator class
        A generator class that creates waveforms.
    waveform_transforms : list, optional
        List of transforms to use to go from the variable args to parameters
        understood by the waveform generator.

    \**kwargs :
        All other keyword arguments are passed to ``BaseLikelihoodEvaluator``.

    Attributes
    ----------
    waveform_generator : dict
        The waveform generator that the class was initialized with.
    data : dict
        The data that the class was initialized with.

    For additional attributes and methods, see ``BaseLikelihoodEvaluator``.
    """
    name = None

    def __init__(self, variable_args, data, waveform_generator,
                 waveform_transforms=None, **kwargs):
        # we'll store a copy of the data
        self._data = {ifo: d.copy() for (ifo, d) in data.items()}
        self._waveform_generator = waveform_generator
        self._waveform_transforms = waveform_transforms
        super(DataBasedLikelihoodEvaluator, self).__init__(
            variable_args, **kwargs)

    @property
    def waveform_generator(self):
        """Returns the waveform generator that was set."""
        return self._waveform_generator

    @property
    def data(self):
        """Returns the data that was set."""
        return self._data

    def _transform_params(self, params):
        """Adds waveform transforms to parent's ``_transform_params``."""
        params = super(DataBasedLikelihoodEvaluator, self)._transform_params(
            params)
        # apply waveform transforms
        if self._waveform_transforms is not None:
            params = transforms.apply_transforms(params,
                                                 self._waveform_transforms,
                                                 inverse=False)
        return params

    @classmethod
    def get_args_from_config(cls, cp, section="likelihood",
                             prior_section="prior"):
        # adds waveform transforms to the arguments
        args = super(DataBasedLikelihoodEvaluator, cls).get_args_from_config(
            cp, section=section, prior_section=prior_section)
        if any(cp.get_subsections('waveform_transforms')):
            logging.info("Loading waveform transforms")
            args['waveform_transforms'] = \
                transforms.read_transforms_from_config(cp,
                                                       'waveform_transforms')
        return args

    @classmethod
    def from_config(cls, cp, data, delta_f=None, delta_t=None,
                    gates=None, recalibration=None,
                    section="likelihood", prior_section="prior",
                    **kwargs):
        """Initializes an instance of this class from the given config file.

        Parameters
        ----------
        cp : WorkflowConfigParser
            Config file parser to read.
        data : dict
            A dictionary of data, in which the keys are the detector names and
            the values are the data. This is not retrieved from the config
            file, and so must be provided.
        delta_f : float
            The frequency spacing of the data; needed for waveform generation.
        delta_t : float
            The time spacing of the data; needed for time-domain waveform
            generators.
        recalibration : dict of pycbc.calibration.Recalibrate, optional
            Dictionary of detectors -> recalibration class instances for
            recalibrating data.
        gates : dict of tuples, optional
            Dictionary of detectors -> tuples of specifying gate times. The
            sort of thing returned by `pycbc.gate.gates_from_cli`.
        section : str, optional
            The section to read the arguments to the likelihood class from.
            Default is 'likelihood'.
        prior_section : str, optional
            The section to read the prior arguments from. Default is 'prior'.
        \**kwargs :
            All additional keyword arguments are passed to the class. Any
            provided keyword will over ride what is in the config file.
        """
        if data is None:
            raise ValueError("must provide data")

        args = cls.get_args_from_config(cp, section=section,
                                        prior_section=prior_section)
        args['data'] = data
        args.update(kwargs)

        variable_args = args['variable_args']
        try:
            static_args = args['static_args']
        except KeyError:
            static_args = {}

        # set up waveform generator
        try:
            approximant = static_args['approximant']
        except KeyError:
            raise ValueError("no approximant provided in the static args")
        generator_function = generator.select_waveform_generator(approximant)
        waveform_generator = generator.FDomainDetFrameGenerator(
            generator_function, epoch=data.values()[0].start_time,
            variable_args=variable_args, detectors=data.keys(),
            delta_f=delta_f, delta_t=delta_t,
            recalib=recalibration, gates=gates,
            **static_args)
        args['waveform_generator'] = waveform_generator

        return cls(**args)


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


likelihood_evaluators = {cls.name: cls for cls in (
    TestEggbox,
    TestNormal,
    TestRosenbrock,
    TestVolcano,
    GaussianLikelihood,
    MarginalizedPhaseGaussianLikelihood,
)}


def read_sampling_args_from_config(cp, section_group=None,
                                   section='sampling_parameters'):
    """Reads sampling parameters from the given config file.

    Parameters are read from the `[({section_group}_){section}]` section.
    The options should list the variable args to transform; the parameters they
    point to should list the parameters they are to be transformed to for
    sampling. If a multiple parameters are transformed together, they should
    be comma separated. Example:

    .. code-block:: ini

        [sampling_parameters]
        mass1, mass2 = mchirp, logitq
        spin1_a = logitspin1_a

    Note that only the final sampling parameters should be listed, even if
    multiple intermediate transforms are needed. (In the above example, a
    transform is needed to go from mass1, mass2 to mchirp, q, then another one
    needed to go from q to logitq.) These transforms should be specified
    in separate sections; see ``transforms.read_transforms_from_config`` for
    details.

    Parameters
    ----------
    cp : WorkflowConfigParser
        An open config parser to read from.
    section_group : str, optional
        Append `{section_group}_` to the section name. Default is None.
    section : str, optional
        The name of the section. Default is 'sampling_parameters'.

    Returns
    -------
    sampling_params : list
        The list of sampling parameters to use instead.
    replaced_params : list
        The list of variable args to replace in the sampler.
    """
    if section_group is not None:
        section_prefix = '{}_'.format(section_group)
    else:
        section_prefix = ''
    section = section_prefix + section
    replaced_params = set()
    sampling_params = set()
    for args in cp.options(section):
        map_args = cp.get(section, args)
        sampling_params.update(set(map(str.strip, map_args.split(','))))
        replaced_params.update(set(map(str.strip, args.split(','))))
    return list(sampling_params), list(replaced_params)


def read_from_config(cp, section="likelihood", **kwargs):
    """Initializes a LikelihoodEvaluator from the given config file.
    """
    # use the name to get the distribution
    name = cp.get(section, "name")
    return likelihood_evaluators[name].from_config(
        cp, section=section, **kwargs)


__all__ = ['BaseLikelihoodEvaluator',
           'TestNormal', 'TestEggbox', 'TestVolcano', 'TestRosenbrock',
           'GaussianLikelihood', 'MarginalizedPhaseGaussianLikelihood',
           'likelihood_evaluators']
