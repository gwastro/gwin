###################################
GWin on the command line (``gwin``)
###################################

===================
Introduction
===================

This page gives details on how to use the various parameter estimation
executables and modules available in GWin. The ``gwin`` subpackage
contains classes and functions for evaluating probability distributions,
likelihoods, and running Bayesian samplers.

==================================================
Sampling the parameter space (``gwin``)
==================================================

--------
Overview
--------

The executable ``gwin`` is designed to sample the parameter space
and save the samples in an HDF file. A high-level description of the
``gwin`` algorithm is

#. Read priors from a configuration file.

#. Setup the model to use. If the model uses data, then:

    #. Read gravitational-wave strain from a gravitational-wave model or use recolored fake strain.

    #. Estimate a PSD.

#. Run a sampler to estimate the posterior distribution of the model.

---------------------------------------------------
Options for samplers, models, and priors
---------------------------------------------------

For a full listing of all options run ``gwin --help``. In this subsection we reference documentation for Python classes that contain more information about choices for samplers, likelihood models, and priors.

The user specifies the sampler on the command line with the ``--sampler`` option.
A complete list of samplers is given in ``gwin --help``.
These samplers are described in :py:class:`gwin.sampler.KombineSampler`, :py:class:`gwin.sampler.EmceeEnsembleSampler`, and :py:class:`gwin.sampler.EmceePTSampler`.
In addition to ``--sampler`` the user will need to specify the number of walkers to use ``--nwalkers``, and for parallel-tempered samplers the number of temperatures ``--ntemps``. You also need to either specify the number of iterations to run for using ``--niterations`` **or** the number of independent samples to collect using ``--n-independent-samples``. For the latter, a burn-in function must be specified using ``--burn-in-function``. In this case, the program will run until the sampler has burned in, at which point the number of independent samples equals the number of walkers. If the number of independent samples desired is greater than the number of walkers, the program will continue to run until it has collected the specified number of independent samples (to do this, an autocorrelation length is computed at each checkpoint to determine how many iterations need to be skipped to obtain independent samples).

The user specifies a configuration file that defines the priors with the ``--config-files`` option.
The syntax of the configuration file is described in the following subsection.

-------------------------
Configuration file syntax
-------------------------

Configuration files follow the ``ConfigParser`` syntax.
There are three required sections.

One is a ``[model]`` section that contains the name of the model class
to use for evaluating the posterior. An example::

    [model]
    name = gaussian_noise

In this case, the :py:class:`gwin.models.GaussianNoise` would be used.
Examples of using this model on a BBH injection and on GW150914 are
given below. Any name that starts with ``test_`` is an analytic test
distribution that requires no data or waveform generation; see the section
below on running on an analytic distribution for more details.

The other two required sections are ``[variable_params]``, and ``[static_params]``.
The ``[variable_params]`` section contains a list of parameters in which the prior
will be defined, and that will varied to obtain a posterior distribution. The ``[static_params]`` 
section contains a list of parameters that are held fixed through out the run.

Each parameter in ``[variable_params]`` must have a subsection in ``[prior]``.
To create a subsection use the ``-`` char, e.g. for one of the mass parameters do ``[prior-mass1]``.

Each prior subsection must have a ``name`` option that identifies what prior to use.
These distributions are described in :py:mod:`pycbc.distributions`.

.. include:: /_includes/distributions-table.rst

One or more of the ``variable_params`` may be transformed to a different parameter space for purposes of sampling. This is done by specifying a ``[sampling_params]`` section. This section specifies which ``variable_params`` to replace with which parameters for sampling. This must be followed by one or more ``[sampling_transforms-{sampling_params}]`` sections that provide the transform class to use. For example, the following would cause the sampler to sample in chirp mass (``mchirp``) and mass ratio (``q``) instead of ``mass1`` and ``mass2``::

    [sampling_params]
    mass1, mass2: mchirp, q

    [sampling_transforms-mchirp+q]
    name = mass1_mass2_to_mchirp_q

For a list of all possible transforms see :py:mod:`pycbc.transforms`.

There can be any number of ``variable_params`` with any name. No parameter name is special (with the exception of parameters that start with ``calib_``; see below). However, in order to generate waveforms, certain parameters must be provided for waveform generation. If you would like to specify a ``variable_arg`` that is not one of these parameters, then you must provide a ``[waveforms_transforms-{param}]`` section that provides a transform from the arbitrary ``variable_params`` to the needed waveform parameter(s) ``{param}``. For example, in the following we provide a prior on ``chirp_distance``. Since ``distance``, not ``chirp_distance``, is recognized by the CBC waveforms module, we provide a transform to go from ``chirp_distance`` to ``distance``::

    [variable_params]
    chirp_distance =

    [prior-chirp_distance]
    name = uniform
    min-chirp_distance = 1
    max-chirp_distance = 200

    [waveform_transforms-distance]
    name = chirp_distance_to_distance

Any class in the transforms module may be used. A useful transform for these purposes is the :py:class:`pycbc.transforms.CustomTransform`, which allows for arbitrary transforms using any function in the :py:mod:`pycbc.conversions`, :py:mod:`pycbc.coordinates`, or :py:mod:`pycbc.cosmology` modules, along with numpy math functions. For example, the following would use the I-Love-Q relationship :py:meth:`pycbc.conversions.dquadmon_from_lambda` to relate the quadrupole moment of a neutron star ``dquad_mon1`` to its tidal deformation ``lambda1``::

    [variable_params]
    lambda1 =

    [waveform_transforms-dquad_mon1]
    name = custom
    inputs = lambda1
    dquad_mon1 = dquadmon_from_lambda(lambda1)

The following table lists all parameters understood by the CBC waveform generator:

.. include:: /_includes/waveform-parameters.rst

Some common transforms are pre-defined in the code. These are: the mass parameters ``mass1`` and ``mass2`` can be substituted with ``mchirp`` and ``eta`` or ``mchirp`` and ``q``.
The component spin parameters ``spin1x``, ``spin1y``, and ``spin1z`` can be substituted for polar coordinates ``spin1_a``, ``spin1_azimuthal``, and ``spin1_polar`` (ditto for ``spin2``).

If any calibration parameters are used (prefix ``calib_``), a ``[calibration]`` section must be included. This section must have a ``name`` option that identifies what calibration model to use. The models are described in :py:mod:`pycbc.calibration`. The ``[calibration]`` section must also include reference values ``fc0``, ``fs0``, and ``qinv0``, as well as paths to ASCII transfer function files for the test mass actuation, penultimate mass actuation, sensing function, and digital filter for each IFO being used in the analysis. E.g. for an analysis using H1 only, the required options would be ``h1-fc0``, ``h1-fs0``, ``h1-qinv0``, ``h1-transfer-function-a-tst``, ``h1-transfer-function-a-pu``, ``h1-transfer-function-c``, ``h1-transfer-function-d``.

Simple examples are given in the subsections below.

-----------------------------------
Running on an analytic distribution
-----------------------------------

Several analytic distributions are available to run tests on. These can be run quickly on a laptop to check that a sampler is working properly.

This example demonstrates how to sample a 2D normal distribution with the ``emcee`` sampler. First, create the following configuration file (named ``normal2d.ini``)::

    [model]
    name = test_normal

    [variable_params]
    x =
    y =

    [prior-x]
    name = uniform
    min-x = -10
    max-x = 10

    [prior-y]
    name = uniform
    min-y = -10
    max-y = 10

Then run::

    gwin --verbose \
        --config-files normal2d.ini \
        --output-file normal2d.hdf \
        --sampler emcee \
        --niterations 50 \
        --nwalkers 5000 \
        --nprocesses 2

This will run the ``emcee`` sampler on the 2D analytic normal distribution with 5000 walkers for 100 iterations.

To plot the posterior distribution after the last iteration, run::

    gwin_plot_posterior --verbose \
            --input-file normal2d.hdf \
            --output-file posterior-normal2d.png \
            --plot-scatter \
            --plot-contours \
            --plot-marginal \
            --z-arg 'loglikelihood:$\log p(h|\vartheta)$' \
            --iteration -1

This will plot each walker's position as a single point colored by the log likelihood ratio at that point, with the 50th and 90th percentile contours drawn. See below for more information about using ``gwin_plot_posterior``.

To make a movie showing how the walkers evolved, run::

    gwin_plot_movie --verbose \
        --input-file normal2d.hdf \
        --output-prefix frames-normal2d \
        --movie-file normal2d_mcmc_evolution.mp4 \
        --cleanup \
        --plot-scatter \
        --plot-contours \
        --plot-marginal \
        --z-arg 'loglikelihood:$\log p(h|\vartheta)$' \
        --frame-step 1

**Note:** you need ``ffmpeg`` installed for the mp4 to be created. See below for more information on using ``gwin_plot_movie``.

The number of dimensions of the distribution is set by the number of ``variable_params`` in the configuration file. The names of the ``variable_params`` do not matter, just that the prior sections use the same names (in this example ``x`` and ``y`` were used, but ``foo`` and ``bar`` would be equally valid). A higher (or lower) dimensional distribution can be tested by simply adding more (or less) ``variable_params``.

Which analytic distribution is used is set by the ``[model]`` section in
the configuration file. By setting to ``test_normal`` we used
:py:class:`gwin.models.TestNormal`. The other analytic distributions available
are: :py:class:`gwin.models.TestEggbox`,
:py:class:`gwin.models.TestRosenbrock`, and
:py:class:`gwin.models.TestVolcano`. As with ``test_normal``, the
dimensionality of these test distributions is set by the number of
``variable_params`` in the configuration file. The ``test_volcano`` distribution
must be two dimensional, but all of the other distributions can have any number
of dimensions. The configuration file syntax for the other test distributions
is the same as in this example (aside from the name used in the
model section). Indeed, with this configuration file one only
needs to change the ``name`` argument in ``[model]`` argument to try (2D versions of)
the other distributions.

------------------------------
BBH software injection example
------------------------------

This example recovers the parameters of a precessing binary black-hole (BBH).

An example configuration file (named ``gwin.ini``) is::

    [model]
    name = gaussian_noise

    [variable_params]
    ; waveform parameters that will vary in MCMC
    tc =
    mass1 =
    mass2 =
    spin1_a =
    spin1_azimuthal =
    spin1_polar =
    spin2_a =
    spin2_azimuthal =
    spin2_polar =
    distance =
    coa_phase =
    inclination =
    polarization =
    ra =
    dec =

    [static_params]
    ; waveform parameters that will not change in MCMC
    approximant = IMRPhenomPv2
    f_lower = 18
    f_ref = 20

    [prior-tc]
    ; coalescence time prior
    name = uniform
    min-tc = 1126259462.32
    max-tc = 1126259462.52

    [prior-mass1]
    name = uniform
    min-mass1 = 10.
    max-mass1 = 80.

    [prior-mass2]
    name = uniform
    min-mass2 = 10.
    max-mass2 = 80.

    [prior-spin1_a]
    name = uniform
    min-spin1_a = 0.0
    max-spin1_a = 0.99

    [prior-spin1_polar+spin1_azimuthal]
    name = uniform_solidangle
    polar-angle = spin1_polar
    azimuthal-angle = spin1_azimuthal

    [prior-spin2_a]
    name = uniform
    min-spin2_a = 0.0
    max-spin2_a = 0.99

    [prior-spin2_polar+spin2_azimuthal]
    name = uniform_solidangle
    polar-angle = spin2_polar
    azimuthal-angle = spin2_azimuthal

    [prior-distance]
    ; following gives a uniform volume prior
    name = uniform_radius
    min-distance = 10
    max-distance = 1000

    [prior-coa_phase]
    ; coalescence phase prior
    name = uniform_angle

    [prior-inclination]
    ; inclination prior
    name = sin_angle

    [prior-ra+dec]
    ; sky position prior
    name = uniform_sky

    [prior-polarization]
    ; polarization prior
    name = uniform_angle

    ;
    ;   Sampling transforms
    ;
    [sampling_params]
    ; parameters on the left will be sampled in
    ; parametes on the right
    mass1, mass2 : mchirp, q

    [sampling_transforms-mchirp+q]
    ; inputs mass1, mass2
    ; outputs mchirp, q
    name = mass1_mass2_to_mchirp_q

To generate an injection we will use ``pycbc_create_injections``. This program
takes a configuration file to set up the distributions from which to draw the
injection parameters (run ``pycbc_create_injections --help`` for details). The
syntax of that file is the same as the file used for ``gwin``, so we could, if
we wished, simply give the above configuartion file to
``pycbc_create_injections``. However, to ensure we obtain a specific set of
parameters, we will create another configuration file that fixes the injection
parameters to specific values. Create the following file, calling it
``injection.ini``::

    [variable_params]

    [static_params]
    tc = 1126259462.420
    mass1 = 37
    mass2 = 32
    ra = 2.2
    dec = -1.25
    inclincation = 2.5
    coa_phase = 1.5
    polarization = 1.75
    distance = 100
    f_ref = 20
    f_lower = 18
    approximant = IMRPhenomPv2
    taper = start

This will create a non-spinning injection (if no spin parameters are provided,
the injection will be non-spinning by default) using ``IMRPhenomPv2``. (Note
that we still need a ``[variable_params]`` section even though we are fixing all
parameters.) Now run::

    pycbc_create_injections --verbose \
        --config-files injection.ini \
        --ninjections 1 \
        --output-file injection.hdf \
        --variable-params-section variable_params \
        --static-args-section static_params \
        --dist-section prior

This will create a file called ``injection.hdf`` which contains the injection's
parameters. This file can be passed to ``gwin`` with the ``--injection-file``
option. To run ``gwin`` on this injection in simulated fake data, set the
following bash variables::

    # injection parameters
    TRIGGER_TIME_INT=1126259462

    # sampler parameters
    CONFIG_PATH=gwin.ini
    OUTPUT_PATH=gwin.hdf
    SEGLEN=8
    PSD_INVERSE_LENGTH=4
    IFOS="H1 L1"
    STRAIN="H1:aLIGOZeroDetHighPower L1:aLIGOZeroDetHighPower"
    SAMPLE_RATE=2048
    F_MIN=20
    N_UPDATE=500
    N_WALKERS=5000
    N_SAMPLES=5000
    N_CHECKPOINT=1000
    PROCESSING_SCHEME=cpu

    # the following sets the number of cores to use; adjust as needed to
    # your computer's capabilities
    NPROCS=12

    # start and end time of data to read in
    GPS_START_TIME=$((${TRIGGER_TIME_INT} - ${SEGLEN}))
    GPS_END_TIME=$((${TRIGGER_TIME_INT} + ${SEGLEN}))

Now run::

    # run sampler
    # specifies the number of threads for OpenMP
    # Running with OMP_NUM_THREADS=1 stops lalsimulation
    # from spawning multiple jobs that would otherwise be used
    # by gwin and cause a reduced runtime.
    OMP_NUM_THREADS=1 \
    gwin --verbose \
        --seed 12 \
        --instruments ${IFOS} \
        --gps-start-time ${GPS_START_TIME} \
        --gps-end-time ${GPS_END_TIME} \
        --psd-model ${STRAIN} \
        --psd-inverse-length ${PSD_INVERSE_LENGTH} \
        --fake-strain ${STRAIN} \
        --fake-strain-seed 44 \
        --strain-high-pass ${F_MIN} \
        --sample-rate ${SAMPLE_RATE} \
        --low-frequency-cutoff ${F_MIN} \
        --channel-name H1:FOOBAR L1:FOOBAR \
        --injection-file injection.hdf \
        --config-file ${CONFIG_PATH} \
        --output-file ${OUTPUT_PATH} \
        --processing-scheme ${PROCESSING_SCHEME} \
        --sampler kombine \
        --burn-in-function max_posterior \
        --update-interval ${N_UPDATE} \
        --nwalkers ${N_WALKERS} \
        --n-independent-samples ${N_SAMPLES} \
        --checkpoint-interval ${N_CHECKPOINT} \
        --nprocesses ${NPROCS} \
        --save-strain \
        --save-psd \
        --save-stilde \
        --force

While the code is running it will write results to ``gwin.hdf.checkpoint``
after every checkpoint interval (you'll see ``Writing results to file`` when a
checkpoint occurs), with a backup kept in ``gwin.hdf.bkup``. At each
checkpoint, the number of independent samples that have been obtained to that
point will be computed. If the number of independent samples is greater than or
equal to ``n-independent-samples``, the code will finish and exit. Upon
finishing, ``gwin.hdf.checkpoint`` will be renamed to ``gwin.hdf``, and
``gwin.hdf.bkup`` will be deleted.

If the job fails for any reason while running (say your computer loses power)
you can resume from the last checkpoint by re-running the same command as
above, but adding ``--resume-from-checkpoint``. In this case, the code will
automatically detect the checkpoint file, and pickup from where it last left
off.

----------------
GW150914 example
----------------

The configuration file ``gwin.ini`` used for the above injection is the same
as what you need to analyze the data containing GW150914. You only need to
change the command-line arguments to ``gwin`` to point it to the correct data.
To do that, do one of the following:

* **If you are a LIGO member and are running on a LIGO Data Grid cluster:**
  you can use the LIGO data server to automatically obtain the frame files.
  Simply set the following environment variables::

    FRAMES="--frame-type H1:H1_HOFT_C02 L1:L1_HOFT_C02"
    CHANNELS="H1:H1:DCS-CALIB_STRAIN_C02 L1:L1:DCS-CALIB_STRAIN_C02"

* **If you are not a LIGO member, or are not running on a LIGO Data Grid
  cluster:** you need to obtain the data from the
  `LIGO Open Science Center <https://losc.ligo.org>`_. First run the following
  commands to download the needed frame files to your working directory::

    wget https://losc.ligo.org/s/events/GW150914/H-H1_LOSC_4_V2-1126257414-4096.gwf
    wget https://losc.ligo.org/s/events/GW150914/L-L1_LOSC_4_V2-1126257414-4096.gwf

  Then set the following enviornment variables::

    FRAMES="--frame-files H1:H-H1_LOSC_4_V2-1126257414-4096.gwf L1:L-L1_LOSC_4_V2-1126257414-4096.gwf"
    CHANNELS="H1:LOSC-STRAIN L1:LOSC-STRAIN"

Now run::

    # trigger parameters
    TRIGGER_TIME=1126259462.42

    # data to use
    # the longest waveform covered by the prior must fit in these times
    SEARCH_BEFORE=6
    SEARCH_AFTER=2

    # use an extra number of seconds of data in addition to the data specified
    PAD_DATA=8

    # PSD estimation options
    PSD_ESTIMATION="H1:median L1:median"
    PSD_INVLEN=4
    PSD_SEG_LEN=16
    PSD_STRIDE=8
    PSD_DATA_LEN=1024

    # sampler parameters
    CONFIG_PATH=gwin.ini
    OUTPUT_PATH=gwin.hdf
    IFOS="H1 L1"
    SAMPLE_RATE=2048
    F_HIGHPASS=15
    F_MIN=20
    N_UPDATE=500
    N_WALKERS=5000
    N_SAMPLES=5000
    N_CHECKPOINT=1000
    PROCESSING_SCHEME=cpu

    # the following sets the number of cores to use; adjust as needed to
    # your computer's capabilities
    NPROCS=12

    # get coalescence time as an integer
    TRIGGER_TIME_INT=${TRIGGER_TIME%.*}

    # start and end time of data to read in
    GPS_START_TIME=$((${TRIGGER_TIME_INT} - ${SEARCH_BEFORE} - ${PSD_INVLEN}))
    GPS_END_TIME=$((${TRIGGER_TIME_INT} + ${SEARCH_AFTER} + ${PSD_INVLEN}))

    # start and end time of data to read in for PSD estimation
    PSD_START_TIME=$((${TRIGGER_TIME_INT} - ${PSD_DATA_LEN}/2))
    PSD_END_TIME=$((${TRIGGER_TIME_INT} + ${PSD_DATA_LEN}/2))

    # run sampler
    # specifies the number of threads for OpenMP
    # Running with OMP_NUM_THREADS=1 stops lalsimulation
    # to spawn multiple jobs that would otherwise be used
    # by gwin and cause a reduced runtime.
    OMP_NUM_THREADS=1 \
    gwin --verbose \
        --seed 12 \
        --instruments ${IFOS} \
        --gps-start-time ${GPS_START_TIME} \
        --gps-end-time ${GPS_END_TIME} \
        --channel-name ${CHANNELS} \
        ${FRAMES} \
        --strain-high-pass ${F_HIGHPASS} \
        --pad-data ${PAD_DATA} \
        --psd-estimation ${PSD_ESTIMATION} \
        --psd-start-time ${PSD_START_TIME} \
        --psd-end-time ${PSD_END_TIME} \
        --psd-segment-length ${PSD_SEG_LEN} \
        --psd-segment-stride ${PSD_STRIDE} \
        --psd-inverse-length ${PSD_INVLEN} \
        --sample-rate ${SAMPLE_RATE} \
        --low-frequency-cutoff ${F_MIN} \
        --config-file ${CONFIG_PATH} \
        --output-file ${OUTPUT_PATH} \
        --processing-scheme ${PROCESSING_SCHEME} \
        --sampler kombine \
        --burn-in-function max_posterior \
        --update-interval ${N_UPDATE} \
        --nwalkers ${N_WALKERS} \
        --n-independent-samples ${N_SAMPLES} \
        --checkpoint-interval ${N_CHECKPOINT} \
        --nprocesses ${NPROCS} \
        --save-strain \
        --save-psd \
        --save-stilde \
        --force

As discussed in the injection example above, the code will write results to
``gwin.hdf.checkpoint`` at every checkpoint interval, and will continue to run
until it has obtained at least as many independent samples as specified by
``n-independent-samples``.  When this happens, ``gwin.hdf.checkpoint`` will be
moved to ``gwin.hdf`` and the code will exit.
