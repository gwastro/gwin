#####################################################################
``gwin_make_inj_workflow``: A parameter estimation workflow generator
#####################################################################

===============
Introduction
===============

The executable ``gwin_make_inj_workflow`` is a workflow generator to setup a parameter estimation analysis.

===========================
Workflow configuration file
===========================

A simple workflow configuration file::

    [workflow]
    ; basic information used by the workflow generator
    file-retention-level = all_triggers
    h1-channel-name = H1:DCS-CALIB_STRAIN_C02
    l1-channel-name = L1:DCS-CALIB_STRAIN_C02

    [workflow-ifos]
    ; the IFOs to analyze
    h1 =
    l1 =

    [workflow-inference]
    ; how the workflow generator should setup inference nodes
    num-injections = 3
    plot-group-mass = mass1 mass2 mchirp q
    plot-group-orientation =  inclination polarization ra dec
    plot-group-distance = distance redshift
    plot-group-time = tc coa_phase

    [executables]
    ; paths to executables to use in workflow
    create_injections = ${which:pycbc_create_injections}
    inference = ${which:run_gwin}
    inference_intervals = ${which:gwin_plot_inj_intervals}
    inference_posterior = ${which:gwin_plot_posterior}
    inference_rate = ${which:gwin_plot_acceptance_rate}
    inference_recovery = ${which:gwin_plot_inj_recovery}
    inference_samples = ${which:gwin_plot_samples}
    inference_table = ${which:gwin_table_summary}
    results_page = ${which:pycbc_make_html_page}

    [create_injections]
    ; command line options use --help for more information
    ninjections = 1
    dist-section = prior
    variable-args-section = variable_params
    static-args-section = static_params

    [inference]
    ; command line options use --help for more information
    processing-scheme = cpu
    sampler = kombine
    nwalkers = 5000
    checkpoint-interval = 1000
    n-independent-samples = 5000
    update-interval = 500
    burn-in-function = max_posterior
    nprocesses = 24
    fake-strain = aLIGOZeroDetHighPower
    psd-model = aLIGOZeroDetHighPower
    pad-data = 8
    strain-high-pass = 15
    sample-rate = 2048
    low-frequency-cutoff = 20
    config-overrides = static_params:approximant:TaylorF2
    save-psd =
    save-strain =
    save-stilde =
    resmume-from-checkpoint =

    [pegasus_profile-inference]
    ; pegasus profile for inference nodes
    condor|request_memory = 20G
    condor|request_cpus = 24

    [inference_intervals]
    ; command line options use --help for more information

    [inference_posterior]
    ; command line options use --help for more information
    plot-scatter =
    plot-contours =
    plot-marginal =
    z-arg = logposterior

    [inference_rate]
    ; command line options use --help for more information

    [inference_recovery]
    ; command line options use --help for more information

    [inference_samples]
    ; command line options use --help for more information

    [inference_table]
    ; command line options use --help for more information

    [results_page]
    ; command line options use --help for more information
    analysis-title = "PyCBC Inference Test"

Use the ``ninjections`` option in the ``[workflow-inference]`` section to set the number of injections in the analysis.

=====================
Generate the workflow
=====================

To generate a workflow you will need your configuration files. We set the following enviroment variables for this example::

    # name of the workflow
    WORKFLOW_NAME="r1"

    # path to output dir
    OUTPUT_DIR=output

    # input configuration files
    CONFIG_PATH=workflow.ini
    INFERENCE_CONFIG_PATH=gwin.ini

Specify a directory to save the HTML pages::

    # directory that will be populated with HTML pages
    HTML_DIR=${HOME}/public_html/inference_test

If you want to run with a test likelihood function use::

    # option for using test likelihood functions
    DATA_TYPE=analytical

Otherwise if you want to run with simulated data use::

    # option for using simulated data
    DATA_TYPE=simulated_data

=============================
Plan and execute the workflow
=============================

Finally plan and submit the workflow with::

    # submit workflow
    pycbc_submit_dax --dax ${WORKFLOW_NAME}.dax \
        --accounting-group ligo.dev.o3.cbc.explore.test \
        --enable-shared-filesystem

