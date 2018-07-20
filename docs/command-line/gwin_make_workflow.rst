#################################################################
``gwin_make_workflow``: A parameter estimation workflow generator
#################################################################

===============
Introduction
===============

The executable ``gwin_make_workflow`` is a workflow generator to setup a parameter estimation analysis.

===========================
Workflow configuration file
===========================

A simple workflow configuration file

.. literalinclude:: ../../examples/workflow/GW150914_example/workflow.ini
   :language: ini

============================
Inference configuration file
============================

You will also need a configuration file with sections that tells ``gwin`` how to construct the priors. A simple inference configuration file is

.. literalinclude:: ../../examples/workflow/GW150914_example/gwin.ini
   :language: ini

A simple configuration file for parameter estimation on the ringdown is::

    [model]
    name = gaussian_noise

    [variable_params]
    ; parameters to vary in inference sampler
    tc =
    f_0 =
    tau =
    amp =
    phi =

    [static_params]
    ; parameters that do not vary in inference sampler
    approximant = FdQNM
    ra = 2.21535724066
    dec = -1.23649695537
    polarization = 0.
    f_lower = 28.0
    f_final = 512

    [prior-tc]
    ; how to construct prior distribution
    name = uniform
    min-tc = 1126259462.4
    max-tc = 1126259462.5

    [prior-f_0]
    ; how to construct prior distribution
    name = uniform
    min-f_0 = 200.
    max-f_0 = 300.

    [prior-tau]
    ; how to construct prior distribution
    name = uniform
    min-tau = 0.0008
    max-tau = 0.020

    [prior-amp]
    ; how to construct prior distribution
    name = uniform
    min-amp = 0
    max-amp = 1e-20

    [prior-phi]
    ; how to construct prior distribution
    name = uniform_angle

If you want to use another variable parameter in the inference sampler then add its name to ``[variable_params]`` and add a prior section like shown above.

When working on real data, it is necessary to marginalise over calibration uncertainty.
The model and parameters describing the calibration uncertainty can be passed in another ini file, e.g.:

.. literalinclude:: ../../examples/workflow/GW150914_example/calibration.ini
:language: ini

=====================
Generate the workflow
=====================

To generate a workflow you will need your configuration files.
The workflow creates a single config file ``inference.ini`` from all the files specified in ``INFERENCE_CONFIG_PATH``.
We set the following enviroment variables for this example::

    # name of the workflow
    WORKFLOW_NAME="r1"

    # path to output dir
    OUTPUT_DIR=output

    # input configuration files
    CONFIG_PATH=workflow.ini
    INFERENCE_CONFIG_PATH="gwin.ini calibration.ini"

Specify a directory to save the HTML pages::

    # directory that will be populated with HTML pages
    HTML_DIR=${HOME}/public_html/inference_test

If you want to run on the loudest triggers from a PyCBC coincident search workflow then run::

    # run workflow generator on triggers from workflow
    gwin_make_workflow --workflow-name ${WORKFLOW_NAME} \
        --config-files ${CONFIG_PATH} \
        --inference-config-file ${INFERENCE_CONFIG_PATH} \
        --output-dir ${OUTPUT_DIR} \
        --output-file ${WORKFLOW_NAME}.dax \
        --output-map ${OUTPUT_MAP_PATH} \
        --bank-file ${BANK_PATH} \
        --statmap-file ${STATMAP_PATH} \
        --single-detector-triggers ${SNGL_H1_PATHS} ${SNGL_L1_PATHS}
        --config-overrides workflow:start-time:${WORKFLOW_START_TIME} \
                           workflow:end-time:${WORKFLOW_END_TIME} \
                           workflow-inference:data-seconds-before-trigger:8 \
                           workflow-inference:data-seconds-after-trigger:8 \
                           results_page:output-path:${HTML_DIR} \
                           results_page:analysis-subtitle:${WORKFLOW_NAME}

Where ``${BANK_FILE}`` is the path to the template bank HDF file, ``${STATMAP_FILE}`` is the path to the combined statmap HDF file, ``${SNGL_H1_PATHS}`` and ``${SNGL_L1_PATHS}`` are the paths to the merged single-detector HDF files,  and ``${WORKFLOW_START_TIME}`` and ``${WORKFLOW_END_TIME}`` are the start and end time of the coincidence workflow.

Else you can run from a specific GPS end time with the ``--gps-end-time`` option like::

    # run workflow generator on specific GPS end time
    gwin_make_workflow --workflow-name ${WORKFLOW_NAME} \
        --config-files ${CONFIG_PATH} \
        --inference-config-file ${INFERENCE_CONFIG_PATH} \
        --output-dir ${OUTPUT_DIR} \
        --output-file ${WORKFLOW_NAME}.dax \
        --output-map ${OUTPUT_MAP_PATH} \
        --gps-end-time ${GPS_END_TIME} \
        --config-overrides workflow:start-time:$((${GPS_END_TIME}-16)) \
                           workflow:end-time:$((${GPS_END_TIME}+16)) \
                           workflow-inference:data-seconds-before-trigger:2 \
                           workflow-inference:data-seconds-after-trigger:2 \
                           inference:psd-start-time:$((${GPS_END_TIME}-300)) \
                           inference:psd-end-time:$((${GPS_END_TIME}+748)) \
                           results_page:output-path:${HTML_DIR} \
                           results_page:analysis-subtitle:${WORKFLOW_NAME}


Where ``${GPS_END_TIME}`` is the GPS end time of the trigger.

For the CBC example above define the environment variables ``GPS_END_TIME=1126259462`` and ``OUTPUT_MAP_PATH=output.map``.

=============================
Plan and execute the workflow
=============================

Finally plan and submit the workflow with::

    # submit workflow
    pycbc_submit_dax --dax ${WORKFLOW_NAME}.dax \
        --accounting-group ligo.dev.o3.cbc.explore.test \
        --enable-shared-filesystem

