===============
Getting Started
===============

AutoProf is a :class:`Pipeline` building code at its core. However, the rich functionality provided by this framework is safely hidden beneath a simple to use exterior. As a beginning user, you need only provide the most basic information (path to an image, pixel scale, maybe the photometric zero point) and AutoProf will assume the rest. As you get better aquainted with the configuration file construction, you will be able to access more powerful functionality.


Getting Started On A Single Image
---------------------------------

:ap_process_mode: image

The fastest way to run AutoProf for yourself will be for a single image.
The steps below describe how to get AutoProf to run on an image that you provide.
To run in batch mode for many images there isn't much to change, but that will be described later.

1. Copy the basic config file below. 
#. In the config file, edit the following lines::
   
    ap_pixscale = # your image scale in arcsec/pix
    ap_image_file = # filename of your image
     
   and change any other options as they pertain to your image. If you aren't sure what to do, you can just remove an option from the config file. All that is truly needed to get started is *ap_process_mode*, *ap_pixscale*, and *ap_image_file*.
#. Run AutoProf on the configuration file::
   
    autoprof config.py
     
#. Check the .prof file for the surface brightness profile.

Here is the basic config file just make sure to save it as a ``.py`` file::

    ap_process_mode = "image"

    ap_image_file = "<path to your image file>.fits"
    ap_name = "yourimagename"
    ap_pixscale = 0.262
    ap_zeropoint = 22.5
    ap_doplot = True
    ap_isoclip = True

Check the .aux file for extra information, including checks on the success of the fit.
Check the .log file for messages about the progress of the fit, which are updated throughout the fitting procedure.
Also, look at the diagnostic plots to see if the fit appears to have worked.

Note that AutoProf has a list of arguments that it is expecting and it only checks for those.
You can therefore make any variables or functions you need in the config file to construct your list of image files so long as they don't conflict with any of the expected AutoProf arguments.
To see what parameters AutoProf uses by default, see the :doc:`parameters` section.


Other Processing Modes
----------------------

There are 4 main processing modes for AutoProf: image, image list, forced image, forced image list.
The subsections below will outline how to use each mode.

Running AutoProf In Batch Mode
------------------------------

:ap_process_mode: image list

Running AutoProf in batch mode is relatively simple once you have learned how to work with a single image.
For an example batch processing config file, see the *test_batch_config.py* file in the AutoProf test directory.
You must modify the *ap_process_mode* command to::

  ap_process_mode = 'image list'

Then, some config parameters will need to be turned into lists.
The *ap_image_file* variable should now be a list of image files, instead of a single string.
Any other config parameter can be made into a list or left as a single value.
If a parameter is a list, it must be the same length as the *ap_image_file* list, if it is a single value then that value will be used for all instances.
For example, the *ap_pixscale* variable can be left as a float value and AutoProf will use that same value for all images.

Also unique to batch processing is the availability of parallel processing with the *ap_n_procs* variable.
Since image analysis is an "embarrassingly parallel problem" AutoProf can analyze many images simultaneously.
It is suggested that you set *ap_n_procs* equal to the number of processors you have, although you may need to experiment.
Especially if you don't have much ram, this may be the limiting factor.

Here is an example batching config file just make sure to save it as a ``.py`` file::

    ap_process_mode = "image list"

    ap_n_procs = 4
    ap_image_file = [
        "<path to file 1>.fits",
        "<path to file 2>.fits",
        "<path to file 3>.fits",
        "<path to file 4>.fits",
    ]
    ap_pixscale = 0.262
    ap_name = [
        "yourimagename1",
        "yourimagename2",
        "yourimagename3",
        "yourimagename4",
    ]
    ap_doplot = True

Note that this is a python file meaning that your config file can have logic in it! For example you may use glob to collect all the files in a directory and assign them to the ``ap_image_file`` variable instead of writing them all out manually.

Forced Photometry
-----------------

:ap_process_mode: forced image

Forced photometry allows one to take an isophotal solution from one image and apply it (kind of) blindly to another image.
An example forced photometry config file can be found in AutoProf test directory and is named *test_forced_config.py* which will only work once *test_config.py* has been run.
Forced photometry can be used for multi-band images from the same telescope, or between telescopes.
One may need to adjust for different pixel scales, or have to re-center between images, but the ultimate goal is to apply the same ellipticity and position angle profile to the galaxy in each band.
Running forced photometry is very similar to the other processing modes with one extra required parameter (the pre-fit .prof file).

1. Copy the example forced photometry file below.
#. In the config file, edit the following lines::
   
    ap_pixscale = # your image scale in arcsec/pix
    ap_image_file = # filename of your image
    ap_forcing_profile = # filename for the .prof output
     
#. Run AutoProf on the configuration file::
   
    autoprof forced_config.py
     
#. Check the .prof file for the surface brightness profile.

Check the .aux file for extra information, including checks on the success of the fit.
Check the .log file for messages about the progress of the fit, which are updated throughout the fitting procedure.
Also, look at the diagnostic plots to see if the fit appears to have aligned properly with the new image.


**forced photometry parameters**

ap_forced_pa_shift
  global rotation to apply to all forced isophotes. Useful if the base image and the forced image are rotated relative to each other. Likely
  will also need to re-center the galaxy, which can be done by modifying *ap_new_pipeline_steps*. Default is zero. (float) 

Here is the forced photometry file just make sure to save it as a ``.py`` file::

    ap_process_mode = "forced image"
    
    ap_image_file = "test_ESO479-G1_r.fits"
    ap_name = "testforcedimage"
    ap_pixscale = 0.262
    ap_doplot = True
    ap_isoclip = True
    ap_forcing_profile = "testimage.prof"



Batch Forced Photometry
-----------------------

:ap_process_mode: forced image list

You may be able to guess at this point.
To run forced photometry in batch mode, start with a single image forced photometry config file and convert single values into lists wherever necessary.

Going further, descision tree pipelines
---------------------------------------

It's possible to incorporate descision trees into the AutoProf pipeline. This is very flexible, allowing different steps to be run depending on the results from previous steps. Here is an example file to show what can be done::

    import numpy as np

    ap_process_mode = "image"

    ap_image_file = "test_ESO479-G1_r.fits"
    ap_pixscale = 0.262
    ap_name = "testtreeimage"
    ap_doplot = True
    ap_isoband_width = 0.05
    ap_samplegeometricscale = 0.05
    ap_truncate_evaluation = True
    ap_ellipsemodel_resolution = 2.0

    ap_fouriermodes = 4
    ap_slice_anchor = {"x": 1700.0, "y": 1350.0}
    ap_slice_length = 300.0
    ap_isoclip = True

    
    def My_Edgon_Fit_Method(IMG, results, options):
        N = 100
        return IMG, {
            "fit ellip": np.array([results["init ellip"]] * N),
            "fit pa": np.array([results["init pa"]] * N),
            "fit ellip_err": np.array([0.05] * N),
            "fit pa_err": np.array([5 * np.pi / 180] * N),
            "fit R": np.logspace(0, np.log10(results["init R"] * 2), N),
        }


    def whenrerun(IMG, results, options):
        count_checks = 0
        for k in results["checkfit"].keys():
            if not results["checkfit"][k]:
                count_checks += 1
    
        if count_checks <= 0:  # if checks all passed, carry on
            return None, {"onloop": options["onloop"] if "onloop" in options else 0}
        elif (
            not "onloop" in options
        ):  # start by simply re-running the analysis to see if AutoProf got stuck
            return "head", {"onloop": 1}
        elif options["onloop"] == 1 and (
            not results["checkfit"]["FFT coefficients"]
            or not results["checkfit"]["isophote variability"]
        ):  # Try smoothing the fit the result was chaotic
            return "head", {"onloop": 2, "ap_regularize_scale": 3, "ap_fit_limit": 5}
        elif (
            options["onloop"] == 1 and not results["checkfit"]["Light symmetry"]
        ):  # Try testing larger area to find center if fit found high asymmetry (possibly stuck on a star)
            return "head", {"onloop": 2, "ap_centeringring": 20}
        else:  # Don't try a third time, just give up
            return None, {"onloop": options["onloop"] if "onloop" in options else 0}


    ap_new_pipeline_methods = {
        "branch edgeon": lambda IMG, results, options: (
            "edgeon" if results["init ellip"] > 0.8 else "standard",
            {},
        ),
        "branch rerun": whenrerun,
        "edgeonfit": My_Edgon_Fit_Method,
    }
    ap_new_pipeline_steps = {
        "head": [
            "background",
            "psf",
            "center",
            "isophoteinit",
            "branch edgeon",
        ],  
        "standard": [
            "isophotefit",
            "starmask",
            "isophoteextract",
            "checkfit",
            "branch rerun",
            "writeprof",
            "plot image",
            "ellipsemodel",
            "axialprofiles",
            "radialprofiles",
            "sliceprofile",
        ],
        "edgeon": [
            "edgeonfit",
            "isophoteextract",
            "radsample",
            "axialprofiles",
            "writeprof",
        ],
    }

