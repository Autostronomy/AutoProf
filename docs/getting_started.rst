===============
Getting Started
===============

Test inline link to :py:func:`~autoprofutils.Background.Background_Mode` to see if that works. Also trying :func:`~autoprofutils.Background` and also :meth:`~autoprofutils.Background.Background_Mode` and another :class:`~Pipeline` to see what we get.

Getting Started On A Single Image
---------------------------------

:ap_process_mode: image

The fastest way to run AutoProf for yourself will be for a single image.
The steps below describe how to get AutoProf to run on an image that you provide.
To run in batch mode for many images there isn't much to change, but that will be described later.

1. Copy the *test_config.py* script from the AutoProf test directory to the directory with your image. 
#. In the config file, edit the following lines::
   
    ap_pixscale = # your image scale in arcsec/pix
    ap_image_file = # filename of your image
     
   and change any other options as they pertain to your image. If you aren't sure what to do, you can just remove an option from the config file. All that is truly needed to get started is *ap_process_mode*, *ap_pixscale*, and *ap_image_file*.
#. Run AutoProf on the configuration file::
   
    autoprof config.py
     
#. Check the .prof file for the surface brightness profile.

Check the .aux file for extra information, including checks on the success of the fit.
Check the .log file for messages about the progress of the fit, which are updated throughout the fitting procedure.
Also, look at the diagnostic plots to see if the fit appears to have worked.

Note that AutoProf has a list of arguments that it is expecting (see *List Of AutoProf Arguments* for a full list) and it only checks for those.
You can therefore make any variables or functions you need in the config file to construct your list of image files so long as they don't conflict with any of the expected AutoProf arguments.

Main Config Parameters
----------------------

**Required Parameters**

ap_pixscale
  pixel scale in arcsec/pixel (float)

ap_image_file
  path to fits file with image data (string)

ap_process_mode
  analysis mode for AutoProf to run in (string)

ap_forcing_profile
  (required for forced photometry) file path to .prof file providing forced photometry PA and ellip values to apply to *ap_image_file* (string)

**High Level Parameters**

ap_saveto
  path to directory where final profile should be saved. Default is the current directory. (string)

ap_name
  name to use for the galaxy, this will be the name used in output files and in the log file. Default is taken from the filename of the fits image. (string)

ap_n_procs
  number of processes to create when running in batch mode. Default is 1. (int)
ap_doplot
  Generate diagnostic plots during processing. Default is False. (bool).

ap_plotpath
  Path to file where diagnostic plots should be written, see also *ap_doplot*. Default is current directory. (string)

ap_plotdpi
  sets dpi for plots (default 300). Can be used to reduce file size, or to increase detail in images (int)

ap_hdulelement
  index for hdul of fits file where image exists. Default is 0. (int)

ap_new_pipeline_methods
  Allows user to set methods for the AutoProf pipeline analysis. See *Modifying Pipeline Methods* for more information (dict)

ap_new_pipeline_steps
  Allows user to change the AutoProf analysis pipeline by adding, removing, or re-ordering steps. See *Modifying Pipeline Steps* for more information (list)

ap_zeropoint
  Photometric zero point, default is 22.5 (float)

ap_nologo
  tells AutoProf not to put it's logo on plots. Please only use this for figures that will be used in publications that don't allow logos (bool)

There is one argument that AutoProf can take in the command line, which is the name of the log file.
The log file stores information about everything that AutoProf is doing, this is useful for diagnostic purposes.
By default, AutoProf will name the log file *AutoProf.log*, if you wish to make it a different filename then add the filename when running AutoProf:

.. code-block:: bash
   
  autoprof config.py newlogfilename.log

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

Forced Photometry
-----------------

:ap_process_mode: forced image

Forced photometry allows one to take an isophotal solution from one image and apply it (kind of) blindly to another image.
An example forced photometry config file can be found in AutoProf test directory and is named *test_forced_config.py* which will only work once *test_config.py* has been run.
Forced photometry can be used for multi-band images from the same telescope, or between telescopes.
One may need to adjust for different pixel scales, or have to re-center between images, but the ultimate goal is to apply the same ellipticity and position angle profile to the galaxy in each band.
Running forced photometry is very similar to the other processing modes with one extra required parameter (the pre-fit .prof file).

1. Copy the *test_forced_config.py* script to the directory with your image. 
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


Batch Forced Photometry
-----------------------

:ap_process_mode: forced image list

You may be able to guess at this point.
To run forced photometry in batch mode, start with a single image forced photometry config file and convert single values into lists wherever necessary.

