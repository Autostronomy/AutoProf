===============
Getting Started
===============

AutoProf is a :py:class:`~Pipeline` building utility

testing :py:func:`~autoprofutils.Background.Background_Mode` another one

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

Batch Forced Photometry
-----------------------

:ap_process_mode: forced image list

You may be able to guess at this point.
To run forced photometry in batch mode, start with a single image forced photometry config file and convert single values into lists wherever necessary.

