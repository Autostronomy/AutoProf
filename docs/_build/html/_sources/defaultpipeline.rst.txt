=========================
Default AutoProf Pipeline
=========================

In the :doc:`getting_started` section, we learned how to run AutoProf,
now we will dive into what is actually happening. Here you will learn
what is going on by default, and some of the ways that you can change
the setting to suit your project.


Standard Photoemtry
-------------------

coming soon

Forced Photometry
-----------------

coming soon

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
