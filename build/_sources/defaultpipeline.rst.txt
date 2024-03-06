=========================
Default AutoProf Pipeline
=========================

In the :doc:`getting_started` section, we learned how to run AutoProf,
now we will dive into what is actually happening. Here you will learn
what is going on by default, and some of the ways that you can change
the setting to suit your project.

If your just looking for a config file to get you started, here it is::

    ap_process_mode = "image"

    ap_image_file = "<path to your image file>.fits"
    ap_name = "yourimagename"
    ap_pixscale = 0.262
    ap_zeropoint = 22.5
    ap_doplot = True
    ap_isoclip = True

Below are some details on what will actually happen when you run the code!

Standard Photometry
-------------------

The default photometry pipeline includes a minimalist set of pipeline
steps to go from an image to an SB profile. In general, AutoProf
attempts to make no assumptions about the size of the object in the
image, although certain requirements are included for practical
purposes. The main ones to keep in mind is that the galaxy should be
roughly centered, there should be a border of sky around the galaxy
(ie, it doesn't go to the edge), and it should not be overlapping with
a similarly size or larger object.

Put plainly, the default AutoProf pipeline is as follows:

1. Background: :func:`~autoprof.pipeline_steps.Background.Background_Mode`
#. PSF: :func:`~autoprof.pipeline_steps.PSF.PSF_Assumed`
#. Center: :func:`~autoprof.pipeline_steps.Center.Center_HillClimb`
#. Initial Isophote: :func:`~autoprof.pipeline_steps.Isophote_Initialize.Isophote_Initialize`
#. Fit Isophotes: :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_Robust`
#. Extract SB Profile: :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`
#. Check Fit: :func:`~autoprof.pipeline_steps.Check_Fit.Check_Fit`
#. Write the SB Profile: :func:`~autoprof.pipeline_steps.Write_Prof.WriteProf`

each function above links to a detailed description of the method, and
the parameters that it accepts.

The reason a boarder is needed around the galaxy is because the
:func:`~autoprof.pipeline_steps.Background.Background_Mode` method uses a 1/5th
border around the image to estimate the average background level.  The
galaxy needs to be roughly centered on the image because
:func:`~autoprof.pipeline_steps.Center.Center_HillClimb` starts at the image
center by default, you can change this and give it alternate starting
coordinates if you like.  The galaxy should be non-overlapping with
large sources because that would violate the assumptions in the
:func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_Robust` step and
the :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract` step.

The final output should be two files: a profile and an aux file. The
profile (.prof) contains the SB profile and a number of other
important parameters. These include the profile of ellipticity and
position angle, but also some other useful calculations. The profile
is extended by certain options, for example you can choose to add
Fourier coefficients to the profile (typically used to examing b4/a4),
or you can run steps like the
:func:`~autoprof.pipeline_steps.Radial_Profiles.Radial_Profiles` which will add
more columns. The aux file contains global information such as the
time when the fit completed, the settings used, the global
PA/ellipticity, and any other diagnostic messages added by the various
pipeline steps.

Forced Photometry
-----------------

Forced photometry allows a user to apply the solution from one image
onto another image. The default forced photometry pipeline works as
follows:

1. Background: :func:`~autoprof.pipeline_steps.Background.Background_Mode`
#. PSF: :func:`~autoprof.pipeline_steps.PSF.PSF_Assumed`
#. Center: :func:`~autoprof.pipeline_steps.Center.Center_Forced`
#. Initial Isophote: :func:`~autoprof.pipeline_steps.Isophote_Initialize.Isophote_Init_Forced`
#. Fit Isophotes: :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_Forced`
#. Extract SB Profile: :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
#. Write the SB Profile: :func:`~autoprof.pipeline_steps.Write_Prof.WriteProf`

each function above links to a detailed description of the method, and
the parameters that it accepts.

Note that some steps remain unchanged. THe background is still
calculated as normal, this is because it is typical for the background
to change from image-to-image and between bands, so there is little
reason to expect that to remain constant. A similar argument applies
for the PSF, between observing nights and bands, the PSF can be very
different so it is re-calculated. By default the previously fit center
is used, however if you would like a new center to be fit, you can
swap out this step with the :func:`standard centering
<~autoprof.pipeline_steps.Center.Center_HillClimb>` method; this is explained in
:doc:`pipelinemanipulation`. The global isophote fit, and the full
isophote fit are of course taken from the original fit, the pixel
scale can vary between images and AutoProf will adjust
accordingly. The isophote extraction has a forcing specific method
which is near identical to the :func:`standard extraction
<~autoprof.autoprofutils.Isophote_Extract.Isophote_Extract>` method, except
that it is set up to evaluate at exactly the same ellipse parameters
as the original fit. There is no need for fit checks as no fitting has
occured. Then the profile is written as usual.

Main Config Parameters
----------------------

Below is a list of parameters which affect the pipeline at a global
level. Method specific parameters are included in their documentation.

**Required Parameters**

ap_pixscale
  pixel scale in arcsec/pixel (float)

ap_image_file
  path to fits file with image data (string)

ap_process_mode
  analysis mode for AutoProf to run in (string)

ap_forcing_profile
  (required for forced photometry) file path to .prof file providing
  forced photometry PA and ellip values to apply to *ap_image_file*
  (string)

**High Level Parameters**

ap_saveto
  path to directory where final profile should be saved. Default is
  the current directory. (string)

ap_name
  name to use for the galaxy, this will be the name used in output
  files and in the log file. Default is taken from the filename of the
  fits image. (string)

ap_n_procs
  number of processes to create when running in batch mode. Default
  is 1. (int)

ap_doplot
  Generate diagnostic plots during processing. Default is
  False. (bool).

ap_plotpath
  Path to file where diagnostic plots should be written, see also
  *ap_doplot*. Default is current directory. (string)

ap_plotdpi
  sets dpi for plots (default 300). Can be used to reduce file size,
  or to increase detail in images (int)

ap_hdulelement
  index for hdul of fits file where image exists. Default is 0. (int)

ap_new_pipeline_methods
  Allows user to set methods for the AutoProf pipeline analysis. See
  :doc:`pipelinemanipulation` for more information (dict)

ap_new_pipeline_steps
  Allows user to change the AutoProf analysis pipeline by adding,
  removing, or re-ordering steps. See :doc:`pipelinemanipulation` for
  more information (list)

ap_zeropoint
  Photometric zero point, default is 22.5 (float)

ap_nologo
  tells AutoProf not to put it's logo on plots. Please only use this
  for figures that will be used in publications that don't allow logos
  (bool)

There is one argument that AutoProf can take in the command line,
which is the name of the log file.  The log file stores information
about everything that AutoProf is doing, this is useful for diagnostic
purposes.  By default, AutoProf will name the log file *AutoProf.log*,
if you wish to make it a different filename then add the filename when
running AutoProf:

.. code-block:: bash
   
  autoprof config.py newlogfilename.log
