====================
AutoProf Parameters
====================

Here we list all parameters used in the built-in AutoProf methods. The parameters are listed alphabetically for easy searching. For further information, links are included to the individual methods which use these parameters.

ap_axialprof_pa (float, default 0)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Axial_Profiles.Axial_Profiles`

**Description**

user set position angle at which to align the axial profiles
relative to the global position angle+90, in degrees. A common
choice would be "90" which would then sample along the
semi-major axis instead of the semi-minor axis.

ap_background_speedup (int, default 1)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Background.Background_Mode`
- :func:`~autoprof.pipeline_steps.Background.Background_Basic`

**Description**

For large images, this can be millions of pixels, which is not
really needed to achieve an accurate background level, the user
can provide a positive integer factor by which to reduce the
number of pixels used in the calculation.

ap_background_unsharp_lowpass (int, default 3)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Background.Background_Unsharp`

**Description**

User provided FFT coefficient cutoff for constructing unsharp image.

ap_badpixel_exact (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Mask.Bad_Pixel_Mask`

**Description**

flux value that corresponds to a precise bad pixel flag, all
values equal to *ap_badpixel_exact* will be masked if using the
*Bad_Pixel_Mask* pipeline method.

ap_badpixel_high (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Mask.Bad_Pixel_Mask`

**Description**

flux value that corresponds to a saturated pixel or bad pixel
flag, all values above *ap_badpixel_high* will be masked if
using the *Bad_Pixel_Mask* pipeline method.

ap_badpixel_low (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Mask.Bad_Pixel_Mask`

**Description**

flux value that corresponds to a bad pixel flag, all values
below *ap_badpixel_low* will be masked if using the
*Bad_Pixel_Mask* pipeline method.

ap_centeringring (int, default 50)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Center.Center_2DGaussian`
- :func:`~autoprof.pipeline_steps.Center.Center_1DGaussian`
- :func:`~autoprof.pipeline_steps.Center.Center_OfMass`
- :func:`~autoprof.pipeline_steps.Center.Center_HillClimb`
- :func:`~autoprof.pipeline_steps.Center.Center_HillClimb_mean`

**Description**

Size of ring to use when finding galaxy center, in units of
PSF. Larger rings will give the 2D fit more data to work with
and allow for the starting position to be further from the true
galaxy center.  Smaller rings will include fewer spurious
objects, and can stop the 2D fit from being distracted by larger
nearby objects/galaxies.

ap_ellipsemodel_replacemaskedpixels (bool, default False)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Ellipse_Model.EllipseModel`

**Description**

If True, a new galaxy image will be generated with masked pixels
replaced by the ellipse model values.

ap_ellipsemodel_resolution (float, default 1)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Ellipse_Model.EllipseModel`

**Description**

scale factor for the ellipse model resolution. Above 1 increases
the precision of the ellipse model (and computation time),
between 0 and 1 decreases the resolution (and computation
time). Note that the ellipse model resolution is defined
logarithmically, so the center will always be more resolved

ap_extractfull (bool, default False)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

Tells AutoProf to extend the isophotal solution to the edge of
the image. Will be overridden by *ap_truncate_evaluation*.

ap_fit_limit (float, default 2)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Initialize.Isophote_Initialize`
- :func:`~autoprof.pipeline_steps.Isophote_Initialize.Isophote_Initialize_mean`
- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FixedPhase`
- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_Robust`
- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_mean`

**Description**

noise level out to which to extend the fit in units of pixel background noise level. Default is 2, smaller values will end fitting further out in the galaxy image.

ap_fluxunits (str, default "mag")
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Photutils`

**Description**

units for outputted photometry. Can either be "mag" for log
units, or "intensity" for linear units.

ap_forcing_profile (string, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Initialize.Isophote_Init_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_Forced`
- :func:`~autoprof.pipeline_steps.Center.Center_Forced`

**Description**

File path to .prof file providing forced photometry PA and
ellip values to apply to *ap_image_file* (required for forced
photometry)

ap_guess_center (dict, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Plotting_Steps.Plot_Galaxy_Image`
- :func:`~autoprof.pipeline_steps.Center.Center_Forced`
- :func:`~autoprof.pipeline_steps.Center.Center_2DGaussian`
- :func:`~autoprof.pipeline_steps.Center.Center_1DGaussian`
- :func:`~autoprof.pipeline_steps.Center.Center_OfMass`
- :func:`~autoprof.pipeline_steps.Center.Center_HillClimb`
- :func:`~autoprof.pipeline_steps.Center.Center_HillClimb_mean`

**Description**

user provided starting point for center fitting. Center should
be formatted as:

.. code-block:: python

  {'x':float, 'y': float}

, where the floats are the center coordinates in pixels. If not
given, Autoprof will default to a guess of the image center.

ap_guess_psf (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.PSF.PSF_IRAF`
- :func:`~autoprof.pipeline_steps.PSF.PSF_StarFind`
- :func:`~autoprof.pipeline_steps.PSF.PSF_Image`

**Description**

Initialization value for the PSF calculation in pixels. If not
given, AutoProf will default with a guess of 1/*ap_pixscale*

ap_iso_interpolate_method (string, default 'lanczos')
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

Select method for flux interpolation on image, options are
'lanczos' and 'bicubic'. Default is 'lanczos' with a window size
of 3.

ap_iso_interpolate_start (float, default 5)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

Use a Lanczos interpolation for isophotes with semi-major axis
less than this number times the PSF.

ap_iso_interpolate_window (int, default 3)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

Window size for Lanczos interpolation, default is 3, meaning 3
pixels on either side of the sample point are used for
interpolation.

ap_iso_measurecoefs (tuple, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

tuple indicating which fourier modes to extract along fitted
isophotes. Most common is (4,), which identifies boxy/disky
isophotes. Also common is (1,3), which identifies lopsided
galaxies. The outputted values are computed as a_i =
imag(F_i)/abs(F_0) and b_i = real(F_i)/abs(F_0) where F_i is a
fourier coefficient. Not activated by default as it adds to
computation time.

ap_isoaverage_method (string, default 'median')
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Slice_Profiles.Slice_Profile`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`
- :func:`~autoprof.pipeline_steps.Axial_Profiles.Axial_Profiles`

**Description**

Select the method used to compute the averafge flux along an
isophote. Choose from 'mean', 'median', and 'mode'.  In general,
median is fast and robust to a few outliers. Mode is slow but
robust to more outliers. Mean is fast and accurate in low S/N
regimes where fluxes take on near integer values, but not robust
to outliers. The mean should be used along with a mask to remove
spurious objects such as foreground stars or galaxies, and
should always be used with caution.

ap_isoband_fixed (bool, default False)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

Use a fixed width for the size of the isobands, the width is set
by *ap_isoband_width* which now has units of pixels, the default
is 0.5 such that the full band has a width of 1 pixel.

ap_isoband_start (float, default 2)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

The noise level at which to begin sampling a band of pixels to
compute SB instead of sampling a line of pixels near the
isophote in units of pixel flux noise. Will never initiate band
averaging if the band width is less than half a pixel

ap_isoband_width (float, default 0.025)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

The relative size of the isophote bands to sample. flux values
will be sampled at +- *ap_isoband_width* \*R for each radius.

ap_isoclip (bool, default False)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

Perform sigma clipping along extracted isophotes. Removes flux
samples from an isophote that deviate significantly from the
median. Several iterations of sigma clipping are performed until
convergence or *ap_isoclip_iterations* iterations are
reached. Sigma clipping is a useful substitute for masking
objects, though careful masking is better. Also an aggressive
sigma clip may bias results.

ap_isoclip_iterations (int, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

Maximum number of sigma clipping iterations to perform. The
default is infinity, so the sigma clipping procedure repeats
until convergence

ap_isoclip_nsigma (float, default 5)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

Number of sigma above median to apply clipping. All values above
(median + *ap_isoclip_nsigma* x sigma) are removed from the
isophote.

ap_isofit_fitcoefs (tuple, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_Robust`

**Description**

Tuple of FFT coefficients to use in fitting procedure. AutoProf
will attemp to fit ellipses with these Fourier mode
perturbations. Such perturbations allow for lopsided, boxy,
disky, and other types of isophotes beyond straightforward
ellipses. Must be a tuple, not a list. Note that AutoProf will
first fit ellipses, then turn on the Fourier mode perturbations,
thus the fitting time will always be longer.

ap_isofit_fitcoefs_FFTinit (bool, default False)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_Robust`

**Description**

If True, the coefficients for the Fourier modes fitted from
ap_isofit_fitcoefs will be initialized using an FFT
decomposition along fitted elliptical isophotes. This can
improve the fit result, though it is less stable and so users
should examine the results after fitting.

ap_isofit_iterlimitmax (int, default 300)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_Robust`

**Description**

Maximum number of iterations (each iteration adjusts every
isophote once) before automatically stopping optimization. For
galaxies with lots of structure (ie detailed spiral arms) more
iterations may be needed to fully fit the light distribution,
but runtime will be longer.

ap_isofit_iterlimitmin (int, default 0)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_Robust`

**Description**

Minimum number of iterations before optimization is allowed to
stop.

ap_isofit_iterstopnochange (float, default 3)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_Robust`

**Description**

Number of iterations with no updates to parameters before
optimization procedure stops. Lower values will process galaxies
faster, but may still be stuck in local minima, higher values
are more likely to converge on the global minimum but can take a
long time to run. Fractional values are allowed though not
recomended.

ap_isofit_losscoefs (tuple, default (2,))
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_Robust`

**Description**

Tuple of FFT coefficients to use in optimization
procedure. AutoProf will attemp to minimize the power in all
listed FFT coefficients. Must be a tuple, not a list.

ap_isofit_perturbscale_ellip (float, default 0.03)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_Robust`

**Description**

Sampling scale for random adjustments to ellipticity made while
optimizing isophotes. Smaller values will converge faster, but
get stuck in local minima; larger values will escape local
minima, but takes longer to converge.

ap_isofit_perturbscale_pa (float, default 0.06)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_Robust`

**Description**

Sampling scale for random adjustments to position angle made
while optimizing isophotes. Smaller values will converge faster,
but get stuck in local minima; larger values will escape local
minima, but takes longer to converge.

ap_isofit_robustclip (float, default 0.15)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_Robust`

**Description**

quantile of flux values at which to clip when extracting values
along an isophote. Clipping outlier values (such as very bright
stars) while fitting isophotes allows for robust computation of
FFT coefficients along an isophote.

ap_isofit_superellipse (bool, default False)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_Robust`

**Description**

If True, AutoProf will fit superellipses instead of regular
ellipses. A superellipse is typically used to represent
boxy/disky isophotes. The variable controlling the transition
from a rectangle to an ellipse to a four-armed-star like shape
is C. A value of C = 2 represents an ellipse and is the starting
point of the optimization.

ap_isoinit_R_set (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Initialize.Isophote_Initialize`

**Description**

User set initial semi-major axis length, will override the calculation.

ap_isoinit_ellip_set (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Initialize.Isophote_Initialize`

**Description**

User set initial ellipticity (1 - b/a), will override the calculation.

ap_isoinit_pa_set (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Initialize.Isophote_Initialize`

**Description**

User set initial position angle in degrees, will override the calculation.

ap_mask_file (string, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Mask.Mask_Segmentation_Map`

**Description**

path to fits file which is a mask for the image. Must have the same dimensions as the main image.

ap_name (string, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Slice_Profiles.Slice_Profile`

**Description**

Name of the current galaxy, used for making filenames.

ap_plot_sbprof_set_errscale (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Photutils`

**Description**

Float value by which to scale errorbars on the SB profile
this makes them more visible in cases where the statistical
errors are very small.

ap_plot_sbprof_xlim (tuple, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Photutils`

**Description**

Tuple with axes limits for the x-axis in the SB profile
diagnostic plot.

ap_plot_sbprof_ylim (tuple, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Photutils`

**Description**

Tuple with axes limits for the y-axis in the SB profile
diagnostic plot. Be careful when using intensity units
since this will change the ideal axis limits.

ap_psf_deconvolution_iterations (int, default 50)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.PSF.PSF_deconvolve`

**Description**

number of itterations of the Richardson-Lucy deconvolution
algorithm to perform.

ap_psf_file (string, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.PSF.PSF_deconvolve`

**Description**

Optional argument. Path to PSF fits file. For best results the
image should have an odd number of pixels with the PSF centered
in the image.

ap_radialprofiles_expwidth (bool, default False)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Radial_Profiles.Radial_Profiles`

**Description**

Tell AutoProf to use exponentially increasing widths for radial
samples. In this case *ap_radialprofiles_width* corresponds to
the final width of the radial sampling.

ap_radialprofiles_nwedges (int, default 4)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Radial_Profiles.Radial_Profiles`

**Description**

number of radial wedges to sample. Recommended choosing a power
of 2.

ap_radialprofiles_pa (float, default 0)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Radial_Profiles.Radial_Profiles`

**Description**

user set position angle at which to measure radial wedges
relative to the global position angle, in degrees.

ap_radialprofiles_variable_pa (bool, default False)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Radial_Profiles.Radial_Profiles`

**Description**

Tell AutoProf to rotate radial sampling wedges with the position
angle profile of the galaxy.

ap_radialprofiles_width (float, default 15)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Radial_Profiles.Radial_Profiles`

**Description**

User set width of radial sampling wedges in degrees.

ap_regularize_scale (float, default 1)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_Robust`
- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_mean`

**Description**

scale factor to apply to regularization coupling factor between
isophotes.  Default of 1, larger values make smoother fits,
smaller values give more chaotic fits.

ap_sampleendR (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

End radius (in pixels) for isophote sampling from the
image. Default is 3 times the fit radius, also see
*ap_extractfull*.

ap_samplegeometricscale (float, default 0.1)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

growth scale for isophotes when sampling for the final output
profile.  Used when sampling geometrically. By default, each
isophote is 10\% further than the last.

ap_sampleinitR (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

Starting radius (in pixels) for isophote sampling from the
image. Note that a starting radius of zero is not
advised. Default is 1 pixel or 1PSF, whichever is smaller.

ap_samplelinearscale (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

growth scale (in pixels) for isophotes when sampling for the
final output profile. Used when sampling linearly. Default is 1
PSF length.

ap_samplestyle (string, default 'geometric')
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`
- :func:`~autoprof.pipeline_steps.Axial_Profiles.Axial_Profiles`

**Description**

indicate if isophote sampling radii should grow linearly or
geometrically. Can also do geometric sampling at the center and
linear sampling once geometric step size equals linear. Options
are: 'linear', 'geometric', 'geometric-linear'

ap_saveto (string, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Slice_Profiles.Slice_Profile`

**Description**

Directory in which to save profile

ap_scale (float, default 0.2)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FixedPhase`
- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_Robust`
- :func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_mean`

**Description**

growth scale when fitting isophotes, not the same as
*ap_sample---scale*.

ap_set_background (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Background.Background_Mode`
- :func:`~autoprof.pipeline_steps.Background.Background_DilatedSources`
- :func:`~autoprof.pipeline_steps.Background.Background_Basic`

**Description**

User provided background value in flux

ap_set_background_noise (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Background.Background_Mode`
- :func:`~autoprof.pipeline_steps.Background.Background_DilatedSources`
- :func:`~autoprof.pipeline_steps.Background.Background_Basic`

**Description**

User provided background noise level in flux

ap_set_center (dict, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Plotting_Steps.Plot_Galaxy_Image`
- :func:`~autoprof.pipeline_steps.Center.Center_Forced`
- :func:`~autoprof.pipeline_steps.Center.Center_2DGaussian`
- :func:`~autoprof.pipeline_steps.Center.Center_1DGaussian`
- :func:`~autoprof.pipeline_steps.Center.Center_OfMass`
- :func:`~autoprof.pipeline_steps.Center.Center_HillClimb`
- :func:`~autoprof.pipeline_steps.Center.Center_HillClimb_mean`

**Description**

user provided fixed center for rest of calculations. Center
should be formatted as:

.. code-block:: python

  {'x':float, 'y': float}

, where the floats are the center coordinates in pixels. If not
given, Autoprof will default to a guess of the image center.

ap_set_psf (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.PSF.PSF_IRAF`
- :func:`~autoprof.pipeline_steps.PSF.PSF_StarFind`
- :func:`~autoprof.pipeline_steps.PSF.PSF_Image`

**Description**

force AutoProf to use this PSF value (in pixels) instead of
calculating its own.

ap_slice_anchor (dict, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Slice_Profiles.Slice_Profile`

**Description**

Coordinates for the starting point of the slice as a dictionary
formatted "{'x': x-coord, 'y': y-coord}" in pixel units.

ap_slice_length (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Slice_Profiles.Slice_Profile`

**Description**

Length of the slice from anchor point in pixel units. By
default, use init ellipse semi-major axis length

ap_slice_pa (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Slice_Profiles.Slice_Profile`

**Description**

Position angle of the slice in degrees, counter-clockwise
relative to the x-axis.

ap_slice_step (float, default None)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Slice_Profiles.Slice_Profile`

**Description**

Distance between samples for the profile along the
slice. By default use the PSF.

ap_slice_width (float, default 10)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Slice_Profiles.Slice_Profile`

**Description**

Width of the slice in pixel units.

ap_truncate_evaluation (bool, default False)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`

**Description**

Stop evaluating new isophotes once two negative flux isophotes
have been recorded, presumed to have reached the end of the
profile.

ap_zeropoint (float, default 22.5)
----------------------------------------------------------------------

**Referencing Methods**

- :func:`~autoprof.pipeline_steps.Slice_Profiles.Slice_Profile`
- :func:`~autoprof.pipeline_steps.Ellipse_Model.EllipseModel`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Forced`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract`
- :func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Photutils`
- :func:`~autoprof.pipeline_steps.Axial_Profiles.Axial_Profiles`

**Description**

Photometric zero point. For converting flux to mag units.

