=============
Extra Methods
=============

As well as the default pipeline, AutoProf has a number of pre-built methods for different use cases and extending one's analysis beyond a direct isophote fit.
Here we outline a basic description of those methods and some of their use cases (though you may find clever new uses!).
Some are meant as drop-in replacements for methods in the default pipeline, and others are meant as entirely new methods.

Background - Dilated Sources
----------------------------------------------------------------------

**pipeline label: 'background dilatedsources'**

:func:`~autoprof.pipeline_steps.Background.Background_DilatedSources`

Using the photutils *make_source_mask* function, a mask is constructed for bright sources in the image.
A "dilation" is then applied to expand the masked area around each source.
The background is then taken as the median of the unmasked pixels.
The noise is half the 16-84 quartile range of unmasked pixels.

Background - Basic
----------------------------------------------------------------------

**pipeline label: 'background basic'**

:func:`~autoprof.pipeline_steps.Background.Background_Basic`

All pixels in the outer boarder of the image (outer 1/4th of the image) are taken, the mean value is the background, standard deviation is the noise.

Background - Unsharp masking
----------------------------------------------------------------------

**pipeline label: 'background unsharp'**

:func:`~autoprof.pipeline_steps.Background.Background_Unsharp`

A two-dimensional FFT is taken on the image, the background level is computed across the image using a low pass filter of the FFT coefficients.

PSF - IRAF
----------------------------------------------------------------------

**pipeline label: 'psf IRAF'**

:func:`~autoprof.pipeline_steps.PSF.PSF_IRAF`

The photutils IRAF star finder wrapper is used to identify stars in the image, the psf is taken as the average fwhm fitted by IRAF.

PSF - Image
----------------------------------------------------------------------

**pipeline label: 'psf img'**

:func:`~autoprof.pipeline_steps.PSF.PSF_Image`

Method to construct an image of the PSF by stacking many stars extracted from the image.

PSF - Deconvolve
----------------------------------------------------------------------

**pipeline label: 'psf deconvolve'**

:func:`~autoprof.pipeline_steps.PSF.PSF_deconvolve`

Deconvolves a provided PSF from the primary image. Add the step 'psf deconvolve'
early in the pipeline steps and the primary image will be deconvolved using
Lucy-Richardson deconvolution. This is an approximate deconvolution useful in
many scenarios, but it depends on the number of iterations. You can use
`ap_psf_deconvolution_iterations` to set the number of iterations.

Center - Mean
----------------------------------------------------------------------

**pipeline label: 'center mean'**

:func:`~autoprof.pipeline_steps.Center.Center_HillClimb_mean`

Similar to the standard center finding method, except flux values along circular apertures are evaluated using the mean (instead of the median) which is more accurate in the low S/N limit that pixel values are integers.

Center - 2D Gaussian
----------------------------------------------------------------------

**pipeline label: 'center 2DGaussian'**

:func:`~autoprof.pipeline_steps.Center.Center_2DGaussian`

Wrapper for photutils center finding method which fits a 2D Gaussian to the image in order to find the center of a galaxy.

Center - 1D Gaussian
----------------------------------------------------------------------

**pipeline label: 'center 1DGaussian'**

:func:`~autoprof.pipeline_steps.Center.Center_1DGaussian`

Wrapper for photutils center finding method which fits a series of 1D Gaussians to slices of the image to identify the galaxy center.

Center - Of Mass
----------------------------------------------------------------------

**pipeline label: 'center OfMass'**

:func:`~autoprof.pipeline_steps.Center.Center_OfMass`

Wrapper for basic method which finds the flux centroid of an image to determine the center.

Isophote Initialize - Mean
----------------------------------------------------------------------

**pipeline label: 'isophoteinit mean'**

:func:`~autoprof.pipeline_steps.Isophote_Initialize.Isophote_Initialize_mean`

Similar to the standard isophote initialization method, except flux values along isophotes are evaluated using the mean (instead of the median) which is more accurate in the low S/N limit that pixel values are integers.

Plot Clean Image
----------------------------------------------------------------------

**pipeline label: 'plot image'**

:func:`~autoprof.pipeline_steps.Plotting_Steps.Plot_Galaxy_Image`

Simply plots an image of the galaxy using hybrid histogram equalization and log scale, without any other features or tests drawn on top. This can be useful for inspecting the image for spurious features without any ellipses, lines, or other objects drawn overtop. The size of the image will be based on when the step is called in the pipeline, if it is called early in the pipeline then a larger and less centered image will be used, calling later in the pipeline will use later pieces of information to choose the image size and centering.

Isophote Fitting - Fixed
----------------------------------------------------------------------

**pipeline label: 'isophotefit fixed'**

:func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FixedPhase`

Simply applies fixed position angle and ellipticity at the initialization values instead of fitting each isophote.

Isophote Fitting - Mean
----------------------------------------------------------------------

**pipeline label: 'isophotefit mean'**

:func:`~autoprof.pipeline_steps.Isophote_Fit.Isophote_Fit_FFT_mean`

Similar to the standard isophote fitting method, except flux values along isophotes are evaluated using the mean (instead of the median) which is more accurate in the low S/N limit that pixel values are integers.

Isophote Fitting - photutils
----------------------------------------------------------------------

**pipeline label: 'isophotefit photutils'**

:func:`~autoprof.pipeline_steps.Isophote_Fit.Photutils_Fit`

Wrapper for the photutils isophote fitting method which is based on Jedzejewski 1987.

Isophote Extraction - photutils
----------------------------------------------------------------------

**pipeline label: 'isophoteextract photutils'**

:func:`~autoprof.pipeline_steps.Isophote_Extract.Isophote_Extract_Photutils`

Wrapper for the photutils isophote extraction method which returns the mean intensity along each isophote. This method can be called without a fitting step (e.g. 'isophotefit photutils') as it will do it's own fitting.

Masking - Bad Pixels
----------------------------------------------------------------------

**pipeline label: 'mask badpixels'**

:func:`~autoprof.pipeline_steps.Mask.Bad_Pixel_Mask`

Identifies pixels that meet "bad pixel" criteria set by user options and constructs a mask.

Star Masking - IRAF
----------------------------------------------------------------------

**pipeline label: 'starmask'**

:func:`~autoprof.pipeline_steps.Mask.Star_Mask_IRAF`

Using the photutils wrapper of IRAF, identifies stars in the image and masks them.

Masking - Segmentation Map
----------------------------------------------------------------------

**pipeline label: 'mask segmentation map'**

:func:`~autoprof.pipeline_steps.Mask.Mask_Segmentation_Map`

Reads in a user provided segmentation map and converts it into a mask. If a galaxy center has been found it will ignore the segmentation ID where the center lays.

Ellipse Model
----------------------------------------------------------------------

**pipeline label: 'ellipsemodel'**

:func:`~autoprof.pipeline_steps.Ellipse_Model.EllipseModel`

Constructs 2D model image of the galaxy based on the extracted surface brightness, ellipticity, and position angle profile.

Radial Profiles
----------------------------------------------------------------------

**pipeline label: 'radialprofiles'**

:func:`~autoprof.pipeline_steps.Radial_Profiles.Radial_Profiles`

Samples surface brightness values radially from the center of the galaxy. The radial samples are placed on the semi-minor/major axes by default, though more wedges can be requested and their angle can be specified by the user.

Axial Profiles
----------------------------------------------------------------------

**pipeline label: 'axialprofiles'**

:func:`~autoprof.pipeline_steps.Axial_Profiles.Axial_Profiles`

Samples surface brightness values along lines parallel to the semi-minor axis.

Slice Profile
----------------------------------------------------------------------

**pipeline label: 'sliceprofile'**

:func:`~autoprof.pipeline_steps.Slice_Profiles.Slice_Profile`

Samples surface brightness values along a user specified line (slice) on the image. Mostly just for diagnostic purposes. Can be defined entirely in pixel coordinates instead of coordinates relative to galaxy.
