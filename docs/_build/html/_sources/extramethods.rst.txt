=============
Extra Methods
=============

As well as the default pipeline, AutoProf has a number of pre-built methods for different use cases and extending one's analysis beyond a direct isophote fit.
Here we outline a basic description of those methods and some of their use cases (though you may find clever new uses!).
Some are meant as drop-in replacements for methods in the default pipeline, and others are meant as entirely new methods.

Background - Dilated Sources
----------------------------------------------------------------------

**pipeline label: 'background dilatedsources'**

Using the photutils *make_source_mask* function, a mask is constructed for bright sources in the image.
A "dilation" is then applied to expand the masked area around each source.
The background is then taken as the median of the unmasked pixels.
The noise is half the 16-84 quartile range of unmasked pixels.

Background - Basic
----------------------------------------------------------------------

**pipeline label: 'background basic'**

All pixels in the outer boarder of the image (outer 1/4th of the image) are taken, the mean value is the background, standard deviation is the noise.

Background - Unsharp masking
----------------------------------------------------------------------

**pipeline label: 'background unsharp'**

A two-dimensional FFT is taken on the image, the background level is computed across the image using a low pass filter of the FFT coefficients.

PSF - IRAF
----------------------------------------------------------------------

**pipeline label: 'psf IRAF'**

The photutils IRAF star finder wrapper is used to identify stars in the image, the psf is taken as the average fwhm fitted by IRAF.

Center - Mean
----------------------------------------------------------------------

**pipeline label: 'center mean'**

Similar to the standard center finding method, except flux values along circular apertures are evaluated using the mean (instead of the median) which is more accurate in the low S/N limit that pixel values are integers.

Center - 2D Gaussian
----------------------------------------------------------------------

**pipeline label: 'center 2DGaussian'**

Wrapper for photutils center finding method which fits a 2D Gaussian to the image in order to find the center of a galaxy.

Center - 1D Gaussian
----------------------------------------------------------------------

**pipeline label: 'center 1DGaussian'**

Wrapper for photutils center finding method which fits a series of 1D Gaussians to slices of the image to identify the galaxy center.

Center - Of Mass
----------------------------------------------------------------------

**pipeline label: 'center OfMass'**

Wrapper for photutils method which finds the flux centroid of an image to determine the center.

Isophote Initialize - Mean
----------------------------------------------------------------------

**pipeline label: 'isophoteinit mean'**

Similar to the standard isophote initialization method, except flux values along isophotes are evaluated using the mean (instead of the median) which is more accurate in the low S/N limit that pixel values are integers.

Plot Clean Image
----------------------------------------------------------------------

**pipeline label: 'plot image'**

Simply plots an image of the galaxy using hybrid histogram equalization and log scale, without any other features or tests drawn on top. This can be useful for inspecting the image for spurious features without any ellipses, lines, or other objects drawn overtop. The size of the image will be based on when the step is called in the pipeline, if it is called early in the pipeline then a larger and less centered image will be used, calling later in the pipeline will use later pieces of information to choose the image size and centering.

Isophote Fitting - Mean
----------------------------------------------------------------------

**pipeline label: 'isophotefit mean'**

Similar to the standard isophote fitting method, except flux values along isophotes are evaluated using the mean (instead of the median) which is more accurate in the low S/N limit that pixel values are integers.

Isophote Fitting - photutils
----------------------------------------------------------------------

**pipeline label: 'isophotefit photutils'**

Wrapper for the photutils isophote fitting method which is based on Jedzejewski 1987.

Isophote Extraction - photutils
----------------------------------------------------------------------

**pipeline label: 'isophoteextract photutils'**

Wrapper for the photutils isophote extraction method which returns the mean intensity along each isophote. This method can be called without a fitting step (e.g. 'isophotefit photutils') as it will do it's own fitting.

Masking - Bad Pixels
----------------------------------------------------------------------

**pipeline label: 'mask badpixels'**

Identifies pixels that meet "bad pixel" criteria set by user options and constructs a mask.

Star Masking - IRAF
----------------------------------------------------------------------

**pipeline label: 'starmask'**

Using the photutils wrapper of IRAF, identifies stars in the image and masks them.

Masking - Segmentation Map
----------------------------------------------------------------------

**pipeline label: 'mask segmentation map'**

Reads in a user provided segmentation map and converts it into a mask. If a galaxy center has been found it will ignore the segmentation ID where the center lays.

Ellipse Model - Fixed
----------------------------------------------------------------------

**pipeline label: 'ellipsemodel fixed'**

Constructs a 2D model image of the galaxy based on the extracted surface brightness profile and the global ellipticity and position angle values.

Ellipse Model - General
----------------------------------------------------------------------

**pipeline label: 'ellipsemodel'**

Constructs 2D model image of the galaxy based on the extracted surface brightness, ellipticity, and position angle profile.

Radial Profiles
----------------------------------------------------------------------

**pipeline label: 'radialprofiles'**

Samples surface brightness values radially from the center of the galaxy. The radial samples are placed on the semi-minor/major axes by default, though more wedges can be requested and their angle can be specified by the user.

Axial Profiles
----------------------------------------------------------------------

**pipeline label: 'axialprofiles'**

Samples surface brightness values along lines parallel to the semi-minor axis.

Slice Profile
----------------------------------------------------------------------

**pipeline label: 'sliceprofile'**

Samples surface brightness values along a user specified line (slice) on the image. Mostly just for diagnostic purposes. Can be defined entirely in pixel coordinates instead of coordinates relative to galaxy.
