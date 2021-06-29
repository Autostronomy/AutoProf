=====================
SB Profile Extraction
=====================

Description
-----------

**pipeline label: isophoteextract**

The :func:`default SB profile extraction
<~autoprofutils.Isophote_Extract.Isophote_Extract>` method is highly
flexible, allowing users to test a variety of techniques on their data
to determine the most robust. The user may specify a variety of
sampling arguments for the photometry extraction.  For example, a
start or end radius in pixels, or whether to sample geometrically or
linearly in radius.  Geometric sampling is the default as it is
faster.  Once the sampling profile of semi-major axis values has been
chosen, the function interpolates (spline) the position angle and
ellipticity profiles at the requested values.  For any sampling beyond
the outer radius from the *Isophotal Fitting* step, a constant value
is used.  Within 1 PSF, a circular isophote is used.

Output format:

.. code-block:: python
   
  {'prof header': , # List of strings indicating the order to write the .prof file data (list)
   'prof units': , # Dictionary with keys from header, values are strings that give the units for each variable (dict)
   'prof data': , # Dictionary with keys from header, values are lists with the data (dict)
   'prof format': # Dictionary with keys from header, values are format strings for precision of writing the data (dict)
  
  }

Config Parameters
-----------------

ap_samplegeometricscale
  growth scale for isophotes when sampling for the final output profile.
  Used when sampling geometrically. Default is 0.1, meaning each isophote is 10\% further than the last. (float)
  
ap_samplelinearscale
  growth scale (in pixels) for isophotes when sampling for the final output
  profile. Used when sampling linearly. Default is 1 PSF length. (float)
  
ap_samplestyle
  indicate if isophote sampling radii should grow linearly or geometrically. Can
  also do geometric sampling at the center and linear sampling once geometric step
  size equals linear. Default is geometric. (string, ['linear', 'geometric', 'geometric-linear'])

ap_sampleinitR
  Starting radius (in pixels) for isophote sampling from the image. Note that
  a starting radius of zero is not advised. Default is 1 pixel or 1PSF, whichever is smaller. (float)
  
ap_sampleendR
  End radius (in pixels) for isophote sampling from the image. Default is 3 times the fit radius, also see *ap_extractfull*. (float)

ap_isoband_start
  The noise level at which to begin sampling a band of pixels to compute SB instead of sampling a line of pixels near the isophote in units of pixel flux noise. Default is 2, but will never initiate band averaging if the band width is less than half a pixel (float)

ap_isoband_width
  The relative size of the isophote bands to sample. flux values will be sampled at +- *ap_isoband_width* \*R for each radius. default value is 0.025 (float)

ap_isoband_fixed
  Use a fixed width for the size of the isobands, the width is set by *ap_isoband_width* which now has units of pixels, the default is 0.5 such that the full band has a width of 1 pixel. (bool)

ap_truncate_evaluation
  Stop evaluating new isophotes once two negative flux isophotes have been recorded, presumed to have reached the end of the profile (bool)

ap_extractfull
  Tells AutoProf to extend the isophotal solution to the edge of the image. Will be overridden by *ap_truncate_evaluation* (bool)

ap_iso_interpolate_start
  Use a Lanczos interpolation for isophotes with semi-major axis less than this number times the PSF. Default is 5. (float)

ap_iso_interpolate_method
  Select method for flux interpolation on image, options are 'lanczos' and 'bicubic'. Default is 'lanczos' with a window size of 3 (string)

ap_iso_interpolate_window
  Window size for Lanczos interpolation, default is 3, meaning 3 pixels on either side of the sample point are used for interpolation (int)

ap_isoaverage_method
  Select the method used to compute the averafge flux along an isophote. Choose from 'mean', 'median', and 'mode' where median is the default.
  In general, median is fast and robust to a few outliers. Mode is slow but robust to more outliers. Mean is fast and accurate in low S/N regimes
  where fluxes take on near integer values, but not robust to outliers. The mean should be used along with a mask to remove spurious objects
  such as foreground stars or galaxies, and should always be used with caution. (string)

ap_isoclip
  Perform sigma clipping along extracted isophotes. Removes flux samples from an isophote that deviate significantly from the median. Several iterations
  of sigma clipping are performed until convergence or *ap_isoclip_iterations* iterations are reached. Sigma clipping is a useful substitute for masking
  objects, though careful masking is better. Also an aggressive sigma clip may bias results. (bool)

ap_isoclip_iterations
  Maximum number of sigma clipping iterations to perform. The default is infinity, so the sigma clipping procedure repeats until convergence (int)

ap_isoclip_nsigma
  Number of sigma above median to apply clipping. All values above (median + *ap_isoclip_nsigma* x sigma) are removed from the isophote. Default is 5. (float)

ap_fouriermodes
  integer for number of fourier modes to extract along fitted isophotes. Most popular is 4, which identifies boxy/disky isophotes. The outputted
  values are computed as a_i = real(F_i)/abs(F_0) where F_i is a fourier coefficient. Not activated by default as it adds to computation time. (int)
