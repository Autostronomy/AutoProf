===
PSF
===

Description
-----------

**pipeline label: psf**

The :func:`~autoprofutils.PSF.PSF_StarFind` method uses an edge
finding convolution filter to identify candidate star pixels, then
averages their FWHM. Randomly iterates through the pixels and searches
for a local maximum. An FFT is used to identify non-circular star
candidate (artifacts or galaxies) which may have been picked up by the
edge finder. Circular apertures are placed around the star until half
the central flux value is reached, This is recorded as the FWHM for
that star. A collection of 50 stars are identified and the most
circular (by FFT coefficients) half are kept, a median is taken as the
image PSF.

Output format:

.. code-block:: python
   
  {'psf fwhm': ,# estimate of the fwhm of the PSF (float)
   'auxfile psf': # optional, message for aux file to record psf estimate (string)
  
  }

Config Parameters
-----------------

ap_guess_psf
  initialization value for the PSF calculation in pixels. If not given, AutoProf will default with a guess of 1/*ap_pixscale* (float)

ap_set_psf
  force AutoProf to use this PSF value (in pixels) instead of calculating its own. (float)

