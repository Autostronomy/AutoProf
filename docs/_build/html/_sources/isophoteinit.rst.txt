=======================
Isophote Initialization
=======================

Description
-----------

**pipeline label: isophoteinit**

In the :func:`~autoprofutils.Isophote_Initialize.Isophote_Initialize`,
a global position angle and ellipticity are fit in a two step process.
First, a series of circular isophotes are geometrically sampled until
they approach the background level of the image.  An FFT is taken for
the flux values around each isophote and the phase of the second
coefficient is used to determine a direction.  The average direction
for the outer isophotes is taken as the position angle of the galaxy.
Second, with fixed position angle the ellipticity is optimized to
minimize the amplitude of the second FFT coefficient relative to the
median flux in an isophote.

To compute the error on position angle we use the standard deviation
of the outer values from step one.  For ellipticity the error is
computed by optimizing the ellipticity for multiple isophotes within 1
PSF length of each other.

Output format:

.. code-block:: python
   
  {'init ellip': , # Ellipticity of the global fit (float)
   'init pa': ,# Position angle of the global fit (float)
   'init R': ,# Semi-major axis length of global fit (float)
   'auxfile initialize': # optional, message for aux file to record the global ellipticity and postition angle (string)
  
  }


Config Parameters
-----------------



