======
Center
======

Description
-----------

**pipeline label: center**

The :func:`~autoprofutils.Center.Center_HillClimb` finding algorithm
uses a robust hill climbing algorithm to find the local brightness
peak, while ignoring peaks that are of similar scale as the
psf. Depending on the specified parameters, this function will start
at the center of the image or at a user specified center.  From the
starting point, the function will create 10 circular isophotes out to
10 times the PSF size and sample flux values around each isophote.  An
FFT is taken for the flux values around each circular isophote and the
phase of the first FFT coefficient is used to determine a direction on
the image of increasing brightness.  Taking the average direction,
flux values are sampled from the current center out to 10 times the
PSF.  A parabola is fit to the flux values and the center is then
updated to the maximum of the parabola.  This is repeated until the
update steps become negligible.  At this point, a Nelder-Mead simplex
optimizer is used for fine tuning to find a minimum in FFT first
coefficient magnitude.

Output format:

.. code-block:: python
   
   {'center': {'x': , # x coordinate of the center (pix)
	       'y': }, # y coordinate of the center (pix)
   
    'auxfile center': # optional, message for aux file to record galaxy center (string)
   
   }


Config Parameters
-----------------

ap_guess_center
  user provided starting point for center fitting. Center should be formatted as:
  {'x':float, 'y': float}, where the floats are the center coordinates in pixels. If not given, Autoprof will default to a guess of the image center. (dict)

ap_set_center
  user provided center for isophote fitting. Center should be formatted as:
  {'x':float, 'y': float}, where the floats are the center coordinates in pixels. (dict)

ap_centeringring
  Size of ring to use when finding galaxy center, in units of PSF. Default value is 10, larger rings will be robust
  to features (i.e., foreground stars), while smaller rings may be needed for small galaxies. (int)
