==========
Background
==========

Description
-----------

**pipeline label: background**

The default background calculation is done by searching for the "mode" of the pixel flux values.
First, the method extracts the border of the image, taking all pixels that are within 1/5th the image width of the edge.
Then it constructs a density profile in flux space and finds the peak.
This peak is used as the background level, a few rounds of sigma clipping are applied to remove bright signals before taking the background noise level (measured as an interquartile range).

Output format:
.. code-block:: python
   
   {'background': , # flux value representing the background level (float)
    'background noise': ,# measure of scatter around the background level (float)
    'background uncertainty': ,# optional, uncertainty on background level (float)
    'auxfile background': # optional, message for aux file to record background level (string)
   
   }


Config Parameters
-----------------

ap_set_background
  User provided background value in flux (float)

ap_set_background_noise
  User provided background noise level in flux (float)

ap_background_speedup
  speedup factor for background calculation. Speedup is achieved by reducing the number of pixels used
  in calculating the background, only use this option for large images where all pixels are not needed
  to get an accurate background estimate (int)
