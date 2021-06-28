======
Center
======

ap_guess_center
  user provided starting point for center fitting. Center should be formatted as:
  {'x':float, 'y': float}, where the floats are the center coordinates in pixels. If not given, Autoprof will default to a guess of the image center. (dict)

ap_set_center
  user provided center for isophote fitting. Center should be formatted as:
  {'x':float, 'y': float}, where the floats are the center coordinates in pixels. (dict)

ap_centeringring
  Size of ring to use when finding galaxy center, in units of PSF. Default value is 10, larger rings will be robust
  to features (i.e., foreground stars), while smaller rings may be needed for small galaxies. (int)
