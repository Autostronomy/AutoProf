================
Slicing Profiles
================

Description
-----------

Coming soon

Config Paramters
----------------

ap_slice_anchor
  Coordinates for the starting point of the slice as a dictionary formatted "{'x': x-coord, 'y': y-coord}" in pixel units. Default is the center of the galaxy. (dict)

ap_slice_pa
  Position angle of the slice in degrees, counter-clockwise relative to the x-axis. Default is the galaxy PA. (float)

ap_slice_length
  Length of the slice from anchor point in pixel units. Default is the global galaxy PA/ellipse fit semi-major axis length. (float) 

ap_slice_width
  Width of the slice in pixel units. Default is 10. (float)

ap_slice_step
  Distance between samples for the profile along the slice. Default is the PSF, or one 100th of the length if PSF isn't available. (float)

