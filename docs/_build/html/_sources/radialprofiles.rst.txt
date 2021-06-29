===============
Radial Profiles
===============

Description
-----------

Coming soon

Config Parameters
-----------------

ap_radialprofiles_nwedges
  number of radial wedges to sample. Recommended choosing a power of 2, default is 4 (int)

ap_radialprofiles_width
  User set width of radial sampling in degrees. Default value is 15 degrees (float)

ap_radialprofiles_pa
  user set position angle at which to measure radial wedges relative to the global position angle, in degrees. Default is 0. (float)

ap_radialprofiles_expwidth
  tell AutoProf to use exponentially increasing widths for radial samples. In this case *ap_radialprofiles_width* corresponds to the final width of the radial sampling (bool)

ap_radialprofiles_variable_pa
  tell AutoProf to rotate radial sampling wedges with the position angle profile of the galaxy (bool)

