==========
Background
==========

ap_set_background
  User provided background value in flux (float)

ap_set_background_noise
  User provided background noise level in flux (float)

ap_background_speedup
  speedup factor for background calculation. Speedup is achieved by reducing the number of pixels used
  in calculating the background, only use this option for large images where all pixels are not needed
  to get an accurate background estimate (int)
