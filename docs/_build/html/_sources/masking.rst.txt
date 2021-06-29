=======
Maksing
=======

Description
-----------

Coming soon


Config Parameters
-----------------

- ap_badpixel_high: flux value that corresponds to a saturated pixel or bad pixel flag, all values above *ap_badpixel_high* will be masked
  		    if using the *Bad_Pixel_Mask* pipeline method. (float)
- ap_badpixel_low: flux value that corresponds to a bad pixel flag, all values below *ap_badpixel_low* will be masked
  		    if using the *Bad_Pixel_Mask* pipeline method. (float)
- ap_badpixel_exact: flux value that corresponds to a precise bad pixel flag, all values equal to *ap_badpixel_exact* will be masked
  		    if using the *Bad_Pixel_Mask* pipeline method. (float)
- ap_mask_file: path to fits file which is a mask for the image. Must have the same dimensions as the main image (string)
- ap_savemask: indicates if the star mask should be saved after fitting (bool)
