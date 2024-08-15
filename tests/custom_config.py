"""
This config file demonstrates the flexability of AutoProf pipelines to
perform custom tasks. In this case a mask is produced which removes
any pixel with a negative flux value, then computes the background
level of the image. This background is written to a custom aux file
and the pixel mask is saved. No isophote fitting is performed, only
measurement of background level and a count of pixels within a flux
range.

"""

import os
from datetime import datetime
from astropy.io import fits
from time import sleep
import logging
import numpy as np

ap_process_mode = "image"
ap_doplot = True
ap_image_file = "ESO479-G1_r.fits"
ap_name = "testcustomprocessing"
ap_pixscale = 0.262
ap_zeropoint = 22.5
ap_badpixel_low = 0


def mywriteoutput(IMG, results, options):
    saveto = options["ap_saveto"] if "ap_saveto" in options else "./"
    with open(os.path.join(saveto, options["ap_name"] + ".aux"), "w") as f:
        # write profile info
        f.write("written on: %s\n" % str(datetime.now()))
        f.write("name: %s\n" % str(options["ap_name"]))
        for r in sorted(results.keys()):
            if "auxfile" in r:
                f.write(results[r] + "\n")
        for k in sorted(options.keys()):
            if k == "ap_name":
                continue
        f.write("option %s: %s\n" % (k, str(options[k])))
    # Write the mask data, if provided
    if "mask" in results and (not results["mask"] is None):
        header = fits.Header()
        header["IMAGE 1"] = "mask"
        hdul = fits.HDUList(
            [fits.PrimaryHDU(header=header), fits.ImageHDU(results["mask"].astype(int))]
        )
        hdul.writeto(saveto + options["ap_name"] + "_mask.fits", overwrite=True)
        sleep(1)
        # Zip the mask file because it can be large and take a lot of memory, but in principle
        # is very easy to compress
        os.system("gzip -fq " + saveto + options["ap_name"] + "_mask.fits")

    return IMG, {}


def count_pixel_range(IMG, results, options):

    count = np.sum(
        np.logical_and(IMG > options["ap_mycountrange_low"], IMG < options["ap_mycountrange_high"])
    )

    logging.info("%s: counted %i pixels in custom range" % (options["ap_name"], count))

    return IMG, {
        "auxfile count pixels in range": "In range from %.2f to %.2f there were %i pixels"
        % (options["ap_mycountrange_low"], options["ap_mycountrange_low"], count)
    }


ap_new_pipeline_steps = {
    "head": [
        "mask badpixels",
        "background",
        "count pixel range",
        "custom writebackground",
    ]
}
ap_new_pipeline_methods = {
    "custom writebackground": mywriteoutput,
    "count pixel range": count_pixel_range,
}

# note these parameters are not standard for AutoProf, they are only used in the custom function.
# Users can create any such parameters that they like so long as the variable begins with 'ap_'
ap_mycountrange_low = 0.2
ap_mycountrange_high = 0.3
