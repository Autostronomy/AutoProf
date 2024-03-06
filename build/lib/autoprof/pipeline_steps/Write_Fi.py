from astropy.io import fits
import numpy as np
import sys
import os

__all__ = ("WriteFi", )

def WriteFi(IMG, results, options):
    """
    Writes the galaxy image to disk.
    """

    saveto = options["ap_saveto"] if "ap_saveto" in options else "./"
    writeas = options["ap_writeas"] if "ap_writeas" in options else "fits"

    def _iterate_filename(fi):
        """If file exists add one to the end of the existing file to avoid clobbering."""
        dir_, base = os.path.split(fi)
        if os.path.exists(fi):
            sep_base = base.split(os.path.extsep)
            base = (
                sep_base[0] + ".{:03d}".format(int(sep_base[1]) + 1) + "." + sep_base[2]
            )
            return _iterate_filename(os.path.join(dir_, base))
        else:
            return fi

    # Write npy file
    if writeas == "npy":
        fi = saveto + options["ap_name"] + ".000.npy"
        fi = _iterate_filename(fi)
        with open(fi, "wb") as f:
            np.save(f, IMG)

    # Write fits file
    else:
        fi = saveto + options["ap_name"] + ".000.fits"
        fi = _iterate_filename(fi)
        hdu = fits.PrimaryHDU(IMG)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(fi)
        hdulist.close()

    return IMG, {}
