from astropy.io import fits
from astropy.table import Table
import numpy as np
import sys
import os

from ..autoprofutils.SharedFunctions import PA_shift_convention
from datetime import datetime
from time import sleep

__all__ = ("WriteProf", )

def WriteProf(IMG, results, options):
    """Basic method to write SB profile to disk.

    This step writes the results of the AutoProf pipeline analysis to
    a file. There are two files written, a .prof file containing the
    surface brightness profile and acompanying measurements, and a
    .aux file containing global results, messages, and setting used
    for the pipeline. The .prof file looks for specific keywords in
    the results dictionary: prof header, prof units, prof data, and
    prof format. There are the results from the isophotal fitting
    step. prof header gives the column names for the profile, prof
    units is a dictionary which gives the corresponding units for each
    column header key, prof data is a dictionary containing a list of
    values for each header key, and prof format is a dictionary which
    gives the python string format for values under each header key
    (for example '%.4f' gives a number to 4 decimal places). The
    profile is written with comma (or a user specified delimiter)
    separation for each value, where each row corresponds to a given
    isophote at increasing semi-major axis values.

    The .aux file has a less strict format than the .prof file. The
    first line records the date and time that the file was written,
    the second line gives the name of the object as specified by the
    user or the filename. The next lines are taken from the results
    dictionary, any result key with auxfile in the name is taken as a
    message for the .aux file and written (in alphabetical order by
    key) to the file. See the pipeline step output formats for the
    messages that are included in the .aux file. Finally, a record of
    the user specified options is included for reference.

    Parameters
    ---------
    ap_saveto : string, default None
      Directory in which to save profile

    ap_name : string, default None
      Name of the current galaxy, used for making filenames.

    ap_delimiter : string, default ','
      Delimiter to use between entries in the profile.

    ap_profile_format : string, default 'csv'
      Type of file format to use for profile. Can choose from ['csv', 'fits']

    ap_savemask : bool, default False
      Save object mask fits file. This can create large files, depending on the size of the original image.

    Notes
    ----------
    :References:
    - 'prof header'
    - 'prof units'
    - 'prof data'
    - 'mask' (optional)

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {}

    """

    saveto = options["ap_saveto"] if "ap_saveto" in options else "./"

    # Write aux file
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

    # Write the profile
    delim = options["ap_delimiter"] if "ap_delimiter" in options else ","
    delim = options["ap_delimiter"] if "ap_delimiter" in options else ","
    try:
        results["prof data"]["pa"] = list(
            PA_shift_convention(np.array(results["prof data"]["pa"]), deg=True)
        )
    except:
        pass
    T = Table(data=results["prof data"], names=results["prof header"])
    if (
        "ap_profile_format" in options
        and options["ap_profile_format"].lower() == "fits"
    ):
        T.meta["UNITS"] = delim.join(
            results["prof units"][h] for h in results["prof header"]
        )
        T.write(
            os.path.join(saveto, options["ap_name"] + "_prof.fits"),
            format="fits",
            overwrite=True,
        )
    else:
        T.write(
            os.path.join(saveto, options["ap_name"] + ".prof"),
            format="ascii.commented_header",
            delimiter=delim,
            overwrite=True,
            comment="# "
            + delim.join(results["prof units"][h] for h in results["prof header"])
            + "\n",
        )
    try:
        results["prof data"]["pa"] = list(
            PA_shift_convention(np.array(results["prof data"]["pa"]), deg=True)
        )
    except:
        pass

    # Write the mask data, if provided
    if (
        "mask" in results
        and (not results["mask"] is None)
        and "ap_savemask" in options
        and options["ap_savemask"]
    ):
        hdul = fits.HDUList([fits.PrimaryHDU(results["mask"].astype(int))])
        hdul.writeto(saveto + options["ap_name"] + "_mask.fits", overwrite=True)
        sleep(1)
        # Zip the mask file because it can be large and take a lot of memory, but in principle
        # is very easy to compress
        os.system("gzip -fq " + saveto + options["ap_name"] + "_mask.fits")
    return IMG, {}
