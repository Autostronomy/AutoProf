import numpy as np
import sys
import os

from ..autoprofutils.SharedFunctions import (
    _iso_extract,
    _iso_between,
    LSBImage,
    _iso_line,
    AddLogo,
    autocmap,
    _average,
    _scatter,
    flux_to_sb,
)
from ..autoprofutils.Diagnostic_Plots import Plot_Axial_Profiles
from scipy.stats import iqr
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
import matplotlib
import logging

__all__ = ("Axial_Profiles",)

def Axial_Profiles(IMG, results, options):
    """Extracts SB profiles perpendicular to the major (or minor) axis.

    For some applications, such as examining edge on galaxies, it is
    beneficial to observe the vertical structure in a disk. This can
    be achieved with the Axial Profiles method. It will construct a
    series of lines, each one with a starting point on the major axis
    of the galaxy and radiating perpendicular from it. The location of
    these lines are, by default, geometrically spaced so that they can
    gather more light in the fainter outskirts. Along a given line,
    and SB profile is extracted, with the distance between points on
    the profile also increasing geometrically, allowing more light
    collection. The outputted profile is formatted similar to a
    regular SB profile, except that there are many SB profiles with
    each one having a corresponding distance from the center and
    quadrant of the image. A diagnostic image is generated to aid in
    identifying where each profile is extracted.
    
    Parameters
    -----------------
    ap_axialprof_pa : float, default 0
      user set position angle at which to align the axial profiles
      relative to the global position angle+90, in degrees. A common
      choice would be "90" which would then sample along the
      semi-major axis instead of the semi-minor axis.

    ap_zeropoint : float, default 22.5
      Photometric zero point

    ap_samplestyle : string, default 'geometric'
      indicate if isophote sampling radii should grow linearly or
      geometrically. Can also do geometric sampling at the center and
      linear sampling once geometric step size equals linear. Options
      are: 'linear', 'geometric', and 'geometric-linear'.

    ap_isoaverage_method : string, default 'median'
      Select the method used to compute the averafge flux along an
      isophote. Choose from 'mean', 'median', and 'mode'.  In general,
      median is fast and robust to a few outliers. Mode is slow but
      robust to more outliers. Mean is fast and accurate in low S/N
      regimes where fluxes take on near integer values, but not robust
      to outliers. The mean should be used along with a mask to remove
      spurious objects such as foreground stars or galaxies, and
      should always be used with caution.

    Notes
    ----------
    :References:
    - 'mask' (optional)
    - 'background'
    - 'psf fwhm'
    - 'center'
    - 'prof data' (optional)
    - 'init pa'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      No results provided as this method writes its own profile

      .. code-block:: python

        {}

    """

    mask = results["mask"] if "mask" in results else None
    pa = results["init pa"] + (
        (options["ap_axialprof_pa"] * np.pi / 180)
        if "ap_axialprof_pa" in options
        else 0.0
    )
    dat = IMG - results["background"]
    zeropoint = options["ap_zeropoint"] if "ap_zeropoint" in options else 22.5

    if "prof data" in results:
        Rproflim = results["prof data"]["R"][-1] / options["ap_pixscale"]
    else:
        Rproflim = min(IMG.shape) / 2

    R = [0]
    while R[-1] < Rproflim:
        if "ap_samplestyle" in options and options["ap_samplestyle"] == "linear":
            step = (
                options["ap_samplelinearscale"]
                if "ap_samplelinearscale" in options
                else 0.5 * results["psf fwhm"]
            )
        else:
            step = R[-1] * (
                options["ap_samplegeometricscale"]
                if "ap_samplegeometricscale" in options
                else 0.1
            )
        R.append(R[-1] + max(1, step))

    sb = {}
    sbE = {}
    for rd in [1, -1]:
        for ang in [1, -1]:
            key = (rd, ang)
            sb[key] = []
            sbE[key] = []
            branch_pa = (pa + ang * np.pi / 2) % (2 * np.pi)
            for pi, pR in enumerate(R):
                sb[key].append([])
                sbE[key].append([])
                width = (R[pi] - R[pi - 1]) if pi > 0 else 1.0
                flux, XX = _iso_line(
                    dat,
                    R[-1],
                    width,
                    branch_pa,
                    {
                        "x": results["center"]["x"]
                        + ang * rd * pR * np.cos(pa + (0 if ang > 0 else np.pi)),
                        "y": results["center"]["y"]
                        + ang * rd * pR * np.sin(pa + (0 if ang > 0 else np.pi)),
                    },
                )
                for oi, oR in enumerate(R):
                    length = (R[oi] - R[oi - 1]) if oi > 0 else 1.0
                    CHOOSE = np.logical_and(
                        XX > (oR - length / 2), XX < (oR + length / 2)
                    )
                    if np.sum(CHOOSE) == 0:
                        sb[key][-1].append(99.999)
                        sbE[key][-1].append(99.999)
                        continue
                    medflux = _average(
                        flux[CHOOSE],
                        options["ap_isoaverage_method"]
                        if "ap_isoaverage_method" in options
                        else "median",
                    )
                    scatflux = _scatter(
                        flux[CHOOSE],
                        options["ap_isoaverage_method"]
                        if "ap_isoaverage_method" in options
                        else "median",
                    )
                    sb[key][-1].append(
                        flux_to_sb(medflux, options["ap_pixscale"], zeropoint)
                        if medflux > 0
                        else 99.999
                    )
                    sbE[key][-1].append(
                        (
                            2.5
                            * scatflux
                            / (np.sqrt(np.sum(CHOOSE)) * medflux * np.log(10))
                        )
                        if medflux > 0
                        else 99.999
                    )

    with open(
        "%s%s_axial_profile.prof"
        % (
            (options["ap_saveto"] if "ap_saveto" in options else ""),
            options["ap_name"],
        ),
        "w",
    ) as f:
        f.write("R")
        for rd in [1, -1]:
            for ang in [1, -1]:
                for pR in R:
                    f.write(
                        ",sb[%.3f:%s90],sbE[%.3f:%s90]"
                        % (
                            rd * pR * options["ap_pixscale"],
                            "+" if ang > 0 else "-",
                            rd * pR * options["ap_pixscale"],
                            "+" if ang > 0 else "-",
                        )
                    )
        f.write("\n")
        f.write("arcsec")
        for rd in [1, -1]:
            for ang in [1, -1]:
                for pR in R:
                    f.write(",mag*arcsec^-2,mag*arcsec^-2")
        f.write("\n")
        for oi, oR in enumerate(R):
            f.write("%.4f" % (oR * options["ap_pixscale"]))
            for rd in [1, -1]:
                for ang in [1, -1]:
                    key = (rd, ang)
                    for pi, pR in enumerate(R):
                        f.write(",%.4f,%.4f" % (sb[key][pi][oi], sbE[key][pi][oi]))
            f.write("\n")

    if "ap_doplot" in options and options["ap_doplot"]:
        Plot_Axial_Profiles(dat, R, sb, sbE, pa, results, options)

    return IMG, {}
