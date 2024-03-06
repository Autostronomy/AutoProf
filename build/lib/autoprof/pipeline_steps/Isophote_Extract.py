import numpy as np
from photutils.isophote import EllipseSample, EllipseGeometry, Isophote, IsophoteList
from photutils.isophote import Ellipse as Photutils_Ellipse
from scipy.optimize import minimize
from scipy.stats import iqr
from scipy.fftpack import fft, ifft
from scipy.interpolate import UnivariateSpline
from time import time
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
from copy import copy
import logging
import sys
import os

from ..autoprofutils.SharedFunctions import (
    _x_to_pa,
    _x_to_eps,
    _inv_x_to_eps,
    _inv_x_to_pa,
    SBprof_to_COG_errorprop,
    _iso_extract,
    _iso_between,
    LSBImage,
    AddLogo,
    _average,
    _scatter,
    flux_to_sb,
    flux_to_mag,
    PA_shift_convention,
    autocolours,
    fluxdens_to_fluxsum_errorprop,
    Fmode_fluxdens_to_fluxsum_errorprop,
    mag_to_flux,
)
from ..autoprofutils.Diagnostic_Plots import (
    Plot_SB_Profile,
    Plot_I_Profile,
    Plot_Phase_Profile,
)

__all__ = ("Isophote_Extract_Forced", "Isophote_Extract", "Isophote_Extract_Photutils")

def _Generate_Profile(IMG, results, R, parameters, options):

    # Create image array with background and mask applied
    try:
        if np.any(results["mask"]):
            mask = results["mask"]
        else:
            mask = None
    except:
        mask = None
    dat = IMG - results["background"]
    zeropoint = options["ap_zeropoint"] if "ap_zeropoint" in options else 22.5
    fluxunits = options["ap_fluxunits"] if "ap_fluxunits" in options else "mag"

    for p in range(len(parameters)):
        # Indicate no Fourier modes if supplied parameters does not include it
        if not "m" in parameters[p]:
            parameters[p]["m"] = None
        if not "C" in parameters[p]:
            parameters[p]["C"] = None
        # If no ellipticity error supplied, assume zero
        if not "ellip err" in parameters[p]:
            parameters[p]["ellip err"] = 0.0
        # If no position angle error supplied, assume zero
        if not "pa err" in parameters[p]:
            parameters[p]["pa err"] = 0.0

    sb = []
    sbE = []
    pixels = []
    maskedpixels = []
    cogdirect = []
    sbfix = []
    sbfixE = []
    measFmodes = []

    count_neg = 0
    medflux = np.inf
    end_prof = len(R)
    compare_interp = []
    for i in range(len(R)):
        if "ap_isoband_fixed" in options and options["ap_isoband_fixed"]:
            isobandwidth = (
                options["ap_isoband_width"] if "ap_isoband_width" in options else 0.5
            )
        else:
            isobandwidth = R[i] * (
                options["ap_isoband_width"] if "ap_isoband_width" in options else 0.025
            )
        isisophoteband = False
        if (
            medflux
            > (
                results["background noise"]
                * (options["ap_isoband_start"] if "ap_isoband_start" in options else 2)
            )
            or isobandwidth < 0.5
        ):
            isovals = _iso_extract(
                dat,
                R[i],
                parameters[i],
                results["center"],
                mask=mask,
                more=True,
                rad_interp=(
                    options["ap_iso_interpolate_start"]
                    if "ap_iso_interpolate_start" in options
                    else 5
                )
                * results["psf fwhm"],
                interp_method=(
                    options["ap_iso_interpolate_method"]
                    if "ap_iso_interpolate_method" in options
                    else "lanczos"
                ),
                interp_window=(
                    int(options["ap_iso_interpolate_window"])
                    if "ap_iso_interpolate_window" in options
                    else 5
                ),
                sigmaclip=options["ap_isoclip"] if "ap_isoclip" in options else False,
                sclip_iterations=options["ap_isoclip_iterations"]
                if "ap_isoclip_iterations" in options
                else 10,
                sclip_nsigma=options["ap_isoclip_nsigma"]
                if "ap_isoclip_nsigma" in options
                else 5,
            )
        else:
            isisophoteband = True
            isovals = _iso_between(
                dat,
                R[i] - isobandwidth,
                R[i] + isobandwidth,
                parameters[i],
                results["center"],
                mask=mask,
                more=True,
                sigmaclip=options["ap_isoclip"] if "ap_isoclip" in options else False,
                sclip_iterations=options["ap_isoclip_iterations"]
                if "ap_isoclip_iterations" in options
                else 10,
                sclip_nsigma=options["ap_isoclip_nsigma"]
                if "ap_isoclip_nsigma" in options
                else 5,
            )
        isotot = np.sum(
            _iso_between(dat, 0, R[i], parameters[i], results["center"], mask=mask)
        )
        medflux = _average(
            isovals[0],
            options["ap_isoaverage_method"]
            if "ap_isoaverage_method" in options
            else "median",
        )
        scatflux = _scatter(
            isovals[0],
            options["ap_isoaverage_method"]
            if "ap_isoaverage_method" in options
            else "median",
        )
        if (
            "ap_iso_measurecoefs" in options
            and not options["ap_iso_measurecoefs"] is None
        ):
            if (
                mask is None
                and (not "ap_isoclip" in options or not options["ap_isoclip"])
                and not isisophoteband
            ):
                coefs = fft(isovals[0])
            else:
                N = max(15, int(0.9 * 2 * np.pi * R[i]))
                theta = np.linspace(0, 2 * np.pi * (1.0 - 1.0 / N), N)
                coefs = fft(np.interp(theta, isovals[1], isovals[0], period=2 * np.pi))
            measFmodes.append(
                {
                    "a": [np.imag(coefs[0]) / len(coefs)]
                    + list(
                        np.imag(coefs[np.array(options["ap_iso_measurecoefs"])])
                        / (np.abs(coefs[0]))
                    ),
                    "b": [np.real(coefs[0]) / len(coefs)]
                    + list(
                        np.real(coefs[np.array(options["ap_iso_measurecoefs"])])
                        / (np.abs(coefs[0]))
                    ),
                }
            )

        pixels.append(len(isovals[0]))
        maskedpixels.append(isovals[2])
        if fluxunits == "intensity":
            sb.append(medflux / options["ap_pixscale"] ** 2)
            sbE.append(scatflux / np.sqrt(len(isovals[0])))
            cogdirect.append(isotot)
        else:
            sb.append(
                flux_to_sb(medflux, options["ap_pixscale"], zeropoint)
                if medflux > 0
                else 99.999
            )
            sbE.append(
                (2.5 * scatflux / (np.sqrt(len(isovals[0])) * medflux * np.log(10)))
                if medflux > 0
                else 99.999
            )
            cogdirect.append(flux_to_mag(isotot, zeropoint) if isotot > 0 else 99.999)
        if medflux <= 0:
            count_neg += 1
        if (
            "ap_truncate_evaluation" in options
            and options["ap_truncate_evaluation"]
            and count_neg >= 2
        ):
            end_prof = i + 1
            break

    # Compute Curve of Growth from SB profile
    if fluxunits == "intensity":
        cog, cogE = Fmode_fluxdens_to_fluxsum_errorprop(
            R[:end_prof] * options["ap_pixscale"],
            np.array(sb),
            np.array(sbE),
            parameters[:end_prof],
            N=100,
            symmetric_error=True,
        )

        if cog is None:
            cog = -99.999 * np.ones(len(R))
            cogE = -99.999 * np.ones(len(R))
        else:
            cog[np.logical_not(np.isfinite(cog))] = -99.999
            cogE[cog < 0] = -99.999
    else:
        cog, cogE = SBprof_to_COG_errorprop(
            R[:end_prof] * options["ap_pixscale"],
            np.array(sb),
            np.array(sbE),
            parameters[:end_prof],
            N=100,
            symmetric_error=True,
        )
        if cog is None:
            cog = 99.999 * np.ones(len(R))
            cogE = 99.999 * np.ones(len(R))
        else:
            cog[np.logical_not(np.isfinite(cog))] = 99.999
            cogE[cog > 99] = 99.999

    # For each radius evaluation, write the profile parameters
    if fluxunits == "intensity":
        params = [
            "R",
            "I",
            "I_e",
            "totflux",
            "totflux_e",
            "ellip",
            "ellip_e",
            "pa",
            "pa_e",
            "pixels",
            "maskedpixels",
            "totflux_direct",
        ]

        SBprof_units = {
            "R": "arcsec",
            "I": "flux*arcsec^-2",
            "I_e": "flux*arcsec^-2",
            "totflux": "flux",
            "totflux_e": "flux",
            "ellip": "unitless",
            "ellip_e": "unitless",
            "pa": "deg",
            "pa_e": "deg",
            "pixels": "count",
            "maskedpixels": "count",
            "totflux_direct": "flux",
        }
    else:
        params = [
            "R",
            "SB",
            "SB_e",
            "totmag",
            "totmag_e",
            "ellip",
            "ellip_e",
            "pa",
            "pa_e",
            "pixels",
            "maskedpixels",
            "totmag_direct",
        ]

        SBprof_units = {
            "R": "arcsec",
            "SB": "mag*arcsec^-2",
            "SB_e": "mag*arcsec^-2",
            "totmag": "mag",
            "totmag_e": "mag",
            "ellip": "unitless",
            "ellip_e": "unitless",
            "pa": "deg",
            "pa_e": "deg",
            "pixels": "count",
            "maskedpixels": "count",
            "totmag_direct": "mag",
        }

    SBprof_data = dict((h, None) for h in params)
    SBprof_data["R"] = list(R[:end_prof] * options["ap_pixscale"])
    SBprof_data["I" if fluxunits == "intensity" else "SB"] = list(sb)
    SBprof_data["I_e" if fluxunits == "intensity" else "SB_e"] = list(sbE)
    SBprof_data["totflux" if fluxunits == "intensity" else "totmag"] = list(cog)
    SBprof_data["totflux_e" if fluxunits == "intensity" else "totmag_e"] = list(cogE)
    SBprof_data["ellip"] = list(parameters[p]["ellip"] for p in range(end_prof))
    SBprof_data["ellip_e"] = list(parameters[p]["ellip err"] for p in range(end_prof))
    SBprof_data["pa"] = list(parameters[p]["pa"] * 180 / np.pi for p in range(end_prof))
    SBprof_data["pa_e"] = list(
        parameters[p]["pa err"] * 180 / np.pi for p in range(end_prof)
    )
    SBprof_data["pixels"] = list(pixels)
    SBprof_data["maskedpixels"] = list(maskedpixels)
    SBprof_data[
        "totflux_direct" if fluxunits == "intensity" else "totmag_direct"
    ] = list(cogdirect)

    if "ap_iso_measurecoefs" in options and not options["ap_iso_measurecoefs"] is None:
        whichcoefs = [0] + list(options["ap_iso_measurecoefs"])
        for i in list(range(len(whichcoefs))):
            aa, bb = "a%i" % whichcoefs[i], "b%i" % whichcoefs[i]
            params += [aa, bb]
            SBprof_units.update(
                {
                    aa: "flux" if whichcoefs[i] == 0 else "a%i/F0" % whichcoefs[i],
                    bb: "flux" if whichcoefs[i] == 0 else "b%i/F0" % whichcoefs[i],
                }
            )
            SBprof_data[aa] = list(F["a"][i] for F in measFmodes)
            SBprof_data[bb] = list(F["b"][i] for F in measFmodes)

    if any(not p["m"] is None for p in parameters):
        for m in range(len(parameters[0]["m"])):
            AA, PP = "A%i" % parameters[0]["m"][m], "Phi%i" % parameters[0]["m"][m]
            params += [AA, PP]
            SBprof_units.update({AA: "unitless", PP: "deg"})
            SBprof_data[AA] = list(p["Am"][m] for p in parameters[:end_prof])
            SBprof_data[PP] = list(p["Phim"][m] for p in parameters[:end_prof])
    if any(not p["C"] is None for p in parameters):
        params += ["C"]
        SBprof_units["C"] = "unitless"
        SBprof_data["C"] = list(p["C"] for p in parameters[:end_prof])

    if "ap_doplot" in options and options["ap_doplot"]:
        Plot_Phase_Profile(
            np.array(SBprof_data["R"]), parameters[:end_prof], results, options
        )
        if fluxunits == "intensity":
            Plot_I_Profile(
                dat,
                np.array(SBprof_data["R"]),
                np.array(SBprof_data["I"]),
                np.array(SBprof_data["I_e"]),
                parameters[:end_prof],
                results,
                options,
            )
        else:
            Plot_SB_Profile(
                dat,
                np.array(SBprof_data["R"]),
                np.array(SBprof_data["SB"]),
                np.array(SBprof_data["SB_e"]),
                parameters[:end_prof],
                results,
                options,
            )

    return {"prof header": params, "prof units": SBprof_units, "prof data": SBprof_data}


def Isophote_Extract_Forced(IMG, results, options):
    """Method for extracting SB profiles that have been set by forced photometry.

    This is nearly identical to the general isophote extraction
    method, except that it does not choose which radii to sample the
    profile, instead it takes the radii, PA, and ellipticities as
    given.

    Parameters
    -----------------
    ap_zeropoint : float, default 22.5
      Photometric zero point. For converting flux to mag units.

    ap_fluxunits : str, default "mag"
      units for outputted photometry. Can either be "mag" for log
      units, or "intensity" for linear units.
    
    ap_isoband_start : float, default 2
      The noise level at which to begin sampling a band of pixels to
      compute SB instead of sampling a line of pixels near the
      isophote in units of pixel flux noise. Will never initiate band
      averaging if the band width is less than half a pixel

    ap_isoband_width : float, default 0.025
      The relative size of the isophote bands to sample. flux values
      will be sampled at +- *ap_isoband_width* \*R for each radius.

    ap_isoband_fixed : bool, default False
      Use a fixed width for the size of the isobands, the width is set
      by *ap_isoband_width* which now has units of pixels, the default
      is 0.5 such that the full band has a width of 1 pixel.

    ap_truncate_evaluation : bool, default False
      Stop evaluating new isophotes once two negative flux isophotes
      have been recorded, presumed to have reached the end of the
      profile.

    ap_iso_interpolate_start : float, default 5
      Use a Lanczos interpolation for isophotes with semi-major axis
      less than this number times the PSF.

    ap_iso_interpolate_method : string, default 'lanczos'
      Select method for flux interpolation on image, options are
      'lanczos' and 'bicubic'. Default is 'lanczos' with a window size
      of 3.

    ap_iso_interpolate_window : int, default 3
      Window size for Lanczos interpolation, default is 3, meaning 3
      pixels on either side of the sample point are used for
      interpolation.

    ap_isoaverage_method : string, default 'median'
      Select the method used to compute the averafge flux along an
      isophote. Choose from 'mean', 'median', and 'mode'.  In general,
      median is fast and robust to a few outliers. Mode is slow but
      robust to more outliers. Mean is fast and accurate in low S/N
      regimes where fluxes take on near integer values, but not robust
      to outliers. The mean should be used along with a mask to remove
      spurious objects such as foreground stars or galaxies, and
      should always be used with caution.

    ap_isoclip : bool, default False
      Perform sigma clipping along extracted isophotes. Removes flux
      samples from an isophote that deviate significantly from the
      median. Several iterations of sigma clipping are performed until
      convergence or *ap_isoclip_iterations* iterations are
      reached. Sigma clipping is a useful substitute for masking
      objects, though careful masking is better. Also an aggressive
      sigma clip may bias results.

    ap_isoclip_iterations : int, default None
      Maximum number of sigma clipping iterations to perform. The
      default is infinity, so the sigma clipping procedure repeats
      until convergence

    ap_isoclip_nsigma : float, default 5
      Number of sigma above median to apply clipping. All values above
      (median + *ap_isoclip_nsigma* x sigma) are removed from the
      isophote.

    ap_iso_measurecoefs : tuple, default None
      tuple indicating which fourier modes to extract along fitted
      isophotes. Most common is (4,), which identifies boxy/disky
      isophotes. Also common is (1,3), which identifies lopsided
      galaxies. The outputted values are computed as a_i =
      imag(F_i)/abs(F_0) and b_i = real(F_i)/abs(F_0) where F_i is a
      fourier coefficient. Not activated by default as it adds to
      computation time.

    ap_plot_sbprof_ylim : tuple, default None
      Tuple with axes limits for the y-axis in the SB profile
      diagnostic plot. Be careful when using intensity units
      since this will change the ideal axis limits.
    
    ap_plot_sbprof_xlim : tuple, default None
      Tuple with axes limits for the x-axis in the SB profile
      diagnostic plot.

    ap_plot_sbprof_set_errscale : float, default None
      Float value by which to scale errorbars on the SB profile
      this makes them more visible in cases where the statistical
      errors are very small.
    
    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'psf fwhm'
    - 'center'
    - 'init ellip'
    - 'init pa'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'prof header': , # List object with strings giving the items in the header of the final SB profile (list)
         'prof units': , # dict object that links header strings to units (given as strings) for each variable (dict)
         'prof data': # dict object linking header strings to list objects containing the rows for a given variable (dict)

        }

    """

    with open(options["ap_forcing_profile"], "r") as f:
        raw = f.readlines()
        for i, l in enumerate(raw):
            if l[0] != "#":
                readfrom = i
                break
        header = list(h.strip() for h in raw[readfrom].split(","))
        force = dict((h, []) for h in header)
        for l in raw[readfrom + 2 :]:
            for d, h in zip(l.split(","), header):
                force[h].append(float(d.strip()))

    force["pa"] = PA_shift_convention(np.array(force["pa"]), deg=True) * np.pi / 180

    parameters = list(
        {
            "ellip": force["ellip"][i],
            "pa": (
                force["pa"][i]
                + (
                    options["ap_forced_pa_shift"]
                    if "ap_forced_pa_shift" in options
                    else 0.0
                )
            )
            % np.pi,
        }
        for i in range(len(force["R"]))
    )
    for i in range(len(force["R"])):
        if "ellip_e" in force and "pa_e" in force:
            parameters[i]["ellip_err"] = force["ellip_e"][i]
            parameters[i]["pa_err"] = force["pa_e"][i] * np.pi / 180
        else:
            parameters[i]["pa_err"] = 0.0
            parameters[i]["ellip_err"] = 0.0

    return IMG, _Generate_Profile(
        IMG, results, np.array(force["R"]) / options["ap_pixscale"], parameters, options
    )


def Isophote_Extract(IMG, results, options):
    """General method for extracting SB profiles.

    The default SB profile extraction method is highly
    flexible, allowing users to test a variety of techniques on their data
    to determine the most robust. The user may specify a variety of
    sampling arguments for the photometry extraction.  For example, a
    start or end radius in pixels, or whether to sample geometrically or
    linearly in radius.  Geometric sampling is the default as it is
    faster.  Once the sampling profile of semi-major axis values has been
    chosen, the function interpolates (spline) the position angle and
    ellipticity profiles at the requested values.  For any sampling beyond
    the outer radius from the *Isophotal Fitting* step, a constant value
    is used.  Within 1 PSF, a circular isophote is used.

    Parameters
    -----------------
    ap_zeropoint : float, default 22.5
      Photometric zero point. For converting flux to mag units.

    ap_fluxunits : str, default "mag"
      units for outputted photometry. Can either be "mag" for log
      units, or "intensity" for linear units.
    
    ap_samplegeometricscale : float, default 0.1
      growth scale for isophotes when sampling for the final output
      profile.  Used when sampling geometrically. By default, each
      isophote is 10\% further than the last.

    ap_samplelinearscale : float, default None
      growth scale (in pixels) for isophotes when sampling for the
      final output profile. Used when sampling linearly. Default is 1
      PSF length.

    ap_samplestyle : string, default 'geometric'
      indicate if isophote sampling radii should grow linearly or
      geometrically. Can also do geometric sampling at the center and
      linear sampling once geometric step size equals linear. Options
      are: 'linear', 'geometric', 'geometric-linear'

    ap_sampleinitR : float, default None
      Starting radius (in pixels) for isophote sampling from the
      image. Note that a starting radius of zero is not
      advised. Default is 1 pixel or 1PSF, whichever is smaller.

    ap_sampleendR : float, default None
      End radius (in pixels) for isophote sampling from the
      image. Default is 3 times the fit radius, also see
      *ap_extractfull*.

    ap_isoband_start : float, default 2
      The noise level at which to begin sampling a band of pixels to
      compute SB instead of sampling a line of pixels near the
      isophote in units of pixel flux noise. Will never initiate band
      averaging if the band width is less than half a pixel

    ap_isoband_width : float, default 0.025
      The relative size of the isophote bands to sample. flux values
      will be sampled at +- *ap_isoband_width* \*R for each radius.

    ap_isoband_fixed : bool, default False
      Use a fixed width for the size of the isobands, the width is set
      by *ap_isoband_width* which now has units of pixels, the default
      is 0.5 such that the full band has a width of 1 pixel.

    ap_truncate_evaluation : bool, default False
      Stop evaluating new isophotes once two negative flux isophotes
      have been recorded, presumed to have reached the end of the
      profile.

    ap_extractfull : bool, default False
      Tells AutoProf to extend the isophotal solution to the edge of
      the image. Will be overridden by *ap_truncate_evaluation*.

    ap_iso_interpolate_start : float, default 5
      Use a Lanczos interpolation for isophotes with semi-major axis
      less than this number times the PSF.

    ap_iso_interpolate_method : string, default 'lanczos'
      Select method for flux interpolation on image, options are
      'lanczos' and 'bicubic'. Default is 'lanczos' with a window size
      of 3.

    ap_iso_interpolate_window : int, default 3
      Window size for Lanczos interpolation, default is 3, meaning 3
      pixels on either side of the sample point are used for
      interpolation.

    ap_isoaverage_method : string, default 'median'
      Select the method used to compute the averafge flux along an
      isophote. Choose from 'mean', 'median', and 'mode'.  In general,
      median is fast and robust to a few outliers. Mode is slow but
      robust to more outliers. Mean is fast and accurate in low S/N
      regimes where fluxes take on near integer values, but not robust
      to outliers. The mean should be used along with a mask to remove
      spurious objects such as foreground stars or galaxies, and
      should always be used with caution.

    ap_isoclip : bool, default False
      Perform sigma clipping along extracted isophotes. Removes flux
      samples from an isophote that deviate significantly from the
      median. Several iterations of sigma clipping are performed until
      convergence or *ap_isoclip_iterations* iterations are
      reached. Sigma clipping is a useful substitute for masking
      objects, though careful masking is better. Also an aggressive
      sigma clip may bias results.

    ap_isoclip_iterations : int, default None
      Maximum number of sigma clipping iterations to perform. The
      default is infinity, so the sigma clipping procedure repeats
      until convergence

    ap_isoclip_nsigma : float, default 5
      Number of sigma above median to apply clipping. All values above
      (median + *ap_isoclip_nsigma* x sigma) are removed from the
      isophote.

    ap_iso_measurecoefs : tuple, default None
      tuple indicating which fourier modes to extract along fitted
      isophotes. Most common is (4,), which identifies boxy/disky
      isophotes. Also common is (1,3), which identifies lopsided
      galaxies. The outputted values are computed as a_i =
      imag(F_i)/abs(F_0) and b_i = real(F_i)/abs(F_0) where F_i is a
      fourier coefficient. Not activated by default as it adds to
      computation time.

    ap_plot_sbprof_ylim : tuple, default None
      Tuple with axes limits for the y-axis in the SB profile
      diagnostic plot. Be careful when using intensity units
      since this will change the ideal axis limits.
    
    ap_plot_sbprof_xlim : tuple, default None
      Tuple with axes limits for the x-axis in the SB profile
      diagnostic plot.
    
    ap_plot_sbprof_set_errscale : float, default None
      Float value by which to scale errorbars on the SB profile
      this makes them more visible in cases where the statistical
      errors are very small.
    
    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'psf fwhm'
    - 'center'
    - 'init ellip'
    - 'init pa'
    - 'fit R'
    - 'fit ellip'
    - 'fit pa'
    - 'fit ellip_err' (optional)
    - 'fit pa_err' (optional)

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'prof header': , # List object with strings giving the items in the header of the final SB profile (list)
         'prof units': , # dict object that links header strings to units (given as strings) for each variable (dict)
         'prof data': # dict object linking header strings to list objects containing the rows for a given variable (dict)

        }

    """
    use_center = results["center"]

    # Radius values to evaluate isophotes
    R = [
        options["ap_sampleinitR"]
        if "ap_sampleinitR" in options
        else min(1.0, results["psf fwhm"] / 2)
    ]
    while (
        (
            (R[-1] < options["ap_sampleendR"] if "ap_sampleendR" in options else True)
            and R[-1] < 3 * results["fit R"][-1]
        )
        or (options["ap_extractfull"] if "ap_extractfull" in options else False)
    ) and R[-1] < max(IMG.shape) / np.sqrt(2):
        if (
            "ap_samplestyle" in options
            and options["ap_samplestyle"] == "geometric-linear"
        ):
            if len(R) > 1 and abs(R[-1] - R[-2]) >= (
                options["ap_samplelinearscale"]
                if "ap_samplelinearscale" in options
                else 3 * results["psf fwhm"]
            ):
                R.append(
                    R[-1]
                    + (
                        options["ap_samplelinearscale"]
                        if "ap_samplelinearscale" in options
                        else results["psf fwhm"] / 2
                    )
                )
            else:
                R.append(
                    R[-1]
                    * (
                        1.0
                        + (
                            options["ap_samplegeometricscale"]
                            if "ap_samplegeometricscale" in options
                            else 0.1
                        )
                    )
                )
        elif "ap_samplestyle" in options and options["ap_samplestyle"] == "linear":
            R.append(
                R[-1]
                + (
                    options["ap_samplelinearscale"]
                    if "ap_samplelinearscale" in options
                    else 0.5 * results["psf fwhm"]
                )
            )
        else:
            R.append(
                R[-1]
                * (
                    1.0
                    + (
                        options["ap_samplegeometricscale"]
                        if "ap_samplegeometricscale" in options
                        else 0.1
                    )
                )
            )
    R = np.array(R)
    logging.info(
        "%s: R complete in range [%.1f,%.1f]" % (options["ap_name"], R[0], R[-1])
    )

    # Interpolate profile values, when extrapolating just take last point
    tmp_pa_s = UnivariateSpline(
        results["fit R"], np.sin(2 * results["fit pa"]), ext=3, s=0
    )(R)
    tmp_pa_c = UnivariateSpline(
        results["fit R"], np.cos(2 * results["fit pa"]), ext=3, s=0
    )(R)
    E = _x_to_eps(
        UnivariateSpline(
            results["fit R"], _inv_x_to_eps(results["fit ellip"]), ext=3, s=0
        )(R)
    )
    # np.arctan(tmp_pa_s / tmp_pa_c) + (np.pi * (tmp_pa_c < 0))
    PA = _x_to_pa(((np.arctan2(tmp_pa_s, tmp_pa_c)) % (2 * np.pi)) / 2) 
    parameters = list({"ellip": E[i], "pa": PA[i]} for i in range(len(R)))

    if "fit Fmodes" in results:
        for i in range(len(R)):
            parameters[i]["m"] = results["fit Fmodes"]
            parameters[i]["Am"] = np.array(
                list(
                    UnivariateSpline(
                        results["fit R"],
                        results["fit Fmode A%i" % results["fit Fmodes"][m]],
                        ext=3,
                        s=0,
                    )(R[i])
                    for m in range(len(results["fit Fmodes"]))
                )
            )
            parameters[i]["Phim"] = np.array(
                list(
                    UnivariateSpline(
                        results["fit R"],
                        results["fit Fmode Phi%i" % results["fit Fmodes"][m]],
                        ext=3,
                        s=0,
                    )(R[i])
                    for m in range(len(results["fit Fmodes"]))
                )
            )

    if "fit C" in results:
        CC = UnivariateSpline(results["fit R"], results["fit C"], ext=3, s=0)(R)
        for i in range(len(R)):
            parameters[i]["C"] = CC[i]

    # Get errors for pa and ellip
    for i in range(len(R)):
        if (
            "fit ellip_err" in results
            and (not results["fit ellip_err"] is None)
            and "fit pa_err" in results
            and (not results["fit pa_err"] is None)
        ):
            parameters[i]["ellip err"] = np.clip(
                UnivariateSpline(
                    results["fit R"], results["fit ellip_err"], ext=3, s=0
                )(R[i]),
                a_min=1e-3,
                a_max=None,
            )
            parameters[i]["pa err"] = np.clip(
                UnivariateSpline(results["fit R"], results["fit pa_err"], ext=3, s=0)(
                    R[i]
                ),
                a_min=1e-3,
                a_max=None,
            )
        else:
            parameters[i]["ellip err"] = 0.0
            parameters[i]["pa err"] = 0.0

    return IMG, _Generate_Profile(IMG, results, R, parameters, options)


def Isophote_Extract_Photutils(IMG, results, options):
    """Wrapper of photutils method for extracting SB profiles.

    This simply gives users access to the photutils isophote
    extraction methods. The one exception is that SB values are taken
    as the median instead of the mean, as recomended in the photutils
    documentation. See: `photutils
    <https://photutils.readthedocs.io/en/stable/isophote.html>`_ for
    more information.

    Parameters
    ----------
    ap_zeropoint : float, default 22.5
      Photometric zero point. For converting flux to mag units.

    ap_fluxunits : str, default "mag"
      units for outputted photometry. Can either be "mag" for log
      units, or "intensity" for linear units.
    
    ap_plot_sbprof_ylim : tuple, default None
      Tuple with axes limits for the y-axis in the SB profile
      diagnostic plot. Be careful when using intensity units
      since this will change the ideal axis limits.
    
    ap_plot_sbprof_xlim : tuple, default None
      Tuple with axes limits for the x-axis in the SB profile
      diagnostic plot.
    
    ap_plot_sbprof_set_errscale : float, default None
      Float value by which to scale errorbars on the SB profile
      this makes them more visible in cases where the statistical
      errors are very small.
    
    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'psf fwhm'
    - 'center'
    - 'init R' (optional)
    - 'init ellip' (optional)
    - 'init pa' (optional)
    - 'fit R' (optional)
    - 'fit ellip' (optional)
    - 'fit pa' (optional)
    - 'fit photutils isolist' (optional)

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'prof header': , # List object with strings giving the items in the header of the final SB profile (list)
         'prof units': , # dict object that links header strings to units (given as strings) for each variable (dict)
         'prof data': # dict object linking header strings to list objects containing the rows for a given variable (dict)

        }

    """

    zeropoint = options["ap_zeropoint"] if "ap_zeropoint" in options else 22.5
    fluxunits = options["ap_fluxunits"] if "ap_fluxunits" in options else "mag"

    if fluxunits == "intensity":
        params = [
            "R",
            "I",
            "I_e",
            "totflux",
            "totflux_e",
            "ellip",
            "ellip_e",
            "pa",
            "pa_e",
            "a3",
            "a3_e",
            "b3",
            "b3_e",
            "a4",
            "a4_e",
            "b4",
            "b4_e",
        ]
        SBprof_units = {
            "R": "arcsec",
            "I": "flux*arcsec^-2",
            "I_e": "flux*arcsec^-2",
            "totflux": "flux",
            "totflux_e": "flux",
            "ellip": "unitless",
            "ellip_e": "unitless",
            "pa": "deg",
            "pa_e": "deg",
            "a3": "unitless",
            "a3_e": "unitless",
            "b3": "unitless",
            "b3_e": "unitless",
            "a4": "unitless",
            "a4_e": "unitless",
            "b4": "unitless",
            "b4_e": "unitless",
        }
    else:
        params = [
            "R",
            "SB",
            "SB_e",
            "totmag",
            "totmag_e",
            "ellip",
            "ellip_e",
            "pa",
            "pa_e",
            "a3",
            "a3_e",
            "b3",
            "b3_e",
            "a4",
            "a4_e",
            "b4",
            "b4_e",
        ]
        SBprof_units = {
            "R": "arcsec",
            "SB": "mag*arcsec^-2",
            "SB_e": "mag*arcsec^-2",
            "totmag": "mag",
            "totmag_e": "mag",
            "ellip": "unitless",
            "ellip_e": "unitless",
            "pa": "deg",
            "pa_e": "deg",
            "a3": "unitless",
            "a3_e": "unitless",
            "b3": "unitless",
            "b3_e": "unitless",
            "a4": "unitless",
            "a4_e": "unitless",
            "b4": "unitless",
            "b4_e": "unitless",
        }
    SBprof_data = dict((h, []) for h in params)
    res = {}
    dat = IMG - results["background"]
    if not "fit R" in results and not "fit photutils isolist" in results:
        logging.info(
            "%s: photutils fitting and extracting image data" % options["ap_name"]
        )
        geo = EllipseGeometry(
            x0=results["center"]["x"],
            y0=results["center"]["y"],
            sma=results["init R"] / 2,
            eps=results["init ellip"],
            pa=results["init pa"],
        )
        ellipse = Photutils_Ellipse(dat, geometry=geo)

        isolist = ellipse.fit_image(fix_center=True, linear=False)
        res.update(
            {
                "fit photutils isolist": isolist,
                "auxfile fitlimit": "fit limit semi-major axis: %.2f pix"
                % isolist.sma[-1],
            }
        )
    elif not "fit photutils isolist" in results:
        logging.info("%s: photutils extracting image data" % options["ap_name"])
        list_iso = []
        for i in range(len(results["fit R"])):
            if results["fit R"][i] <= 0:
                continue
            # Container for ellipse geometry
            geo = EllipseGeometry(
                sma=results["fit R"][i],
                x0=results["center"]["x"],
                y0=results["center"]["y"],
                eps=results["fit ellip"][i],
                pa=results["fit pa"][i],
            )
            # Extract the isophote information
            ES = EllipseSample(dat, sma=results["fit R"][i], geometry=geo)
            ES.update(fixed_parameters=None)
            list_iso.append(Isophote(ES, niter=30, valid=True, stop_code=0))

        isolist = IsophoteList(list_iso)
        res.update(
            {
                "fit photutils isolist": isolist,
                "auxfile fitlimit": "fit limit semi-major axis: %.2f pix"
                % isolist.sma[-1],
            }
        )
    else:
        isolist = results["fit photutils isolist"]

    for i in range(len(isolist.sma)):
        SBprof_data["R"].append(isolist.sma[i] * options["ap_pixscale"])
        if fluxunits == "intensity":
            SBprof_data["I"].append(
                np.median(isolist.sample[i].values[2]) / options["ap_pixscale"] ** 2
            )
            SBprof_data["I_e"].append(isolist.int_err[i])
            SBprof_data["totflux"].append(isolist.tflux_e[i])
            SBprof_data["totflux_e"].append(isolist.rms[i] / np.sqrt(isolist.npix_e[i]))
        else:
            SBprof_data["SB"].append(
                flux_to_sb(
                    np.median(isolist.sample[i].values[2]),
                    options["ap_pixscale"],
                    zeropoint,
                )
            )
            SBprof_data["SB_e"].append(
                2.5 * isolist.int_err[i] / (isolist.intens[i] * np.log(10))
            )
            SBprof_data["totmag"].append(flux_to_mag(isolist.tflux_e[i], zeropoint))
            SBprof_data["totmag_e"].append(
                2.5
                * isolist.rms[i]
                / (np.sqrt(isolist.npix_e[i]) * isolist.tflux_e[i] * np.log(10))
            )
        SBprof_data["ellip"].append(isolist.eps[i])
        SBprof_data["ellip_e"].append(isolist.ellip_err[i])
        SBprof_data["pa"].append(isolist.pa[i] * 180 / np.pi)
        SBprof_data["pa_e"].append(isolist.pa_err[i] * 180 / np.pi)
        SBprof_data["a3"].append(isolist.a3[i])
        SBprof_data["a3_e"].append(isolist.a3_err[i])
        SBprof_data["b3"].append(isolist.b3[i])
        SBprof_data["b3_e"].append(isolist.b3_err[i])
        SBprof_data["a4"].append(isolist.a4[i])
        SBprof_data["a4_e"].append(isolist.a4_err[i])
        SBprof_data["b4"].append(isolist.b4[i])
        SBprof_data["b4_e"].append(isolist.b4_err[i])
        for k in SBprof_data.keys():
            if not np.isfinite(SBprof_data[k][-1]):
                SBprof_data[k][-1] = 99.999
    res.update(
        {"prof header": params, "prof units": SBprof_units, "prof data": SBprof_data}
    )

    if "ap_doplot" in options and options["ap_doplot"]:
        if fluxunits == "intensity":
            Plot_I_Profile(
                dat,
                np.array(SBprof_data["R"]),
                np.array(SBprof_data["I"]),
                np.array(SBprof_data["I_e"]),
                np.array(SBprof_data["ellip"]),
                np.array(SBprof_data["pa"]),
                results,
                options,
            )
        else:
            Plot_SB_Profile(
                dat,
                np.array(SBprof_data["R"]),
                np.array(SBprof_data["SB"]),
                np.array(SBprof_data["SB_e"]),
                np.array(SBprof_data["ellip"]),
                np.array(SBprof_data["pa"]),
                results,
                options,
            )

    return IMG, res
