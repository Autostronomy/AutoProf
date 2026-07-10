import numpy as np
from photutils.isophote import EllipseSample, EllipseGeometry, Isophote, IsophoteList
from photutils.isophote import Ellipse as Photutils_Ellipse
from scipy.optimize import minimize
from scipy.stats import iqr
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
    _iso_extract_with_interp_cutoff,
    _iso_between,
    _iso_interpolate_radius,
    _resolve_isoextract_interp_method,
    _validate_interpolate_method,
    _photutils_masked_data,
    LSBImage,
    AddLogo,
    _average,
    _scatter,
    flux_to_sb,
    sb_to_flux,
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


def _finite_median(values):
    values = np.ma.asarray(values).compressed()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return np.median(values)


def _finite_divide(numerator, denominator):
    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or denominator == 0
    ):
        return np.nan
    return numerator / denominator


def _isoband_flux_threshold(results, options, zeropoint):
    if "ap_isoband_start_sb" in options and not options["ap_isoband_start_sb"] is None:
        return sb_to_flux(
            options["ap_isoband_start_sb"],
            options["ap_pixscale"],
            zeropoint,
        )
    return results["background noise"] * (
        options["ap_isoband_start"] if "ap_isoband_start" in options else 2
    )


def _normalize_sampling_method(method):
    method = str(method).strip()
    if method in ("band", "nearest"):
        return method
    if method in ("lanczos", "bicubic", "bilinear"):
        return _validate_interpolate_method(method, "sampling_method")
    raise ValueError(
        "Unrecognized sampling_method '%s'. Expected lanczos, bicubic, bilinear, nearest, or band."
        % method
    )


def _sparse_scatter_model_flux(
    medfluxes, scatfluxes, pixels, labels=None, target_label=None, min_anchor_samples=2
):
    # One-sample rows have no empirical scatter, so estimate scatflux^2 from
    # nearby rows with enough samples using a simple flux-dependent model.
    medfluxes = np.asarray(medfluxes, dtype=float)
    scatfluxes = np.asarray(scatfluxes, dtype=float)
    pixels = np.asarray(pixels, dtype=int)
    use = (
        (pixels >= min_anchor_samples)
        & np.isfinite(medfluxes)
        & (medfluxes > 0)
        & np.isfinite(scatfluxes)
        & (scatfluxes > 0)
    )
    if not target_label is None:
        use &= np.asarray(labels) == target_label
    flux = medfluxes[use]
    scatter2 = scatfluxes[use] ** 2
    if len(scatter2) == 0:
        return None
    if len(scatter2) < 2 or np.ptp(flux) == 0:
        return 0.0, np.median(scatter2)

    slope, intercept = np.linalg.lstsq(
        np.column_stack([flux, np.ones_like(flux)]), scatter2, rcond=None
    )[0]
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    return max(0.0, slope), max(0.0, intercept)


def _estimate_sparse_scatflux(medfluxes, scatfluxes, pixels, sample_labels):
    medfluxes = np.asarray(medfluxes, dtype=float)
    scatfluxes = np.asarray(scatfluxes, dtype=float).copy()
    pixels = np.asarray(pixels, dtype=int)
    sample_labels = np.asarray(sample_labels)
    sparse = pixels == 1
    if not np.any(sparse):
        return scatfluxes

    # Prefer anchors from the same sampling regime, since interpolated line,
    # nearest-neighbor line, and band samples can have different scatter.
    fallback_model = _sparse_scatter_model_flux(medfluxes, scatfluxes, pixels)
    for sample_label in np.unique(sample_labels[sparse]):
        sparse_indices = np.flatnonzero(sparse & (sample_labels == sample_label))
        model = _sparse_scatter_model_flux(
            medfluxes, scatfluxes, pixels, sample_labels, sample_label
        )
        if model is None:
            model = fallback_model
        if model is None:
            continue

        slope, intercept = model
        model_flux = np.maximum(medfluxes[sparse_indices], 0.0)
        model_scatter = np.sqrt(np.maximum(slope * model_flux + intercept, 0.0))
        replace = np.isfinite(model_scatter) & (model_scatter > 0)
        scatfluxes[sparse_indices[replace]] = model_scatter[replace]
    return scatfluxes


def _empty_fmode_measurements(modes):
    return {
        "a": [np.nan] * (len(modes) + 1),
        "b": [np.nan] * (len(modes) + 1),
    }


def _fit_harmonic_measurements(flux, theta, modes):
    modes = tuple(modes)
    fit_modes = tuple(range(1, max(modes) + 1)) if len(modes) > 0 else ()
    flux = np.asarray(flux, dtype=float)
    theta = np.asarray(theta, dtype=float) % (2 * np.pi)
    keep = np.isfinite(flux) & np.isfinite(theta)
    flux = flux[keep]
    theta = theta[keep]
    if len(flux) == 0:
        return _empty_fmode_measurements(modes)

    design = [np.ones(len(theta))]
    # Include intervening orders in the fit to reduce leakage from gappy sampling.
    for mode in fit_modes:
        design.append(np.sin(mode * theta))
        design.append(np.cos(mode * theta))
    design = np.column_stack(design)
    if len(flux) < design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        return _empty_fmode_measurements(modes)

    coeffs, _, _, _ = np.linalg.lstsq(design, flux, rcond=None)
    if not np.all(np.isfinite(coeffs)):
        return _empty_fmode_measurements(modes)
    mean_flux = coeffs[0]
    norm = np.abs(mean_flux)
    fit_coeffs = {}
    for i, mode in enumerate(fit_modes):
        fit_coeffs[mode] = (coeffs[1 + 2 * i], coeffs[2 + 2 * i])
    measurements = {"a": [0.0], "b": [mean_flux]}
    for mode in modes:
        sin_coeff, cos_coeff = fit_coeffs[mode]
        # AutoProf stores harmonic amplitudes with the same 2*abs(I0) scaling.
        measurements["a"].append(_finite_divide(-0.5 * sin_coeff, norm))
        measurements["b"].append(_finite_divide(0.5 * cos_coeff, norm))
    return measurements


def _isocoefs_interpolate_method(options, fallback_method):
    if "ap_isocoefs_interpolate_method" in options:
        method = options["ap_isocoefs_interpolate_method"]
    elif "ap_isoextract_interpolate_method" in options:
        method = options["ap_isoextract_interpolate_method"]
    else:
        method = fallback_method
    return _validate_interpolate_method(method, "ap_isocoefs_interpolate_method")


def _extract_harmonic_measurement_samples(
    dat, R, parameters, center, mask, options, rad_interp, interp_method
):
    return _iso_extract_with_interp_cutoff(
        dat,
        R,
        parameters,
        center,
        mask=mask,
        more=True,
        rad_interp=rad_interp,
        interp_method=interp_method,
        interp_window=(
            int(options["ap_iso_interpolate_window"])
            if "ap_iso_interpolate_window" in options
            else 3
        ),
        sigmaclip=options["ap_isoclip"] if "ap_isoclip" in options else False,
        sclip_iterations=(
            options["ap_isoclip_iterations"] if "ap_isoclip_iterations" in options else 10
        ),
        sclip_nsigma=options["ap_isoclip_nsigma"] if "ap_isoclip_nsigma" in options else 5,
    )


def _Generate_Profile(IMG, results, R, parameters, options, forced_sampling_methods=None):

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
    # Internal bookkeeping for one-sample uncertainty repair; labels are also
    # written to the profile table when requested.
    medfluxes = []
    scatfluxes = []
    sample_labels = []
    sbfix = []
    sbfixE = []
    measFmodes = []
    output_sampling_method = (
        "ap_iso_output_sampling_method" in options
        and options["ap_iso_output_sampling_method"]
    )
    if forced_sampling_methods is not None:
        forced_sampling_methods = list(
            _normalize_sampling_method(m) for m in forced_sampling_methods
        )

    count_neg = 0
    medflux = np.inf
    end_prof = len(R)
    compare_interp = []
    measure_coefs = "ap_iso_measurecoefs" in options and not options["ap_iso_measurecoefs"] is None
    rad_interp = _iso_interpolate_radius(options, results)
    isoband_flux_threshold = _isoband_flux_threshold(results, options, zeropoint)
    isoextract_interp_method = _validate_interpolate_method(
        options["ap_isoextract_interpolate_method"]
        if "ap_isoextract_interpolate_method" in options
        else "bilinear",
        "ap_isoextract_interpolate_method",
    )
    if measure_coefs:
        isocoefs_interp_method = _isocoefs_interpolate_method(
            options, isoextract_interp_method
        )
    for i in range(len(R)):
        if "ap_isoband_fixed" in options and options["ap_isoband_fixed"]:
            isobandwidth = options["ap_isoband_width"] if "ap_isoband_width" in options else 0.5
        else:
            isobandwidth = R[i] * (
                options["ap_isoband_width"] if "ap_isoband_width" in options else 0.025
            )
        forced_sampling_method = (
            forced_sampling_methods[i]
            if forced_sampling_methods is not None and i < len(forced_sampling_methods)
            else None
        )
        if forced_sampling_method is not None:
            sampling_method = forced_sampling_method
        elif medflux <= isoband_flux_threshold and isobandwidth >= 0.5:
            sampling_method = "band"
        else:
            sampling_method = _resolve_isoextract_interp_method(
                R[i], parameters[i], rad_interp, isoextract_interp_method
            )

        if sampling_method != "band":
            isovals = _iso_extract(
                dat,
                R[i],
                parameters[i],
                results["center"],
                mask=mask,
                more=True,
                interp_method=sampling_method,
                interp_window=(
                    int(options["ap_iso_interpolate_window"])
                    if "ap_iso_interpolate_window" in options
                    else 3
                ),
                sigmaclip=options["ap_isoclip"] if "ap_isoclip" in options else False,
                sclip_iterations=(
                    options["ap_isoclip_iterations"] if "ap_isoclip_iterations" in options else 10
                ),
                sclip_nsigma=options["ap_isoclip_nsigma"] if "ap_isoclip_nsigma" in options else 5,
            )
        else:
            # Band sampling has a different effective noise behavior.
            isovals = _iso_between(
                dat,
                R[i] - isobandwidth,
                R[i] + isobandwidth,
                parameters[i],
                results["center"],
                mask=mask,
                more=True,
                sigmaclip=options["ap_isoclip"] if "ap_isoclip" in options else False,
                sclip_iterations=(
                    options["ap_isoclip_iterations"] if "ap_isoclip_iterations" in options else 10
                ),
                sclip_nsigma=options["ap_isoclip_nsigma"] if "ap_isoclip_nsigma" in options else 5,
            )
        if measure_coefs:
            coef_isovals = _extract_harmonic_measurement_samples(
                dat,
                R[i],
                parameters[i],
                results["center"],
                mask,
                options,
                rad_interp,
                isocoefs_interp_method,
            )
            coef_measurements = _fit_harmonic_measurements(
                coef_isovals[0], coef_isovals[1], options["ap_iso_measurecoefs"]
            )
        if len(isovals[0]) == 0:
            pixels.append(0)
            maskedpixels.append(isovals[2])
            medfluxes.append(np.nan)
            scatfluxes.append(np.nan)
            sample_labels.append(sampling_method)
            if fluxunits == "intensity":
                sb.append(np.nan)
                sbE.append(np.nan)
                cogdirect.append(np.nan)
            else:
                sb.append(np.nan)
                sbE.append(np.nan)
                cogdirect.append(np.nan)
            if measure_coefs:
                measFmodes.append(coef_measurements)
            continue
        isotot = np.sum(_iso_between(dat, 0, R[i], parameters[i], results["center"], mask=mask))
        medflux = _average(
            isovals[0],
            options["ap_isoaverage_method"] if "ap_isoaverage_method" in options else "median",
        )
        scatflux = _scatter(
            isovals[0],
            options["ap_isoaverage_method"] if "ap_isoaverage_method" in options else "median",
        )
        if measure_coefs:
            measFmodes.append(coef_measurements)

        pixels.append(len(isovals[0]))
        maskedpixels.append(isovals[2])
        medfluxes.append(medflux)
        scatfluxes.append(scatflux)
        sample_labels.append(sampling_method)
        if fluxunits == "intensity":
            sb.append(medflux / options["ap_pixscale"] ** 2)
            sbE.append(scatflux / np.sqrt(len(isovals[0])))
            cogdirect.append(isotot)
        else:
            sb.append(
                flux_to_sb(medflux, options["ap_pixscale"], zeropoint)
                if medflux > 0
                else np.nan
            )
            sbE.append(
                (2.5 * scatflux / (np.sqrt(len(isovals[0])) * medflux * np.log(10)))
                if medflux > 0
                else np.nan
            )
            cogdirect.append(flux_to_mag(isotot, zeropoint) if isotot > 0 else np.nan)
        if medflux <= 0:
            count_neg += 1
        if (
            "ap_truncate_evaluation" in options
            and options["ap_truncate_evaluation"]
            and count_neg >= 2
        ):
            end_prof = i + 1
            break

    # Replace only one-sample scatflux values; normal rows keep their measured
    # scatter and empty rows remain invalid.
    scatfluxes[:end_prof] = list(
        _estimate_sparse_scatflux(
            medfluxes[:end_prof],
            scatfluxes[:end_prof],
            pixels[:end_prof],
            sample_labels[:end_prof],
        )
    )
    for i in np.flatnonzero(np.asarray(pixels[:end_prof]) == 1):
        if fluxunits == "intensity":
            sbE[i] = scatfluxes[i]
        elif medfluxes[i] > 0:
            sbE[i] = 2.5 * scatfluxes[i] / (medfluxes[i] * np.log(10))

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
            cog = np.full(end_prof, np.nan)
            cogE = np.full(end_prof, np.nan)
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
            cog = np.full(end_prof, np.nan)
            cogE = np.full(end_prof, np.nan)

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

    # Sentinels are an output convention and must not enter profile calculations.
    if fluxunits == "intensity":
        cog = np.asarray(cog)
        cogE = np.asarray(cogE)
        invalid_cog = np.logical_or(np.logical_not(np.isfinite(cog)), cog < 0)
        cog[invalid_cog] = -99.999
        cogE[np.logical_or(invalid_cog, np.logical_not(np.isfinite(cogE)))] = -99.999
    else:
        sb = np.asarray(sb)
        sbE = np.asarray(sbE)
        cog = np.asarray(cog)
        cogE = np.asarray(cogE)
        cogdirect = np.asarray(cogdirect)
        invalid_sb = np.logical_not(np.isfinite(sb))
        sb[invalid_sb] = 99.999
        sbE[np.logical_or(invalid_sb, np.logical_not(np.isfinite(sbE)))] = 99.999
        invalid_cog = np.logical_or(np.logical_not(np.isfinite(cog)), cog > 99)
        cog[invalid_cog] = 99.999
        cogE[np.logical_or(invalid_cog, np.logical_not(np.isfinite(cogE)))] = 99.999
        cogdirect[np.logical_not(np.isfinite(cogdirect))] = 99.999

    SBprof_data = dict((h, None) for h in params)
    SBprof_data["R"] = list(R[:end_prof] * options["ap_pixscale"])
    SBprof_data["I" if fluxunits == "intensity" else "SB"] = list(sb)
    SBprof_data["I_e" if fluxunits == "intensity" else "SB_e"] = list(sbE)
    SBprof_data["totflux" if fluxunits == "intensity" else "totmag"] = list(cog)
    SBprof_data["totflux_e" if fluxunits == "intensity" else "totmag_e"] = list(cogE)
    SBprof_data["ellip"] = list(parameters[p]["ellip"] for p in range(end_prof))
    SBprof_data["ellip_e"] = list(parameters[p]["ellip err"] for p in range(end_prof))
    SBprof_data["pa"] = list(parameters[p]["pa"] * 180 / np.pi for p in range(end_prof))
    SBprof_data["pa_e"] = list(parameters[p]["pa err"] * 180 / np.pi for p in range(end_prof))
    SBprof_data["pixels"] = list(pixels)
    SBprof_data["maskedpixels"] = list(maskedpixels)
    SBprof_data["totflux_direct" if fluxunits == "intensity" else "totmag_direct"] = list(cogdirect)
    if output_sampling_method:
        params.append("sampling_method")
        SBprof_units["sampling_method"] = "none"
        SBprof_data["sampling_method"] = list(sample_labels[:end_prof])

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
        Plot_Phase_Profile(np.array(SBprof_data["R"]), parameters[:end_prof], results, options)
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

    ap_forced_use_sampling_method : bool, default True
      If the forcing profile has a *sampling_method* column, use it to
      choose lanczos, bicubic, bilinear, nearest-neighbor, or band sampling
      for each forced isophote.

    ap_isoband_start : float, default 2
      The noise level at which to begin sampling a band of pixels to
      compute SB instead of sampling a line of pixels near the
      isophote in units of pixel flux noise. Will never initiate band
      averaging if the band width is less than half a pixel

    ap_isoband_start_sb : float, default None
      Surface-brightness level in mag arcsec^-2 at which to begin
      sampling a band of pixels instead of sampling a line of pixels
      near the isophote. Overrides *ap_isoband_start* if set.

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
      Use image interpolation for isophotes with semi-major axis
      less than this number times the PSF HWHM.

    ap_isoextract_interpolate_method : string, default 'bilinear'
      Select method for flux interpolation on image, options are
      'lanczos', 'bicubic', and 'bilinear'. Default is 'bilinear'.

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
      galaxies. Harmonic terms are fit directly to the valid
      azimuthal samples. For a fitted term
      I(theta) = I_0 + A_i sin(i theta) + B_i cos(i theta), AutoProf
      reports a_i = -0.5 A_i/abs(I_0) and b_i = 0.5 B_i/abs(I_0).
      The fit includes every harmonic order up to the highest requested
      order, but only requested orders are reported. Not activated by
      default as it adds to computation time.

    ap_isocoefs_interpolate_method : string, default None
      Select image sampling method used for *ap_iso_measurecoefs*.
      Options are 'lanczos', 'bicubic', and 'bilinear'. By default,
      this follows *ap_isoextract_interpolate_method*. Coefficient
      measurement samples the fitted isophote line, even where the SB
      profile uses isophote-band sampling.

    ap_iso_output_sampling_method : bool, default False
      Add a *sampling_method* column to the output profile. Values are
      'lanczos', 'bicubic', 'bilinear', 'nearest', and 'band'.

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
            if len(l.strip()) == 0 or l[0] == "#":
                continue
            readfrom = i
            break
        header = list(h.strip() for h in raw[readfrom].split(","))
        force = dict((h, []) for h in header)
        for l in raw[readfrom + 1 :]:
            if len(l) > 0 and l[0] == "#":
                continue  # skip comments
            D = list(l.split(","))
            if len(D) != len(header):
                continue  # Skip missmatched rows with header
            try:
                float(D[0].strip())
            except ValueError:
                continue  # Skip non-numeric rows
            for d, h in zip(D, header):
                if h == "sampling_method":
                    force[h].append(d.strip())
                else:
                    force[h].append(float(d.strip()))

    force["pa"] = PA_shift_convention(np.array(force["pa"]), deg=True) * np.pi / 180

    parameters = list(
        {
            "ellip": force["ellip"][i],
            "pa": (
                force["pa"][i]
                + (options["ap_forced_pa_shift"] if "ap_forced_pa_shift" in options else 0.0)
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

    forced_sampling_methods = (
        force["sampling_method"]
        if (
            "sampling_method" in force
            and (
                "ap_forced_use_sampling_method" not in options
                or options["ap_forced_use_sampling_method"]
            )
        )
        else None
    )

    return IMG, _Generate_Profile(
        IMG,
        results,
        np.array(force["R"]) / options["ap_pixscale"],
        parameters,
        options,
        forced_sampling_methods=forced_sampling_methods,
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

    ap_isoband_start_sb : float, default None
      Surface-brightness level in mag arcsec^-2 at which to begin
      sampling a band of pixels instead of sampling a line of pixels
      near the isophote. Overrides *ap_isoband_start* if set.

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
      Use image interpolation for isophotes with semi-major axis
      less than this number times the PSF HWHM.

    ap_isoextract_interpolate_method : string, default 'bilinear'
      Select method for flux interpolation on image, options are
      'lanczos', 'bicubic', and 'bilinear'. Default is 'bilinear'.

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
      galaxies. Harmonic terms are fit directly to the valid
      azimuthal samples. For a fitted term
      I(theta) = I_0 + A_i sin(i theta) + B_i cos(i theta), AutoProf
      reports a_i = -0.5 A_i/abs(I_0) and b_i = 0.5 B_i/abs(I_0).
      The fit includes every harmonic order up to the highest requested
      order, but only requested orders are reported. Not activated by
      default as it adds to computation time.

    ap_isocoefs_interpolate_method : string, default None
      Select image sampling method used for *ap_iso_measurecoefs*.
      Options are 'lanczos', 'bicubic', and 'bilinear'. By default,
      this follows *ap_isoextract_interpolate_method*. Coefficient
      measurement samples the fitted isophote line, even where the SB
      profile uses isophote-band sampling.

    ap_iso_output_sampling_method : bool, default False
      Add a *sampling_method* column to the output profile. Values are
      'lanczos', 'bicubic', 'bilinear', 'nearest', and 'band'.

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
        (
            options["ap_sampleinitR"]
            if "ap_sampleinitR" in options
            else min(1.0, results["psf fwhm"] / 2)
        )
    ]
    while (
        (
            (R[-1] < options["ap_sampleendR"] if "ap_sampleendR" in options else True)
            and R[-1] < 3 * results["fit R"][-1]
        )
        or (options["ap_extractfull"] if "ap_extractfull" in options else False)
    ) and R[-1] < max(IMG.shape) / np.sqrt(2):
        if "ap_samplestyle" in options and options["ap_samplestyle"] == "geometric-linear":
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
    logging.info("%s: R complete in range [%.1f,%.1f]" % (options["ap_name"], R[0], R[-1]))

    # Interpolate profile values, when extrapolating just take last point
    tmp_pa_s = UnivariateSpline(results["fit R"], np.sin(2 * results["fit pa"]), ext=3, s=0)(R)
    tmp_pa_c = UnivariateSpline(results["fit R"], np.cos(2 * results["fit pa"]), ext=3, s=0)(R)
    E = _x_to_eps(
        UnivariateSpline(results["fit R"], _inv_x_to_eps(results["fit ellip"]), ext=3, s=0)(R)
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
                UnivariateSpline(results["fit R"], results["fit ellip_err"], ext=3, s=0)(R[i]),
                a_min=1e-3,
                a_max=None,
            )
            parameters[i]["pa err"] = np.clip(
                UnivariateSpline(results["fit R"], results["fit pa_err"], ext=3, s=0)(R[i]),
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
    photutils_dat = _photutils_masked_data(dat, results)
    if not "fit R" in results and not "fit photutils isolist" in results:
        logging.info("%s: photutils fitting and extracting image data" % options["ap_name"])
        geo = EllipseGeometry(
            x0=results["center"]["x"],
            y0=results["center"]["y"],
            sma=results["init R"] / 2,
            eps=results["init ellip"],
            pa=results["init pa"],
        )
        ellipse = Photutils_Ellipse(photutils_dat, geometry=geo)

        isolist = ellipse.fit_image(fix_center=True, linear=False)
        res.update(
            {
                "fit photutils isolist": isolist,
                "auxfile fitlimit": (
                    "fit limit semi-major axis: %.2f pix" % isolist.sma[-1]
                    if len(isolist.sma) > 0
                    else "fit limit semi-major axis: no valid isophotes"
                ),
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
            ES = EllipseSample(photutils_dat, sma=results["fit R"][i], geometry=geo)
            ES.update(fixed_parameters=None)
            list_iso.append(Isophote(ES, niter=30, valid=True, stop_code=0))

        isolist = IsophoteList(list_iso)
        res.update(
            {
                "fit photutils isolist": isolist,
                "auxfile fitlimit": (
                    "fit limit semi-major axis: %.2f pix" % isolist.sma[-1]
                    if len(isolist.sma) > 0
                    else "fit limit semi-major axis: no valid isophotes"
                ),
            }
        )
    else:
        isolist = results["fit photutils isolist"]

    for i in range(len(isolist.sma)):
        SBprof_data["R"].append(isolist.sma[i] * options["ap_pixscale"])
        medflux = _finite_median(isolist.sample[i].values[2])
        if fluxunits == "intensity":
            SBprof_data["I"].append(
                medflux / options["ap_pixscale"] ** 2
            )
            SBprof_data["I_e"].append(isolist.int_err[i])
            SBprof_data["totflux"].append(isolist.tflux_e[i])
            SBprof_data["totflux_e"].append(
                _finite_divide(isolist.rms[i], np.sqrt(isolist.npix_e[i]))
            )
        else:
            SBprof_data["SB"].append(
                flux_to_sb(medflux, options["ap_pixscale"], zeropoint)
                if medflux > 0
                else np.nan
            )
            SBprof_data["SB_e"].append(
                _finite_divide(2.5 * isolist.int_err[i], isolist.intens[i] * np.log(10))
            )
            SBprof_data["totmag"].append(
                flux_to_mag(isolist.tflux_e[i], zeropoint)
                if np.isfinite(isolist.tflux_e[i]) and isolist.tflux_e[i] > 0
                else np.nan
            )
            SBprof_data["totmag_e"].append(
                _finite_divide(
                    2.5 * isolist.rms[i],
                    np.sqrt(isolist.npix_e[i]) * isolist.tflux_e[i] * np.log(10),
                )
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
    res.update({"prof header": params, "prof units": SBprof_units, "prof data": SBprof_data})

    if "ap_doplot" in options and options["ap_doplot"] and len(SBprof_data["R"]) > 0:
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
