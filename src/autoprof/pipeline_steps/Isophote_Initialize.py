import numpy as np
from scipy.fftpack import fft, ifft, dct, idct
from scipy.optimize import minimize
from scipy.stats import iqr
import sys
import os

from ..autoprofutils.SharedFunctions import (
    _iso_extract,
    _has_enough_isophote_coverage,
    _interpolate_invalid_isophote_samples,
    _x_to_eps,
    _x_to_pa,
    _inv_x_to_pa,
    _inv_x_to_eps,
    LSBImage,
    Angle_Average,
    Angle_Median,
    AddLogo,
    PA_shift_convention,
    Sigma_Clip_Upper,
    autocolours,
    Smooth_Mode,
)
from ..autoprofutils.Diagnostic_Plots import (
    Plot_Isophote_Init_Ellipse,
    Plot_Isophote_Init_Optimize,
)
import logging
from copy import copy
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from time import time

__all__ = ("Isophote_Init_Forced", "Isophote_Initialize", "Isophote_Initialize_mean")


def _isophote_flux(isovals):
    return isovals[0] if isinstance(isovals, tuple) else isovals


def _finite_average(values):
    finite_values = [v for v in values if np.isfinite(v)]
    if len(finite_values) == 0:
        return np.inf
    return np.mean(finite_values)


def _init_radius_candidates(radius, max_radius):
    seen = set()
    for scale in (1.0, 0.8, 1.2, 0.6, 1.4, 0.5, 1.6, 0.4, 1.8, 0.3, 2.0):
        candidate = radius * scale
        key = round(candidate, 6)
        if candidate < 1.0 or candidate >= max_radius or key in seen:
            continue
        seen.add(key)
        yield candidate


def _extract_init_fft_samples(
    dat,
    radius,
    params,
    center,
    mask,
    interp_method=None,
    sigmaclip=False,
    sclip_nsigma=3,
):
    kwargs = {}
    if not interp_method is None:
        kwargs["interp_method"] = interp_method
    flux, theta, choose, _ = _iso_extract(
        dat,
        radius,
        params,
        center,
        more=True,
        mask=mask,
        sigmaclip=sigmaclip,
        sclip_nsigma=sclip_nsigma,
        return_choose=True,
        **kwargs,
    )
    if not _has_enough_isophote_coverage(theta, choose):
        return None
    flux, theta = _interpolate_invalid_isophote_samples(flux, theta, choose)
    if len(flux) < 3:
        return None
    return flux, theta


def _ellip_loss_grid(test_ellip, loss_func, dat, radius, phase, center, noise, mask):
    test_f2 = []
    for e in test_ellip:
        test_f2.append(
            _finite_average(
                loss_func(e, dat, radius * m, phase, center, noise, mask)
                for m in np.linspace(0.8, 1.2, 5)
            )
        )
    return test_f2


def _select_init_radius(test_ellip, loss_func, dat, radius, phase, center, noise, mask, max_radius):
    test_f2 = []
    for test_radius in _init_radius_candidates(radius, max_radius):
        test_f2 = _ellip_loss_grid(test_ellip, loss_func, dat, test_radius, phase, center, noise, mask)
        if np.any(np.isfinite(test_f2)):
            return test_radius, test_f2
    return radius, test_f2


def Isophote_Init_Forced(IMG, results, options):
    """Read global elliptical isophote to a galaxy from an aux file.

    Extracts global ellipse parameters from the corresponding aux file for a given .prof file.

    Parameters
    -----------------
    ap_forcing_profile : string, default None
      File path to .prof file providing forced photometry PA and
      ellip values to apply to *ap_image_file* (required for forced
      photometry)

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'psf fwhm'
    - 'center'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'init ellip': , # Ellipticity of the global fit (float)
         'init pa': ,# Position angle of the global fit (float)
         'init R': ,# Semi-major axis length of global fit (float)
         'auxfile initialize': # optional, message for aux file to record the global ellipticity and postition angle (string)

        }

    """

    with open(options["ap_forcing_profile"][:-4] + "aux", "r") as f:
        for line in f.readlines():
            if "global ellipticity" in line:
                ellip = float(line[line.find(":") + 1 : line.find("+-")].strip())
                ellip_err = float(line[line.find("+-") + 2 : line.find(",")].strip())
                pa = (
                    PA_shift_convention(
                        float(
                            line[line.find("pa:") + 3 : line.find("+-", line.find("pa:"))].strip()
                        ),
                        deg=True,
                    )
                    * np.pi
                    / 180
                )
                pa_err = (
                    float(line[line.find("+-", line.find("pa:")) + 2 : line.find("deg")].strip())
                    * np.pi
                    / 180
                )
                R = float(
                    line[line.find("size:") + 5 : line.find("pix", line.find("size:"))].strip()
                )
                break

    auxmessage = "global ellipticity: %.3f +- %.3f, pa: %.3f +- %.3f deg, size: %f pix" % (
        ellip,
        ellip_err,
        PA_shift_convention(pa) * 180 / np.pi,
        pa_err * 180 / np.pi,
        R,
    )

    return IMG, {
        "init ellip": ellip,
        "init ellip_err": ellip_err,
        "init pa": pa,
        "init pa_err": pa_err,
        "init R": R,
        "auxfile initialize": auxmessage,
    }


def _fitEllip_loss(e, dat, r, p, c, n, m, interp_method=None):
    isovals = _extract_init_fft_samples(
        dat,
        r,
        {"ellip": e, "pa": p},
        c,
        m,
        interp_method=interp_method,
        sigmaclip=True,
        sclip_nsigma=3,
    )
    if isovals is None:
        return np.inf
    isovals = isovals[0]
    coefs = fft(np.clip(isovals, a_max=np.quantile(isovals, 0.85), a_min=None))
    denominator = max(0, np.median(isovals)) + n
    if (not np.isfinite(denominator)) or denominator <= 0:
        return np.inf
    loss = (iqr(isovals, rng=[16, 84]) / 2 + np.abs(coefs[2]) / len(isovals)) / denominator
    return loss if np.isfinite(loss) else np.inf


def Isophote_Initialize(IMG, results, options):
    """Fit global elliptical isophote to a galaxy image using FFT coefficients.

    A global position angle and ellipticity are fit in a two step
    process.  First, a series of circular isophotes are geometrically
    sampled until they approach the background level of the image.  An
    FFT is taken for the flux values around each isophote and the
    phase of the second coefficient is used to determine a direction.
    The average direction for the outer isophotes is taken as the
    position angle of the galaxy.  Second, with fixed position angle
    the ellipticity is optimized to minimize the amplitude of the
    second FFT coefficient relative to the median flux in an isophote.

    To compute the error on position angle we use the standard
    deviation of the outer values from step one.  For ellipticity the
    error is computed by optimizing the ellipticity for multiple
    isophotes within 1 PSF length of each other.

    Parameters
    -----------------
    ap_fit_limit : float, default 2
      noise level out to which to extend the fit in units of pixel background noise level. Default is 2, smaller values will end fitting further out in the galaxy image.

    ap_isoinit_pa_set : float, default None
      User set initial position angle in degrees, will override the calculation.

    ap_isoinit_ellip_set : float, default None
      User set initial ellipticity (1 - b/a), will override the calculation.

    ap_isoinit_R_set : float, default None
        User set initial semi-major axis length, will override the calculation.

    ap_isoinit_interpolate_method : string, default None
      Select method for flux interpolation while initializing
      isophotes. Options are 'lanczos', 'bicubic', and 'bilinear'.
      Default None uses the standard isophote extraction default.

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'psf fwhm'
    - 'center'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'init ellip': , # Ellipticity of the global fit (float)
         'init pa': ,# Position angle of the global fit (float)
         'init R': ,# Semi-major axis length of global fit (float)
         'auxfile initialize': # optional, message for aux file to record the global ellipticity and postition angle (string)

        }

    """

    ######################################################################
    # Initial attempt to find size of galaxy in image
    # based on when isophotes SB values start to get
    # close to the background noise level
    circ_ellipse_radii = [1.0]
    allphase = []
    dat = IMG - results["background"]
    mask = results["mask"] if "mask" in results else None
    if not np.any(mask):
        mask = None
    interp_method = (
        options["ap_isoinit_interpolate_method"]
        if "ap_isoinit_interpolate_method" in options
        else None
    )

    if "ap_isoinit_R_set" in options:
        sample_radii = np.logspace(
            np.log10(circ_ellipse_radii[0]),
            np.log10(options["ap_isoinit_R_set"]),
            10,
        )
        for r in sample_radii[1:]:
            isovals = _extract_init_fft_samples(
                dat,
                r,
                {"ellip": 0.0, "pa": 0.0},
                results["center"],
                mask=mask,
                interp_method=interp_method,
                sigmaclip=True,
                sclip_nsigma=3,
            )
            if isovals is None:
                continue
            circ_ellipse_radii.append(r)
            coefs = fft(isovals[0])
            allphase.append(coefs[2])
    else:
        r = circ_ellipse_radii[-1]
        while r < (len(IMG) / 2):
            r *= 1 + 0.2
            isovals = _extract_init_fft_samples(
                dat,
                r,
                {"ellip": 0.0, "pa": 0.0},
                results["center"],
                mask=mask,
                interp_method=interp_method,
                sigmaclip=True,
                sclip_nsigma=3,
            )
            if isovals is None:
                continue
            circ_ellipse_radii.append(r)
            coefs = fft(isovals[0])
            allphase.append(coefs[2])
            # Stop when at 3 time background noise
            if (
                np.quantile(isovals[0], 0.8)
                < (
                    (options["ap_fit_limit"] + 1 if "ap_fit_limit" in options else 3)
                    * results["background noise"]
                )
                and len(circ_ellipse_radii) > 4
            ):
                break

    if len(allphase) == 0 or len(circ_ellipse_radii) < 2:
        raise ValueError("Could not initialize isophotes: no finite samples found.")

    logging.info("%s: init scale: %f pix" % (options["ap_name"], circ_ellipse_radii[-1]))
    # Find global position angle.
    phase = (-Angle_Median(np.angle(allphase[-5:])) / 2) % np.pi
    if "ap_isoinit_pa_set" in options:
        phase = PA_shift_convention(options["ap_isoinit_pa_set"] * np.pi / 180)

    # Find global ellipticity
    test_ellip = np.linspace(0.05, 0.95, 15)
    init_radius, test_f2 = _select_init_radius(
        test_ellip,
        lambda e, d, r, p, c, n, m: _fitEllip_loss(
            e, d, r, p, c, n, m, interp_method=interp_method
        ),
        dat,
        circ_ellipse_radii[-2],
        phase,
        results["center"],
        results["background noise"],
        mask,
        len(IMG) / 2,
    )
    if not np.any(np.isfinite(test_f2)):
        raise ValueError("Could not initialize isophotes: no finite ellipticity samples found.")
    # Find global ellipticity: second pass
    ellip = test_ellip[np.argmin(test_f2)]
    test_ellip = np.linspace(ellip - 0.05, ellip + 0.05, 15)
    init_radius, test_f2 = _select_init_radius(
        test_ellip,
        lambda e, d, r, p, c, n, m: _fitEllip_loss(
            e, d, r, p, c, n, m, interp_method=interp_method
        ),
        dat,
        init_radius,
        phase,
        results["center"],
        results["background noise"],
        mask,
        len(IMG) / 2,
    )
    if not np.any(np.isfinite(test_f2)):
        raise ValueError("Could not initialize isophotes: no finite ellipticity samples found.")
    ellip = test_ellip[np.argmin(test_f2)]
    res = minimize(
        lambda e, d, r, p, c, n, msk: _finite_average(
                _fitEllip_loss(
                    _x_to_eps(e[0]), d, r * m, p, c, n, msk, interp_method=interp_method
                )
                for m in np.linspace(0.8, 1.2, 5)
        ),
        x0=_inv_x_to_eps(ellip),
        args=(
            dat,
            init_radius,
            phase,
            results["center"],
            results["background noise"],
            mask,
        ),
        method="Nelder-Mead",
        options={
            "initial_simplex": [
                [_inv_x_to_eps(ellip) - 1 / 15],
                [_inv_x_to_eps(ellip) + 1 / 15],
            ]
        },
    )
    if res.success and np.isfinite(res.fun):
        logging.debug(
            "%s: using optimal ellipticity %.3f over grid ellipticity %.3f"
            % (options["ap_name"], _x_to_eps(res.x[0]), ellip)
        )
        ellip = _x_to_eps(res.x[0])
    if "ap_isoinit_ellip_set" in options:
        ellip = options["ap_isoinit_ellip_set"]

    # Compute the error on the parameters
    ######################################################################
    RR = np.linspace(
        init_radius - results["psf fwhm"],
        init_radius + results["psf fwhm"],
        10,
    )
    errallphase = []
    err_radii = []
    for rr in RR:
        isovals = _extract_init_fft_samples(
            dat,
            rr,
            {"ellip": 0.0, "pa": 0.0},
            results["center"],
            mask=mask,
            interp_method=interp_method,
            sigmaclip=True,
            sclip_nsigma=3,
        )
        if isovals is None:
            continue
        coefs = fft(isovals[0])
        errallphase.append(coefs[2])
        err_radii.append(rr)
    if len(errallphase) > 0 and np.isfinite(np.mean(errallphase)) and np.mean(errallphase) != 0:
        sample_pas = (-np.angle(1j * np.array(errallphase) / np.mean(errallphase)) / 2) % np.pi
        pa_err = iqr(sample_pas, rng=[16, 84]) / 2
    else:
        sample_pas = np.array([])
        pa_err = np.nan
    res_multi = [
        minimize(
            lambda e, d, r, p, c, n, m: _fitEllip_loss(
                _x_to_eps(e[0]), d, r, p, c, n, m, interp_method=interp_method
            ),
            x0=_inv_x_to_eps(ellip),
            args=(
                dat,
                rrp[0],
                rrp[1],
                results["center"],
                results["background noise"],
                mask,
            ),
            method="Nelder-Mead",
            options={
                "initial_simplex": [
                    [_inv_x_to_eps(ellip) - 1 / 15],
                    [_inv_x_to_eps(ellip) + 1 / 15],
                ]
            },
        )
        for rrp in zip(err_radii, sample_pas)
    ]
    ellip_samples = [
        _x_to_eps(rm.x[0])
        for rm in res_multi
        if rm.success and np.isfinite(rm.fun) and np.all(np.isfinite(rm.x))
    ]
    ellip_err = iqr(ellip_samples, rng=[16, 84]) / 2 if len(ellip_samples) > 0 else np.nan

    circ_ellipse_radii = np.array(circ_ellipse_radii)

    if "ap_doplot" in options and options["ap_doplot"]:
        Plot_Isophote_Init_Ellipse(dat, circ_ellipse_radii, ellip, phase, results, options)
        Plot_Isophote_Init_Optimize(
            circ_ellipse_radii,
            allphase,
            phase,
            pa_err,
            test_ellip,
            test_f2,
            ellip,
            ellip_err,
            results,
            options,
        )

    auxmessage = "global ellipticity: %.3f +- %.3f, pa: %.3f +- %.3f deg, size: %f pix" % (
        ellip,
        ellip_err,
        PA_shift_convention(phase) * 180 / np.pi,
        pa_err * 180 / np.pi,
        init_radius,
    )
    return IMG, {
        "init ellip": ellip,
        "init ellip_err": ellip_err,
        "init pa": phase,
        "init pa_err": pa_err,
        "init R": init_radius,
        "auxfile initialize": auxmessage,
    }


def _fitEllip_mean_loss(e, dat, r, p, c, n, m, interp_method=None):
    isovals = _extract_init_fft_samples(
        dat, r, {"ellip": e, "pa": p}, c, mask=m, interp_method=interp_method
    )
    if isovals is None:
        return np.inf
    isovals = isovals[0]
    coefs = fft(isovals)
    denominator = len(isovals) * (max(0, np.mean(isovals)) + n)
    if (not np.isfinite(denominator)) or denominator <= 0:
        return np.inf
    loss = np.abs(coefs[2]) / denominator
    return loss if np.isfinite(loss) else np.inf


def Isophote_Initialize_mean(IMG, results, options):
    """Fit global elliptical isophote to a galaxy image using FFT coefficients.

    Same as the default isophote initialization routine, except uses
    mean/std measures for low S/N applications.

    Parameters
    -----------------
    ap_fit_limit : float, default 2
      noise level out to which to extend the fit in units of pixel
      background noise level. Default is 2, smaller values will end
      fitting further out in the galaxy image.

    ap_isoinit_interpolate_method : string, default None
      Select method for flux interpolation while initializing
      isophotes. Options are 'lanczos', 'bicubic', and 'bilinear'.
      Default None uses the standard isophote extraction default.

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'psf fwhm'
    - 'center'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'init ellip': , # Ellipticity of the global fit (float)
         'init pa': ,# Position angle of the global fit (float)
         'init R': ,# Semi-major axis length of global fit (float)
         'auxfile initialize': # optional, message for aux file to record the global ellipticity and postition angle (string)

        }

    """

    ######################################################################
    # Initial attempt to find size of galaxy in image
    # based on when isophotes SB values start to get
    # close to the background noise level
    circ_ellipse_radii = [results["psf fwhm"]]
    allphase = []
    dat = IMG - results["background"]
    mask = results["mask"] if "mask" in results else None
    if not np.any(mask):
        mask = None
    interp_method = (
        options["ap_isoinit_interpolate_method"]
        if "ap_isoinit_interpolate_method" in options
        else None
    )

    r = circ_ellipse_radii[-1]
    while r < (len(IMG) / 2):
        r *= 1 + 0.2
        isovals = _extract_init_fft_samples(
            dat,
            r,
            {"ellip": 0.0, "pa": 0.0},
            results["center"],
            mask=mask,
            interp_method=interp_method,
        )
        if isovals is None:
            continue
        circ_ellipse_radii.append(r)
        coefs = fft(isovals[0])
        allphase.append(coefs[2])
        # Stop when at 3 times background noise
        if np.mean(isovals[0]) < (3 * results["background noise"]) and len(circ_ellipse_radii) > 4:
            break
    if len(allphase) == 0 or len(circ_ellipse_radii) < 2:
        raise ValueError("Could not initialize isophotes: no finite samples found.")

    logging.info("%s: init scale: %f pix" % (options["ap_name"], circ_ellipse_radii[-1]))
    # Find global position angle.
    phase = (
        -Angle_Median(np.angle(allphase[-5:])) / 2
    ) % np.pi  # (-np.angle(np.mean(allphase[-5:]))/2) % np.pi

    # Find global ellipticity
    test_ellip = np.linspace(0.05, 0.95, 15)
    init_radius, test_f2 = _select_init_radius(
        test_ellip,
        lambda e, d, r, p, c, n, m: _fitEllip_mean_loss(
            e, d, r, p, c, n, m, interp_method=interp_method
        ),
        dat,
        circ_ellipse_radii[-2],
        phase,
        results["center"],
        results["background noise"],
        mask,
        len(IMG) / 2,
    )
    if not np.any(np.isfinite(test_f2)):
        raise ValueError("Could not initialize isophotes: no finite ellipticity samples found.")
    # Find global ellipticity: second pass
    ellip = test_ellip[np.argmin(test_f2)]
    test_ellip = np.linspace(ellip - 0.05, ellip + 0.05, 15)
    init_radius, test_f2 = _select_init_radius(
        test_ellip,
        lambda e, d, r, p, c, n, m: _fitEllip_mean_loss(
            e, d, r, p, c, n, m, interp_method=interp_method
        ),
        dat,
        init_radius,
        phase,
        results["center"],
        results["background noise"],
        mask,
        len(IMG) / 2,
    )
    if not np.any(np.isfinite(test_f2)):
        raise ValueError("Could not initialize isophotes: no finite ellipticity samples found.")
    ellip = test_ellip[np.argmin(test_f2)]
    res = minimize(
        lambda e, d, r, p, c, n, msk: _finite_average(
                _fitEllip_mean_loss(
                    _x_to_eps(e[0]), d, r * m, p, c, n, msk, interp_method=interp_method
                )
                for m in np.linspace(0.8, 1.2, 5)
        ),
        x0=_inv_x_to_eps(ellip),
        args=(
            dat,
            init_radius,
            phase,
            results["center"],
            results["background noise"],
            mask,
        ),
        method="Nelder-Mead",
        options={
            "initial_simplex": [
                [_inv_x_to_eps(ellip) - 1 / 15],
                [_inv_x_to_eps(ellip) + 1 / 15],
            ]
        },
    )
    if res.success and np.isfinite(res.fun):
        logging.debug(
            "%s: using optimal ellipticity %.3f over grid ellipticity %.3f"
            % (options["ap_name"], _x_to_eps(res.x[0]), ellip)
        )
        ellip = _x_to_eps(res.x[0])

    # Compute the error on the parameters
    ######################################################################
    RR = np.linspace(
        init_radius - results["psf fwhm"],
        init_radius + results["psf fwhm"],
        10,
    )
    errallphase = []
    err_radii = []
    for rr in RR:
        isovals = _extract_init_fft_samples(
            dat,
            rr,
            {"ellip": 0.0, "pa": 0.0},
            results["center"],
            mask=mask,
            interp_method=interp_method,
        )
        if isovals is None:
            continue
        coefs = fft(isovals[0])
        errallphase.append(coefs[2])
        err_radii.append(rr)
    if len(errallphase) > 0 and np.isfinite(np.mean(errallphase)) and np.mean(errallphase) != 0:
        sample_pas = (-np.angle(1j * np.array(errallphase) / np.mean(errallphase)) / 2) % np.pi
        pa_err = np.std(sample_pas)
    else:
        sample_pas = np.array([])
        pa_err = np.nan
    res_multi = [
        minimize(
            lambda e, d, r, p, c, n, m: _fitEllip_mean_loss(
                _x_to_eps(e[0]), d, r, p, c, n, m, interp_method=interp_method
            ),
            x0=_inv_x_to_eps(ellip),
            args=(dat, rrp[0], rrp[1], results["center"], results["background noise"], mask),
            method="Nelder-Mead",
            options={
                "initial_simplex": [
                    [_inv_x_to_eps(ellip) - 1 / 15],
                    [_inv_x_to_eps(ellip) + 1 / 15],
                ]
            },
        )
        for rrp in zip(err_radii, sample_pas)
    ]
    ellip_samples = [
        _x_to_eps(rm.x[0])
        for rm in res_multi
        if rm.success and np.isfinite(rm.fun) and np.all(np.isfinite(rm.x))
    ]
    ellip_err = np.std(ellip_samples) if len(ellip_samples) > 0 else np.nan

    circ_ellipse_radii = np.array(circ_ellipse_radii)

    if "ap_doplot" in options and options["ap_doplot"]:

        ranges = [
            [
                max(0, int(results["center"]["x"] - circ_ellipse_radii[-1] * 1.5)),
                min(
                    dat.shape[1],
                    int(results["center"]["x"] + circ_ellipse_radii[-1] * 1.5),
                ),
            ],
            [
                max(0, int(results["center"]["y"] - circ_ellipse_radii[-1] * 1.5)),
                min(
                    dat.shape[0],
                    int(results["center"]["y"] + circ_ellipse_radii[-1] * 1.5),
                ),
            ],
        ]

        LSBImage(
            dat[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]],
            results["background noise"],
        )
        # plt.imshow(np.clip(dat[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]],a_min = 0, a_max = None),
        #            origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        plt.gca().add_patch(
            Ellipse(
                xy=(
                    results["center"]["x"] - ranges[0][0],
                    results["center"]["y"] - ranges[1][0],
                ),
                width=2 * circ_ellipse_radii[-1],
                height=2 * circ_ellipse_radii[-1] * (1.0 - ellip),
                angle=phase * 180 / np.pi,
                fill=False,
                linewidth=1,
                color="y",
            )
        )
        plt.plot(
            [results["center"]["x"] - ranges[0][0]],
            [results["center"]["y"] - ranges[1][0]],
            marker="x",
            markersize=3,
            color="r",
        )
        plt.tight_layout()
        if not ("ap_nologo" in options and options["ap_nologo"]):
            AddLogo(plt.gcf())
        plt.savefig(
            f"{options.get('ap_plotpath','')}initialize_ellipse_{options['ap_name']}.{options.get('ap_plot_extension', 'jpg')}",
            dpi=options["ap_plotdpi"] if "ap_plotdpi" in options else 300,
        )
        plt.close()

        fig, ax = plt.subplots(2, 1, figsize=(6, 6))
        ax[0].plot(
            circ_ellipse_radii[:-1],
            ((-np.angle(allphase) / 2) % np.pi) * 180 / np.pi,
            color="k",
        )
        ax[0].axhline(phase * 180 / np.pi, color="r")
        ax[0].axhline((phase + pa_err) * 180 / np.pi, color="r", linestyle="--")
        ax[0].axhline((phase - pa_err) * 180 / np.pi, color="r", linestyle="--")
        # ax[0].axvline(circ_ellipse_radii[-2], color = 'orange', linestyle = '--')
        ax[0].set_xlabel("Radius [pix]")
        ax[0].set_ylabel("FFT$_{1}$ phase [deg]")
        ax[1].plot(test_ellip, test_f2, color="k")
        ax[1].axvline(ellip, color="r")
        ax[1].axvline(ellip + ellip_err, color="r", linestyle="--")
        ax[1].axvline(ellip - ellip_err, color="r", linestyle="--")
        ax[1].set_xlabel("Ellipticity [1 - b/a]")
        ax[1].set_ylabel("Loss [FFT$_{2}$/med(flux)]")
        plt.tight_layout()
        if not ("ap_nologo" in options and options["ap_nologo"]):
            AddLogo(plt.gcf())
        plt.savefig(
            f"{options.get('ap_plotpath','')}initialize_ellipse_optimize_{options['ap_name']}.{options.get('ap_plot_extension', 'jpg')}",
            dpi=options["ap_plotdpi"] if "ap_plotdpi" in options else 300,
        )
        plt.close()

    auxmessage = "global ellipticity: %.3f +- %.3f, pa: %.3f +- %.3f deg, size: %f pix" % (
        ellip,
        ellip_err,
        PA_shift_convention(phase) * 180 / np.pi,
        pa_err * 180 / np.pi,
        init_radius,
    )
    return IMG, {
        "init ellip": ellip,
        "init ellip_err": ellip_err,
        "init pa": phase,
        "init pa_err": pa_err,
        "init R": init_radius,
        "auxfile initialize": auxmessage,
    }
