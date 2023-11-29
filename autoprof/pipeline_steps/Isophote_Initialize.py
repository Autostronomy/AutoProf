import numpy as np
from scipy.fftpack import fft, ifft, dct, idct
from scipy.optimize import minimize
from scipy.stats import iqr
import sys
import os

from ..autoprofutils.SharedFunctions import (
    _iso_extract,
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
                            line[
                                line.find("pa:") + 3 : line.find("+-", line.find("pa:"))
                            ].strip()
                        ),
                        deg=True,
                    )
                    * np.pi
                    / 180
                )
                pa_err = (
                    float(
                        line[
                            line.find("+-", line.find("pa:")) + 2 : line.find("deg")
                        ].strip()
                    )
                    * np.pi
                    / 180
                )
                R = float(
                    line[
                        line.find("size:") + 5 : line.find("pix", line.find("size:"))
                    ].strip()
                )
                break

    auxmessage = (
        "global ellipticity: %.3f +- %.3f, pa: %.3f +- %.3f deg, size: %f pix"
        % (
            ellip,
            ellip_err,
            PA_shift_convention(pa) * 180 / np.pi,
            pa_err * 180 / np.pi,
            R,
        )
    )

    return IMG, {
        "init ellip": ellip,
        "init ellip_err": ellip_err,
        "init pa": pa,
        "init pa_err": pa_err,
        "init R": R,
        "auxfile initialize": auxmessage,
    }


def _fitEllip_loss(e, dat, r, p, c, n, m):
    isovals = _iso_extract(
        dat,
        r,
        {"ellip": e, "pa": p},
        c,
        sigmaclip=True,
        sclip_nsigma=3,
        mask=m,
        interp_mask=True,
    )
    coefs = fft(np.clip(isovals, a_max=np.quantile(isovals, 0.85), a_min=None))
    return (iqr(isovals, rng=[16, 84]) / 2 + np.abs(coefs[2]) / len(isovals)) / (
        max(0, np.median(isovals)) + n
    )


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

    while circ_ellipse_radii[-1] < (len(IMG) / 2):
        circ_ellipse_radii.append(circ_ellipse_radii[-1] * (1 + 0.2))
        isovals = _iso_extract(
            dat,
            circ_ellipse_radii[-1],
            {"ellip": 0.0, "pa": 0.0},
            results["center"],
            more=True,
            mask=mask,
            sigmaclip=True,
            sclip_nsigma=3,
            interp_mask=True,
        )
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
    logging.info(
        "%s: init scale: %f pix" % (options["ap_name"], circ_ellipse_radii[-1])
    )
    # Find global position angle.
    phase = (-Angle_Median(np.angle(allphase[-5:])) / 2) % np.pi
    if "ap_isoinit_pa_set" in options:
        phase = PA_shift_convention(options["ap_isoinit_pa_set"] * np.pi / 180)

    # Find global ellipticity
    test_ellip = np.linspace(0.05, 0.95, 15)
    test_f2 = []
    for e in test_ellip:
        test_f2.append(
            sum(
                list(
                    _fitEllip_loss(
                        e,
                        dat,
                        circ_ellipse_radii[-2] * m,
                        phase,
                        results["center"],
                        results["background noise"],
                        mask,
                    )
                    for m in np.linspace(0.8, 1.2, 5)
                )
            )
        )
    # Find global ellipticity: second pass
    ellip = test_ellip[np.argmin(test_f2)]
    test_ellip = np.linspace(ellip - 0.05, ellip + 0.05, 15)
    test_f2 = []
    for e in test_ellip:
        test_f2.append(
            sum(
                list(
                    _fitEllip_loss(
                        e,
                        dat,
                        circ_ellipse_radii[-2] * m,
                        phase,
                        results["center"],
                        results["background noise"],
                        mask,
                    )
                    for m in np.linspace(0.8, 1.2, 5)
                )
            )
        )
    ellip = test_ellip[np.argmin(test_f2)]
    res = minimize(
        lambda e, d, r, p, c, n, msk: sum(
            list(
                _fitEllip_loss(_x_to_eps(e[0]), d, r * m, p, c, n, msk)
                for m in np.linspace(0.8, 1.2, 5)
            )
        ),
        x0=_inv_x_to_eps(ellip),
        args=(
            dat,
            circ_ellipse_radii[-2],
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
    if res.success:
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
        circ_ellipse_radii[-2] - results["psf fwhm"],
        circ_ellipse_radii[-2] + results["psf fwhm"],
        10,
    )
    errallphase = []
    for rr in RR:
        isovals = _iso_extract(
            dat,
            rr,
            {"ellip": 0.0, "pa": 0.0},
            results["center"],
            more=True,
            sigmaclip=True,
            sclip_nsigma=3,
            interp_mask=True,
        )
        coefs = fft(isovals[0])
        errallphase.append(coefs[2])
    sample_pas = (
        -np.angle(1j * np.array(errallphase) / np.mean(errallphase)) / 2
    ) % np.pi
    pa_err = iqr(sample_pas, rng=[16, 84]) / 2
    res_multi = map(
        lambda rrp: minimize(
            lambda e, d, r, p, c, n, m: _fitEllip_loss(
                _x_to_eps(e[0]), d, r, p, c, n, m
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
        ),
        zip(RR, sample_pas),
    )
    ellip_err = iqr(list(_x_to_eps(rm.x[0]) for rm in res_multi), rng=[16, 84]) / 2

    circ_ellipse_radii = np.array(circ_ellipse_radii)

    if "ap_doplot" in options and options["ap_doplot"]:
        Plot_Isophote_Init_Ellipse(
            dat, circ_ellipse_radii, ellip, phase, results, options
        )
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

    auxmessage = (
        "global ellipticity: %.3f +- %.3f, pa: %.3f +- %.3f deg, size: %f pix"
        % (
            ellip,
            ellip_err,
            PA_shift_convention(phase) * 180 / np.pi,
            pa_err * 180 / np.pi,
            circ_ellipse_radii[-2],
        )
    )
    return IMG, {
        "init ellip": ellip,
        "init ellip_err": ellip_err,
        "init pa": phase,
        "init pa_err": pa_err,
        "init R": circ_ellipse_radii[-2],
        "auxfile initialize": auxmessage,
    }


def _fitEllip_mean_loss(e, dat, r, p, c, n):
    isovals = _iso_extract(dat, r, {"ellip": e, "pa": p}, c)
    coefs = fft(isovals)
    return np.abs(coefs[2]) / (len(isovals) * (max(0, np.mean(isovals)) + n))


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

    while circ_ellipse_radii[-1] < (len(IMG) / 2):
        circ_ellipse_radii.append(circ_ellipse_radii[-1] * (1 + 0.2))
        isovals = _iso_extract(
            dat,
            circ_ellipse_radii[-1],
            {"ellip": 0.0, "pa": 0.0},
            results["center"],
            more=True,
        )
        coefs = fft(isovals[0])
        allphase.append(coefs[2])
        # Stop when at 3 times background noise
        if (
            np.mean(isovals[0]) < (3 * results["background noise"])
            and len(circ_ellipse_radii) > 4
        ):
            break
    logging.info(
        "%s: init scale: %f pix" % (options["ap_name"], circ_ellipse_radii[-1])
    )
    # Find global position angle.
    phase = (
        -Angle_Median(np.angle(allphase[-5:])) / 2
    ) % np.pi  # (-np.angle(np.mean(allphase[-5:]))/2) % np.pi

    # Find global ellipticity
    test_ellip = np.linspace(0.05, 0.95, 15)
    test_f2 = []
    for e in test_ellip:
        test_f2.append(
            sum(
                list(
                    _fitEllip_mean_loss(
                        e,
                        dat,
                        circ_ellipse_radii[-2] * m,
                        phase,
                        results["center"],
                        results["background noise"],
                    )
                    for m in np.linspace(0.8, 1.2, 5)
                )
            )
        )
    # Find global ellipticity: second pass
    ellip = test_ellip[np.argmin(test_f2)]
    test_ellip = np.linspace(ellip - 0.05, ellip + 0.05, 15)
    test_f2 = []
    for e in test_ellip:
        test_f2.append(
            sum(
                list(
                    _fitEllip_mean_loss(
                        e,
                        dat,
                        circ_ellipse_radii[-2] * m,
                        phase,
                        results["center"],
                        results["background noise"],
                    )
                    for m in np.linspace(0.8, 1.2, 5)
                )
            )
        )
    ellip = test_ellip[np.argmin(test_f2)]
    res = minimize(
        lambda e, d, r, p, c, n: sum(
            list(
                _fitEllip_mean_loss(_x_to_eps(e[0]), d, r * m, p, c, n)
                for m in np.linspace(0.8, 1.2, 5)
            )
        ),
        x0=_inv_x_to_eps(ellip),
        args=(
            dat,
            circ_ellipse_radii[-2],
            phase,
            results["center"],
            results["background noise"],
        ),
        method="Nelder-Mead",
        options={
            "initial_simplex": [
                [_inv_x_to_eps(ellip) - 1 / 15],
                [_inv_x_to_eps(ellip) + 1 / 15],
            ]
        },
    )
    if res.success:
        logging.debug(
            "%s: using optimal ellipticity %.3f over grid ellipticity %.3f"
            % (options["ap_name"], _x_to_eps(res.x[0]), ellip)
        )
        ellip = _x_to_eps(res.x[0])

    # Compute the error on the parameters
    ######################################################################
    RR = np.linspace(
        circ_ellipse_radii[-2] - results["psf fwhm"],
        circ_ellipse_radii[-2] + results["psf fwhm"],
        10,
    )
    errallphase = []
    for rr in RR:
        isovals = _iso_extract(
            dat, rr, {"ellip": 0.0, "pa": 0.0}, results["center"], more=True
        )
        coefs = fft(isovals[0])
        errallphase.append(coefs[2])
    sample_pas = (
        -np.angle(1j * np.array(errallphase) / np.mean(errallphase)) / 2
    ) % np.pi
    pa_err = np.std(sample_pas)
    res_multi = map(
        lambda rrp: minimize(
            lambda e, d, r, p, c, n: _fitEllip_mean_loss(
                _x_to_eps(e[0]), d, r, p, c, n
            ),
            x0=_inv_x_to_eps(ellip),
            args=(dat, rrp[0], rrp[1], results["center"], results["background noise"]),
            method="Nelder-Mead",
            options={
                "initial_simplex": [
                    [_inv_x_to_eps(ellip) - 1 / 15],
                    [_inv_x_to_eps(ellip) + 1 / 15],
                ]
            },
        ),
        zip(RR, sample_pas),
    )
    ellip_err = np.std(list(_x_to_eps(rm.x[0]) for rm in res_multi))

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
                xy = (
                    results["center"]["x"] - ranges[0][0],
                    results["center"]["y"] - ranges[1][0],
                ),
                width = 2 * circ_ellipse_radii[-1],
                height = 2 * circ_ellipse_radii[-1] * (1.0 - ellip),
                angle = phase * 180 / np.pi,
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

    auxmessage = (
        "global ellipticity: %.3f +- %.3f, pa: %.3f +- %.3f deg, size: %f pix"
        % (
            ellip,
            ellip_err,
            PA_shift_convention(phase) * 180 / np.pi,
            pa_err * 180 / np.pi,
            circ_ellipse_radii[-2],
        )
    )
    return IMG, {
        "init ellip": ellip,
        "init ellip_err": ellip_err,
        "init pa": phase,
        "init pa_err": pa_err,
        "init R": circ_ellipse_radii[-2],
        "auxfile initialize": auxmessage,
    }
