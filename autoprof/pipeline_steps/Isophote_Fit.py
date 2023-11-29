import numpy as np
from scipy.stats import iqr
from scipy.fftpack import fft, ifft
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from time import time
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.isophote import EllipseSample, EllipseGeometry, Isophote, IsophoteList
from photutils.isophote import Ellipse as Photutils_Ellipse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from copy import copy, deepcopy
import logging
import sys
import os

from ..autoprofutils.SharedFunctions import (
    _iso_extract,
    _x_to_pa,
    _x_to_eps,
    _inv_x_to_eps,
    _inv_x_to_pa,
    Angle_TwoAngles_sin,
    Angle_TwoAngles_cos,
    Angle_Scatter,
    LSBImage,
    AddLogo,
    PA_shift_convention,
    autocolours,
)
from ..autoprofutils.Diagnostic_Plots import Plot_Isophote_Fit

__all__ = ("Photutils_Fit", "Isophote_Fit_FixedPhase", "Isophote_Fit_FFT_Robust", "Isophote_Fit_Forced", "Isophote_Fit_FFT_mean")

def Photutils_Fit(IMG, results, options):
    """Photutils elliptical isophote wrapper.

    This simply gives users access to the photutils isophote
    fitting method. See: `photutils
    <https://photutils.readthedocs.io/en/stable/isophote.html>`_ for
    more information.

    Notes
    ----------
    :References:
    - 'background'
    - 'center'
    - 'init R'
    - 'init ellip'
    - 'init pa'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'fit ellip': , # array of ellipticity values (ndarray)
         'fit pa': , # array of PA values (ndarray)
         'fit R': , # array of semi-major axis values (ndarray)
         'fit ellip_err': , # optional, array of ellipticity error values (ndarray)
         'fit pa_err': , # optional, array of PA error values (ndarray)
         'auxfile fitlimit': # optional, auxfile message (string)

        }
    """

    dat = IMG - results["background"]
    geo = EllipseGeometry(
        x0=results["center"]["x"],
        y0=results["center"]["y"],
        sma=results["init R"] / 2,
        eps=results["init ellip"],
        pa=results["init pa"],
    )
    ellipse = Photutils_Ellipse(dat, geometry=geo)

    isolist = ellipse.fit_image(fix_center=True, linear=False)
    res = {
        "fit R": isolist.sma[1:],
        "fit ellip": isolist.eps[1:],
        "fit ellip_err": isolist.ellip_err[1:],
        "fit pa": isolist.pa[1:],
        "fit pa_err": isolist.pa_err[1:],
        "fit photutils isolist": isolist,
        "auxfile fitlimit": "fit limit semi-major axis: %.2f pix" % isolist.sma[-1],
    }

    if "ap_doplot" in options and options["ap_doplot"]:
        Plot_Isophote_Fit(
            dat,
            res["fit R"],
            res["fit ellip"],
            res["fit pa"],
            res["fit ellip_err"],
            res["fit pa_err"],
            results,
            options,
        )

    return IMG, res


def Isophote_Fit_FixedPhase(IMG, results, options):
    """Simply applies fixed position angle and ellipticity at the initialization values.

    Parameters
    -----------------
    ap_scale : float, default 0.2
      growth scale when fitting isophotes, not the same as
      *ap_sample---scale*.

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
    - 'mask' (optional)
    - 'init ellip'
    - 'init pa'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'fit ellip': , # array of ellipticity values (ndarray)
         'fit pa': , # array of PA values (ndarray)
         'fit R': , # array of semi-major axis values (ndarray)
         'fit ellip_err': , # optional, array of ellipticity error values (ndarray)
         'fit pa_err': , # optional, array of PA error values (ndarray)
         'auxfile fitlimit': # optional, auxfile message (string)

        }

    """

    if "ap_scale" in options:
        scale = options["ap_scale"]
    else:
        scale = 0.2
    # subtract background from image during processing
    dat = IMG - results["background"]
    mask = results["mask"] if "mask" in results else None
    if not np.any(mask):
        mask = None

    # Determine sampling radii
    ######################################################################
    shrink = 0
    while shrink < 5:
        sample_radii = [max(1.0, results["psf fwhm"] / 2)]
        while sample_radii[-1] < (max(IMG.shape) / 2):
            isovals = _iso_extract(
                dat,
                sample_radii[-1],
                {"ellip": results["init ellip"], "pa": results["init pa"]},
                results["center"],
                more=False,
                mask=mask,
            )
            if (
                np.median(isovals)
                < (options["ap_fit_limit"] if "ap_fit_limit" in options else 2)
                * results["background noise"]
            ):
                break
            sample_radii.append(sample_radii[-1] * (1.0 + scale / (1.0 + shrink)))
        if len(sample_radii) < 15:
            shrink += 1
        else:
            break
    if shrink >= 5:
        raise Exception(
            "Unable to initialize ellipse fit, check diagnostic plots. Possible missed center."
        )
    ellip = np.ones(len(sample_radii)) * results["init ellip"]
    pa = np.ones(len(sample_radii)) * results["init pa"]
    logging.debug("%s: sample radii: %s" % (options["ap_name"], str(sample_radii)))

    res = {
        "fit ellip": ellip,
        "fit pa": pa,
        "fit R": sample_radii,
        "auxfile fitlimit": "fit limit semi-major axis: %.2f pix" % sample_radii[-1],
    }

    if "init ellip_err" in results:
        res["fit ellip_err"] = np.ones(len(sample_radii)) * results["init ellip_err"]
    if "init pa_err" in results:
        res["fit pa_err"] = np.ones(len(sample_radii)) * results["init pa_err"]

    return IMG, res


def _ellip_smooth(R, E, deg):
    model = make_pipeline(PolynomialFeatures(deg), HuberRegressor(epsilon=2.0))
    model.fit(np.log10(R).reshape(-1, 1), _inv_x_to_eps(E))
    return _x_to_eps(model.predict(np.log10(R).reshape(-1, 1)))


def _pa_smooth(R, PA, deg):

    model_s = make_pipeline(PolynomialFeatures(deg), HuberRegressor())
    model_c = make_pipeline(PolynomialFeatures(deg), HuberRegressor())
    model_c.fit(np.log10(R).reshape(-1, 1), np.cos(2 * PA))
    model_s.fit(np.log10(R).reshape(-1, 1), np.sin(2 * PA))
    pred_pa_s = np.clip(model_s.predict(np.log10(R).reshape(-1, 1)), a_min=-1, a_max=1)
    pred_pa_c = np.clip(model_c.predict(np.log10(R).reshape(-1, 1)), a_min=-1, a_max=1)

    #  np.arctan(pred_pa_s / pred_pa_c) + (np.pi * (pred_pa_c < 0))
    return ((np.arctan2(pred_pa_s, pred_pa_c)) % (2 * np.pi)) / 2


def _FFT_Robust_loss(
        dat, R, PARAMS, i, C, noise, mask=None, reg_scale=1.0, robust_clip=0.15, fit_coefs=None, name=""
):

    isovals = _iso_extract(
        dat,
        R[i],
        PARAMS[i],
        C,
        mask=mask,
        interp_mask=False if mask is None else True,
        interp_method="bicubic",
    )

    try:
        coefs = fft(np.clip(isovals, a_max=np.quantile(isovals, 1. - robust_clip), a_min=None))
    except:
        coefs = np.zeros(100)
        isovals = np.zeros(100)
        
    if fit_coefs is None:
        f2_loss = np.abs(coefs[2]) / (
            len(isovals) * (max(0, np.median(isovals)) + noise / np.sqrt(len(isovals)))
        )
    else:
        f2_loss = np.sum(np.abs(coefs[np.array(fit_coefs)])) / (
            len(fit_coefs)
            * len(isovals)
            * (max(0, np.median(isovals)) + noise / np.sqrt(len(isovals)))
        )

    reg_loss = 0
    if not PARAMS[i]["m"] is None:
        fmode_scale = 1.0 / len(PARAMS[i]["m"])
    if i < (len(R) - 1):
        reg_loss += abs(
            (PARAMS[i]["ellip"] - PARAMS[i + 1]["ellip"]) / (1 - PARAMS[i + 1]["ellip"])
        )
        reg_loss += abs(
            Angle_TwoAngles_sin(PARAMS[i]["pa"], PARAMS[i + 1]["pa"]) / (0.2)
        )
        if not PARAMS[i]["m"] is None:
            for m in range(len(PARAMS[i]["m"])):
                reg_loss += fmode_scale * abs(
                    (PARAMS[i]["Am"][m] - PARAMS[i + 1]["Am"][m]) / 0.2
                )
                reg_loss += fmode_scale * abs(
                    Angle_TwoAngles_cos(
                        PARAMS[i]["m"][m] * PARAMS[i]["Phim"][m],
                        PARAMS[i + 1]["m"][m] * PARAMS[i + 1]["Phim"][m],
                    )
                    / (PARAMS[i]["m"][m] * 0.1)
                )
        if not PARAMS[i]["C"] is None:
            reg_loss += abs(np.log10(PARAMS[i]["C"] / PARAMS[i + 1]["C"])) / 0.1
    if i > 0:
        reg_loss += abs(
            (PARAMS[i]["ellip"] - PARAMS[i - 1]["ellip"]) / (1 - PARAMS[i - 1]["ellip"])
        )
        reg_loss += abs(
            Angle_TwoAngles_sin(PARAMS[i]["pa"], PARAMS[i - 1]["pa"]) / (0.2)
        )
        if not PARAMS[i]["m"] is None:
            for m in range(len(PARAMS[i]["m"])):
                reg_loss += fmode_scale * abs(
                    (PARAMS[i]["Am"][m] - PARAMS[i - 1]["Am"][m]) / 0.2
                )
                reg_loss += fmode_scale * abs(
                    Angle_TwoAngles_cos(
                        PARAMS[i]["m"][m] * PARAMS[i]["Phim"][m],
                        PARAMS[i - 1]["m"][m] * PARAMS[i - 1]["Phim"][m],
                    )
                    / (PARAMS[i]["m"][m] * 0.1)
                )
        if not PARAMS[i]["C"] is None:
            reg_loss += abs(np.log10(PARAMS[i]["C"] / PARAMS[i - 1]["C"])) / 0.1

    return f2_loss * (1 + reg_loss * reg_scale)


def _FFT_Robust_Errors(
        dat, R, PARAMS, C, noise, mask=None, reg_scale=1.0, robust_clip=0.15, fit_coefs=None, name=""
):

    PA_err = np.zeros(len(R))
    E_err = np.zeros(len(R))
    for ri in range(len(R)):
        temp_fits = []
        for i in range(10):
            low_ri = max(0, ri - 1)
            high_ri = min(len(R) - 1, ri + 1)
            temp_fits.append(
                minimize(
                    lambda x: _FFT_Robust_loss(
                        dat,
                        [R[low_ri], R[ri] * (1 - 0.05 + i * 0.1 / 9), R[high_ri]],
                        [
                            PARAMS[low_ri],
                            {
                                "ellip": np.clip(x[0], 0, 0.999),
                                "pa": x[1] % np.pi,
                                "m": PARAMS[ri]["m"],
                                "Am": PARAMS[ri]["Am"],
                                "Phim": PARAMS[ri]["Phim"],
                            },
                            PARAMS[high_ri],
                        ],
                        1,
                        C,
                        noise,
                        mask=mask,
                        reg_scale=reg_scale,
                        robust_clip = robust_clip,
                        fit_coefs=fit_coefs,
                        name=name,
                    ),
                    x0=[PARAMS[ri]["ellip"], PARAMS[ri]["pa"]],
                    method="SLSQP",
                    options={"ftol": 0.001},
                ).x
            )
        temp_fits = np.array(temp_fits)
        E_err[ri] = iqr(np.clip(temp_fits[:, 0], 0, 1), rng=[16, 84]) / 2
        PA_err[ri] = (
            Angle_Scatter(2 * (temp_fits[:, 1] % np.pi)) / 4.0
        )  # multiply by 2 to get [0, 2pi] range
    return E_err, PA_err


def Isophote_Fit_FFT_Robust(IMG, results, options):
    """Fit elliptical isophotes to a galaxy image using FFT coefficients and regularization.

    The isophotal fitting routine simultaneously optimizes a
    collection of elliptical isophotes by minimizing the 2nd FFT
    coefficient power, regularized for robustness. A series of
    isophotes are constructed which grow geometrically until they
    begin to reach the background level.  Then the algorithm
    iteratively updates the position angle and ellipticity of each
    isophote individually for many rounds.  Each round updates every
    isophote in a random order.  Each round cycles between three
    options: optimizing position angle, ellipticity, or both.  To
    optimize the parameters, 5 values (pa, ellip, or both) are
    randomly sampled and the "loss" is computed.  The loss is a
    combination of the relative amplitude of the second FFT
    coefficient (compared to the median flux), and a regularization
    term.  The regularization term penalizes adjacent isophotes for
    having different position angle or ellipticity (using the l1
    norm).  Thus, all the isophotes are coupled and tend to fit
    smoothly varying isophotes.  When the optimization has completed
    three rounds without any isophotes updating, the profile is
    assumed to have converged.

    An uncertainty for each ellipticity and position angle value is
    determined by repeatedly re-optimizing each ellipse with slight
    adjustments to it's semi-major axis length (+- 5%). The standard
    deviation of the PA/ellipticity after repeated fitting gives the
    uncertainty.

    Parameters
    -----------------
    ap_scale : float, default 0.2
      growth scale when fitting isophotes, not the same as
      *ap_sample---scale*.

    ap_fit_limit : float, default 2
      noise level out to which to extend the fit in units of pixel
      background noise level. Default is 2, smaller values will end
      fitting further out in the galaxy image.

    ap_regularize_scale : float, default 1
      scale factor to apply to regularization coupling factor between
      isophotes.  Default of 1, larger values make smoother fits,
      smaller values give more chaotic fits.

    ap_isofit_robustclip : float, default 0.15
      quantile of flux values at which to clip when extracting values
      along an isophote. Clipping outlier values (such as very bright
      stars) while fitting isophotes allows for robust computation of
      FFT coefficients along an isophote.

    ap_isofit_losscoefs : tuple, default (2,)
      Tuple of FFT coefficients to use in optimization
      procedure. AutoProf will attemp to minimize the power in all
      listed FFT coefficients. Must be a tuple, not a list.

    ap_isofit_superellipse : bool, default False
      If True, AutoProf will fit superellipses instead of regular
      ellipses. A superellipse is typically used to represent
      boxy/disky isophotes. The variable controlling the transition
      from a rectangle to an ellipse to a four-armed-star like shape
      is C. A value of C = 2 represents an ellipse and is the starting
      point of the optimization.

    ap_isofit_fitcoefs : tuple, default None
      Tuple of FFT coefficients to use in fitting procedure. AutoProf
      will attemp to fit ellipses with these Fourier mode
      perturbations. Such perturbations allow for lopsided, boxy,
      disky, and other types of isophotes beyond straightforward
      ellipses. Must be a tuple, not a list. Note that AutoProf will
      first fit ellipses, then turn on the Fourier mode perturbations,
      thus the fitting time will always be longer.

    ap_isofit_fitcoefs_FFTinit : bool, default False
      If True, the coefficients for the Fourier modes fitted from
      ap_isofit_fitcoefs will be initialized using an FFT
      decomposition along fitted elliptical isophotes. This can
      improve the fit result, though it is less stable and so users
      should examine the results after fitting.

    ap_isofit_perturbscale_ellip : float, default 0.03
      Sampling scale for random adjustments to ellipticity made while
      optimizing isophotes. Smaller values will converge faster, but
      get stuck in local minima; larger values will escape local
      minima, but takes longer to converge.

    ap_isofit_perturbscale_pa : float, default 0.06
      Sampling scale for random adjustments to position angle made
      while optimizing isophotes. Smaller values will converge faster,
      but get stuck in local minima; larger values will escape local
      minima, but takes longer to converge.

    ap_isofit_iterlimitmax : int, default 300
      Maximum number of iterations (each iteration adjusts every
      isophote once) before automatically stopping optimization. For
      galaxies with lots of structure (ie detailed spiral arms) more
      iterations may be needed to fully fit the light distribution,
      but runtime will be longer.

    ap_isofit_iterlimitmin : int, default 0
      Minimum number of iterations before optimization is allowed to
      stop.

    ap_isofit_iterstopnochange : float, default 3
      Number of iterations with no updates to parameters before
      optimization procedure stops. Lower values will process galaxies
      faster, but may still be stuck in local minima, higher values
      are more likely to converge on the global minimum but can take a
      long time to run. Fractional values are allowed though not
      recomended.

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'psf fwhm'
    - 'center'
    - 'mask' (optional)
    - 'init ellip'
    - 'init pa'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'fit ellip': , # array of ellipticity values (ndarray)
         'fit pa': , # array of PA values (ndarray)
         'fit R': , # array of semi-major axis values (ndarray)
         'fit ellip_err': , # optional, array of ellipticity error values (ndarray)
         'fit pa_err': , # optional, array of PA error values (ndarray)
         'fit C': , # optional, superellipse scale parameter (ndarray)
         'fit Fmodes': , # optional, fitted Fourier mode indices (tuple)
         'fit Fmode A*': , # optional, fitted Fourier mode amplitudes, * for each index (ndarray)
         'fit Fmode Phi*': , # optional, fitted Fourier mode phases, * for each index (ndarray)
         'auxfile fitlimit': # optional, auxfile message (string)

        }

    """

    if "ap_scale" in options:
        scale = options["ap_scale"]
    else:
        scale = 0.2

    # subtract background from image during processing
    dat = IMG - results["background"]
    mask = results["mask"] if "mask" in results else None
    if not np.any(mask):
        mask = None

    # Determine sampling radii
    ######################################################################
    shrink = 0
    while shrink < 5:
        sample_radii = [max(1.0, results["psf fwhm"] / 2)]
        while sample_radii[-1] < (max(IMG.shape) / 2):
            isovals = _iso_extract(
                dat,
                sample_radii[-1],
                {"ellip": results["init ellip"], "pa": results["init pa"]},
                results["center"],
                more=False,
                mask=mask,
            )
            if (
                np.median(isovals)
                < (options["ap_fit_limit"] if "ap_fit_limit" in options else 2)
                * results["background noise"]
            ):
                break
            sample_radii.append(sample_radii[-1] * (1.0 + scale / (1.0 + shrink)))
        if len(sample_radii) < 15:
            shrink += 1
        else:
            break
    if shrink >= 5:
        raise Exception(
            "Unable to initialize ellipse fit, check diagnostic plots. Possible missed center."
        )
    ellip = np.ones(len(sample_radii)) * results["init ellip"]
    pa = np.ones(len(sample_radii)) * results["init pa"]
    logging.debug("%s: sample radii: %s" % (options["ap_name"], str(sample_radii)))
    # Fit isophotes
    ######################################################################
    perturb_scale = 0.03
    regularize_scale = (
        options["ap_regularize_scale"] if "ap_regularize_scale" in options else 1.0
    )
    robust_clip = (
        options["ap_isofit_robustclip"] if "ap_isofit_robustclip" in options else 0.15
    )
    N_perturb = 5
    fit_coefs = (
        options["ap_isofit_losscoefs"] if "ap_isofit_losscoefs" in options else None
    )
    fit_params = (
        options["ap_isofit_fitcoefs"] if "ap_isofit_fitcoefs" in options else None
    )
    fit_superellipse = (
        options["ap_isofit_superellipse"]
        if "ap_isofit_superellipse" in options
        else False
    )
    parameters = list(
        {
            "ellip": ellip[i],
            "pa": pa[i],
            "m": fit_params,
            "C": 2 if fit_superellipse else None,
            "Am": None if fit_params is None else np.zeros(len(fit_params)),
            "Phim": None if fit_params is None else np.zeros(len(fit_params)),
        }
        for i in range(len(ellip))
    )

    count = 0

    iterlimitmax = (
        options["ap_isofit_iterlimitmax"]
        if "ap_isofit_iterlimitmax" in options
        else 1000
    )
    iterlimitmin = (
        options["ap_isofit_iterlimitmin"] if "ap_isofit_iterlimitmin" in options else 0
    )
    iterstopnochange = (
        options["ap_isofit_iterstopnochange"]
        if "ap_isofit_iterstopnochange" in options
        else 3
    )
    count_nochange = 0
    use_center = copy(results["center"])
    I = np.array(range(len(sample_radii)))
    param_cycle = 2
    base_params = 2 + int(fit_superellipse)
    while count < iterlimitmax:
        # Periodically include logging message
        if count % 10 == 0:
            logging.debug("%s: count: %i" % (options["ap_name"], count))
        count += 1

        np.random.shuffle(I)
        N_perturb = int(1 + (10 / np.sqrt(count)))

        for i in I:
            perturbations = []
            perturbations.append(deepcopy(parameters))
            perturbations[-1][i]["loss"] = _FFT_Robust_loss(
                dat,
                sample_radii,
                perturbations[-1],
                i,
                use_center,
                results["background noise"],
                mask=mask,
                reg_scale=regularize_scale if count > 4 else 0,
                robust_clip = robust_clip,
                fit_coefs=fit_coefs,
                name=options["ap_name"],
            )
            for n in range(N_perturb):
                perturbations.append(deepcopy(parameters))
                if count % param_cycle == 0:
                    perturbations[-1][i]["ellip"] = _x_to_eps(
                        _inv_x_to_eps(perturbations[-1][i]["ellip"])
                        + np.random.normal(loc=0, scale=perturb_scale)
                    )
                elif count % param_cycle == 1:
                    perturbations[-1][i]["pa"] = (
                        perturbations[-1][i]["pa"]
                        + np.random.normal(loc=0, scale=np.pi * perturb_scale)
                    ) % np.pi
                elif (count % param_cycle) == 2 and not parameters[i]["C"] is None:
                    perturbations[-1][i]["C"] = 10 ** (
                        np.log10(perturbations[-1][i]["C"])
                        + np.random.normal(loc=0, scale=np.log10(1.0 + perturb_scale))
                    )
                elif count % param_cycle < (base_params + len(parameters[i]["m"])):
                    perturbations[-1][i]["Am"][
                        (count % param_cycle) - base_params
                    ] += np.random.normal(loc=0, scale=perturb_scale)
                elif count % param_cycle < (base_params + 2 * len(parameters[i]["m"])):
                    phim_index = (
                        (count % param_cycle) - base_params - len(parameters[i]["m"])
                    )
                    perturbations[-1][i]["Phim"][phim_index] = (
                        perturbations[-1][i]["Phim"][phim_index]
                        + np.random.normal(
                            loc=0,
                            scale=2
                            * np.pi
                            * perturb_scale
                            / parameters[i]["m"][phim_index],
                        )
                    ) % (2 * np.pi / parameters[i]["m"][phim_index])
                else:
                    raise Exception(
                        "Unrecognized optimization parameter id: %i"
                        % (count % param_cycle)
                    )
                perturbations[-1][i]["loss"] = _FFT_Robust_loss(
                    dat,
                    sample_radii,
                    perturbations[-1],
                    i,
                    use_center,
                    results["background noise"],
                    mask=mask,
                    reg_scale=regularize_scale if count > 4 else 0,
                    robust_clip = robust_clip,
                    fit_coefs=fit_coefs,
                    name=options["ap_name"],
                )

            best = np.argmin(list(p[i]["loss"] for p in perturbations))
            if best > 0:
                parameters = deepcopy(perturbations[best])
                del parameters[i]["loss"]
                count_nochange = 0
            else:
                count_nochange += 1
            if not (
                count_nochange < (iterstopnochange * (len(sample_radii) - 1))
                or count < iterlimitmin
            ):
                if param_cycle > 2 or (
                    parameters[i]["m"] is None and not fit_superellipse
                ):
                    break
                elif parameters[i]["m"] is None and fit_superellipse:
                    logging.info(
                        "%s: Started C fitting at iteration %i"
                        % (options["ap_name"], count)
                    )
                    param_cycle = 3
                    iterstopnochange = max(iterstopnochange, param_cycle)
                    count_nochange = 0
                    count = 0
                    if fit_coefs is None:
                        fit_coefs = (2, 4)
                else:
                    logging.info(
                        "%s: Started Fmode fitting at iteration %i"
                        % (options["ap_name"], count)
                    )
                    if fit_superellipse:
                        logging.info(
                            "%s: Started C fitting at iteration %i"
                            % (options["ap_name"], count)
                        )
                    param_cycle = base_params + 2 * len(parameters[i]["m"])
                    iterstopnochange = max(iterstopnochange, param_cycle)
                    count_nochange = 0
                    count = 0
                    if fit_coefs is None and not fit_params is None:
                        fit_coefs = fit_params
                        if not 2 in fit_coefs:
                            fit_coefs = tuple(sorted(set([2] + list(fit_coefs))))
                    if not parameters[i]["C"] is None and (
                        not "ap_isofit_losscoefs" in options
                        or options["ap_isofit_losscoefs"] is None
                    ):
                        fit_coefs = tuple(sorted(set([4] + list(fit_coefs))))
                    if (
                        "ap_isofit_fitcoefs_FFTinit" in options
                        and options["ap_isofit_fitcoefs_FFTinit"]
                    ):
                        for ii in I:
                            isovals = _iso_extract(
                                dat,
                                sample_radii[ii],
                                parameters[ii],
                                use_center,
                                mask=mask,
                                interp_mask=False if mask is None else True,
                                interp_method="bicubic",
                            )

                            if mask is None:
                                coefs = fft(
                                    np.clip(
                                        isovals,
                                        a_max=np.quantile(isovals, 0.85),
                                        a_min=None,
                                    )
                                )
                            else:
                                coefs = fft(
                                    np.clip(
                                        isovals,
                                        a_max=np.quantile(isovals, 0.9),
                                        a_min=None,
                                    )
                                )
                            for m in range(len(parameters[ii]["m"])):
                                parameters[ii]["Am"][m] = np.abs(
                                    coefs[parameters[ii]["m"][m]] / coefs[0]
                                ) * np.sign(np.angle(coefs[parameters[ii]["m"][m]]))
                                parameters[ii]["Phim"][m] = np.angle(
                                    coefs[parameters[ii]["m"][m]]
                                ) % (2 * np.pi)

        if not (
            count_nochange < (iterstopnochange * (len(sample_radii) - 1))
            or count < iterlimitmin
        ):
            break

    logging.info(
        "%s: Completed isohpote fit in %i itterations" % (options["ap_name"], count)
    )
    # Compute errors
    ######################################################################
    ellip_err, pa_err = _FFT_Robust_Errors(
        dat,
        sample_radii,
        parameters,
        use_center,
        results["background noise"],
        mask=mask,
        reg_scale=regularize_scale,
        robust_clip = robust_clip,
        fit_coefs=fit_coefs,
        name=options["ap_name"],
    )
    for i in range(len(ellip)):
        parameters[i]["ellip err"] = ellip_err[i]
        parameters[i]["pa err"] = pa_err[i]
    # Plot fitting results
    ######################################################################
    if "ap_doplot" in options and options["ap_doplot"]:
        Plot_Isophote_Fit(dat, sample_radii, parameters, results, options)

    res = {
        "fit ellip": np.array(
            list(parameters[i]["ellip"] for i in range(len(parameters)))
        ),
        "fit pa": np.array(list(parameters[i]["pa"] for i in range(len(parameters)))),
        "fit R": sample_radii,
        "fit ellip_err": ellip_err,
        "fit pa_err": pa_err,
        "auxfile fitlimit": "fit limit semi-major axis: %.2f pix" % sample_radii[-1],
    }
    if not fit_params is None:
        res.update({"fit Fmodes": fit_params})
        for m in range(len(fit_params)):
            res.update(
                {
                    "fit Fmode A%i"
                    % fit_params[m]: np.array(
                        list(parameters[i]["Am"][m] for i in range(len(parameters)))
                    ),
                    "fit Fmode Phi%i"
                    % fit_params[m]: np.array(
                        list(parameters[i]["Phim"][m] for i in range(len(parameters)))
                    ),
                }
            )
    if fit_superellipse:
        res.update(
            {
                "fit C": np.array(
                    list(parameters[i]["C"] for i in range(len(parameters)))
                )
            }
        )
    return IMG, res


def Isophote_Fit_Forced(IMG, results, options):
    """Read previously fit PA/ellipticity profile.

    Reads a .prof file and extracts the corresponding PA/ellipticity profile. The profile is extracted generically, so any csv file with columns for 'R', 'pa', 'ellip', and optionally 'pa_e' and 'ellip_e' will be able to create a forced fit. This can be used for testing purposes, such as selecting a specific isophote to extract or comparing AutoProf SB extraction methods with other softwares.

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
    - 'center'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'fit ellip': , # array of ellipticity values (ndarray)
         'fit pa': , # array of PA values (ndarray)
         'fit R': , # array of semi-major axis values (ndarray)
         'fit ellip_err': , # optional, array of ellipticity error values (ndarray)
         'fit pa_err': , # optional, array of PA error values (ndarray)

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

    force["pa"] = PA_shift_convention(np.array(force["pa"]), deg=True)

    if "ap_doplot" in options and options["ap_doplot"]:
        parameters = []
        for i in range(len(force["R"])):
            parameters.append({'ellip': force["ellip"][i], 'pa': force["pa"][i], 'C': None})
        Plot_Isophote_Fit(
            IMG - results["background"],
            np.array(force["R"]),
            parameters,
            results,
            options,
        )

    res = {
        "fit ellip": np.array(force["ellip"]),
        "fit pa": np.array(force["pa"]) * np.pi / 180,
        "fit R": list(np.array(force["R"]) / options["ap_pixscale"]),
    }
    if "ellip_e" in force and "pa_e" in force:
        res["fit ellip_err"] = np.array(force["ellip_e"])
        res["fit pa_err"] = np.array(force["pa_e"]) * np.pi / 180
    return IMG, res


######################################################################
def _FFT_mean_loss(dat, R, E, PA, i, C, noise, mask=None, reg_scale=1.0, name=""):

    isovals = _iso_extract(
        dat,
        R[i],
        {"ellip": E[i], "pa": PA[i]},
        C,
        mask=mask,
        interp_mask=False if mask is None else True,
    )

    if not np.all(np.isfinite(isovals)):
        logging.warning(
            "Failed to evaluate isophotal flux values, skipping this ellip/pa combination"
        )
        return np.inf

    coefs = fft(isovals)

    f2_loss = np.abs(coefs[2]) / (len(isovals) * (max(0, np.mean(isovals)) + noise))

    reg_loss = 0
    if i < (len(R) - 1):
        reg_loss += abs((E[i] - E[i + 1]) / (1 - E[i + 1]))
        reg_loss += abs(Angle_TwoAngles_sin(PA[i], PA[i + 1]) / (0.3))
    if i > 0:
        reg_loss += abs((E[i] - E[i - 1]) / (1 - E[i - 1]))
        reg_loss += abs(Angle_TwoAngles_sin(PA[i], PA[i - 1]) / (0.3))

    return f2_loss * (1 + reg_loss * reg_scale)


def Isophote_Fit_FFT_mean(IMG, results, options):
    """Fit elliptical isophotes to a galaxy image using FFT coefficients and regularization.

    Same as the standard isophote fitting routine, except uses less
    robust mean/std measures. This is only intended for low S/N data
    where pixels have low integer counts.

    Parameters
    -----------------
    ap_scale : float, default 0.2
      growth scale when fitting isophotes, not the same as
      *ap_sample---scale*.

    ap_fit_limit : float, default 2
      noise level out to which to extend the fit in units of pixel
      background noise level. Default is 2, smaller values will end
      fitting further out in the galaxy image.

    ap_regularize_scale : float, default 1
      scale factor to apply to regularization coupling factor between
      isophotes.  Default of 1, larger values make smoother fits,
      smaller values give more chaotic fits.

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'center'
    - 'psf fwhm'
    - 'init ellip'
    - 'init pa'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'fit ellip': , # array of ellipticity values (ndarray)
         'fit pa': , # array of PA values (ndarray)
         'fit R': , # array of semi-major axis values (ndarray)
         'fit ellip_err': , # optional, array of ellipticity error values (ndarray)
         'fit pa_err': , # optional, array of PA error values (ndarray)
         'auxfile fitlimit': # optional, auxfile message (string)

        }

    """

    if "ap_scale" in options:
        scale = options["ap_scale"]
    else:
        scale = 0.2

    # subtract background from image during processing
    dat = IMG - results["background"]
    mask = results["mask"] if "mask" in results else None
    if not np.any(mask):
        mask = None

    # Determine sampling radii
    ######################################################################
    shrink = 0
    while shrink < 5:
        sample_radii = [3 * results["psf fwhm"] / 2]
        while sample_radii[-1] < (max(IMG.shape) / 2):
            isovals = _iso_extract(
                dat,
                sample_radii[-1],
                {"ellip": results["init ellip"], "pa": results["init pa"]},
                results["center"],
                more=False,
                mask=mask,
            )
            if (
                np.mean(isovals)
                < (options["ap_fit_limit"] if "ap_fit_limit" in options else 1)
                * results["background noise"]
            ):
                break
            sample_radii.append(sample_radii[-1] * (1.0 + scale / (1.0 + shrink)))
        if len(sample_radii) < 15:
            shrink += 1
        else:
            break
    if shrink >= 5:
        raise Exception(
            "Unable to initialize ellipse fit, check diagnostic plots. Possible missed center."
        )
    ellip = np.ones(len(sample_radii)) * results["init ellip"]
    pa = np.ones(len(sample_radii)) * results["init pa"]
    logging.debug("%s: sample radii: %s" % (options["ap_name"], str(sample_radii)))

    # Fit isophotes
    ######################################################################
    perturb_scale = np.array([0.03, 0.06])
    regularize_scale = (
        options["ap_regularize_scale"] if "ap_regularize_scale" in options else 1.0
    )
    N_perturb = 5

    count = 0

    count_nochange = 0
    use_center = copy(results["center"])
    I = np.array(range(len(sample_radii)))
    while count < 300 and count_nochange < (3 * len(sample_radii)):
        # Periodically include logging message
        if count % 10 == 0:
            logging.debug("%s: count: %i" % (options["ap_name"], count))
        count += 1

        np.random.shuffle(I)
        for i in I:
            perturbations = []
            perturbations.append({"ellip": copy(ellip), "pa": copy(pa)})
            perturbations[-1]["loss"] = _FFT_mean_loss(
                dat,
                sample_radii,
                perturbations[-1]["ellip"],
                perturbations[-1]["pa"],
                i,
                use_center,
                results["background noise"],
                mask=mask,
                reg_scale=regularize_scale if count > 4 else 0,
                name=options["ap_name"],
            )
            for n in range(N_perturb):
                perturbations.append({"ellip": copy(ellip), "pa": copy(pa)})
                if count % 3 in [0, 1]:
                    perturbations[-1]["ellip"][i] = _x_to_eps(
                        _inv_x_to_eps(perturbations[-1]["ellip"][i])
                        + np.random.normal(loc=0, scale=perturb_scale[0])
                    )
                if count % 3 in [1, 2]:
                    perturbations[-1]["pa"][i] = (
                        perturbations[-1]["pa"][i]
                        + np.random.normal(loc=0, scale=perturb_scale[1])
                    ) % np.pi
                perturbations[-1]["loss"] = _FFT_mean_loss(
                    dat,
                    sample_radii,
                    perturbations[-1]["ellip"],
                    perturbations[-1]["pa"],
                    i,
                    use_center,
                    results["background noise"],
                    mask=mask,
                    reg_scale=regularize_scale if count > 4 else 0,
                    name=options["ap_name"],
                )

            best = np.argmin(list(p["loss"] for p in perturbations))
            if best > 0:
                ellip = copy(perturbations[best]["ellip"])
                pa = copy(perturbations[best]["pa"])
                count_nochange = 0
            else:
                count_nochange += 1

    logging.info(
        "%s: Completed isohpote fit in %i itterations" % (options["ap_name"], count)
    )
    # detect collapsed center
    ######################################################################
    for i in range(5):
        if (_inv_x_to_eps(ellip[i]) - _inv_x_to_eps(ellip[i + 1])) > 0.5:
            ellip[: i + 1] = ellip[i + 1]
            pa[: i + 1] = pa[i + 1]

    # Smooth ellip and pa profile
    ######################################################################
    smooth_ellip = copy(ellip)
    smooth_pa = copy(pa)
    ellip[:3] = min(ellip[:3])
    smooth_ellip = _ellip_smooth(sample_radii, smooth_ellip, 5)
    smooth_pa = _pa_smooth(sample_radii, smooth_pa, 5)

    if "ap_doplot" in options and options["ap_doplot"]:
        ranges = [
            [
                max(0, int(use_center["x"] - sample_radii[-1] * 1.2)),
                min(dat.shape[1], int(use_center["x"] + sample_radii[-1] * 1.2)),
            ],
            [
                max(0, int(use_center["y"] - sample_radii[-1] * 1.2)),
                min(dat.shape[0], int(use_center["y"] + sample_radii[-1] * 1.2)),
            ],
        ]
        LSBImage(
            dat[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]],
            results["background noise"],
        )
        # plt.imshow(np.clip(dat[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]],
        #                    a_min = 0,a_max = None), origin = 'lower', cmap = 'Greys', norm = ImageNormalize(stretch=LogStretch()))
        for i in range(len(sample_radii)):
            plt.gca().add_patch(
                Ellipse(
                    xy = (use_center["x"] - ranges[0][0], use_center["y"] - ranges[1][0]),
                    width = 2 * sample_radii[i],
                    height = 2 * sample_radii[i] * (1.0 - ellip[i]),
                    angle = pa[i] * 180 / np.pi,
                    fill=False,
                    linewidth=((i + 1) / len(sample_radii)) ** 2,
                    color="r",
                )
            )
        if not ("ap_nologo" in options and options["ap_nologo"]):
            AddLogo(plt.gcf())
        plt.savefig(
            f"{options.get('ap_plotpath','')}fit_ellipse_{options['ap_name']}.{options.get('ap_plot_extension', 'jpg')}",
            dpi=options["ap_plotdpi"] if "ap_plotdpi" in options else 300,
        )
        plt.close()

        plt.scatter(sample_radii, ellip, color="r", label="ellip")
        plt.scatter(sample_radii, pa / np.pi, color="b", label="pa/$np.pi$")
        show_ellip = _ellip_smooth(sample_radii, ellip, deg=5)
        show_pa = _pa_smooth(sample_radii, pa, deg=5)
        plt.plot(
            sample_radii,
            show_ellip,
            color="orange",
            linewidth=2,
            linestyle="--",
            label="smooth ellip",
        )
        plt.plot(
            sample_radii,
            show_pa / np.pi,
            color="purple",
            linewidth=2,
            linestyle="--",
            label="smooth pa/$np.pi$",
        )
        # plt.xscale('log')
        plt.legend()
        if not ("ap_nologo" in options and options["ap_nologo"]):
            AddLogo(plt.gcf())
        plt.savefig(
            f"{options.get('ap_plotpath','')}phaseprofile_{options['ap_name']}.{options.get('ap_plot_extension', 'jpg')}",
            dpi=options["ap_plotdpi"] if "ap_plotdpi" in options else 300,
        )
        plt.close()

    # Compute errors
    ######################################################################
    ellip_err = np.zeros(len(ellip))
    ellip_err[:2] = np.sqrt(np.sum((ellip[:4] - smooth_ellip[:4]) ** 2) / 4)
    ellip_err[-1] = np.sqrt(np.sum((ellip[-4:] - smooth_ellip[-4:]) ** 2) / 4)
    pa_err = np.zeros(len(pa))
    pa_err[:2] = np.sqrt(np.sum((pa[:4] - smooth_pa[:4]) ** 2) / 4)
    pa_err[-1] = np.sqrt(np.sum((pa[-4:] - smooth_pa[-4:]) ** 2) / 4)
    for i in range(2, len(pa) - 1):
        ellip_err[i] = np.sqrt(
            np.sum((ellip[i - 2 : i + 2] - smooth_ellip[i - 2 : i + 2]) ** 2) / 4
        )
        pa_err[i] = np.sqrt(
            np.sum((pa[i - 2 : i + 2] - smooth_pa[i - 2 : i + 2]) ** 2) / 4
        )

    res = {
        "fit ellip": ellip,
        "fit pa": pa,
        "fit R": sample_radii,
        "fit ellip_err": ellip_err,
        "fit pa_err": pa_err,
        "auxfile fitlimit": "fit limit semi-major axis: %.2f pix" % sample_radii[-1],
    }
    return IMG, res
