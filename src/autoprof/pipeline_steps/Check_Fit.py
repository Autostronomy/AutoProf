import numpy as np
from scipy.stats import iqr
from scipy.fftpack import fft, ifft
import logging
import sys
import os

from ..autoprofutils.SharedFunctions import (
    _iso_extract,
    _x_to_pa,
    _x_to_eps,
    _inv_x_to_eps,
    _inv_x_to_pa,
)

__all__ = ("Check_Fit",)

def Check_Fit(IMG, results, options):
    """Check for cases of failed isophote fits.

    A variety of check methods are applied to ensure that the fit has
    converged to a reasonable solution.  If a fit passes all of these
    checks then it is typically an acceptable fit.  However if it
    fails one or more of the checks then the fit likely either failed
    or the galaxy has strong non-axisymmetric features (and the fit
    itself may be acceptable).

    One check samples the fitted isophotes and looks for cases with
    high variability of flux values along the isophote.  This is done
    by comparing the interquartile range to the median flux, if the
    interquartile range is larger then that isophote is flagged.  If
    enough isophotes are flagged then the fit may have failed.

    A second check operates similarly, checking the second and fourth
    FFT coefficient amplitudes relative to the median flux.  If many
    of the isophotes have large FFT coefficients, or if a few of the
    isophotes have very large FFT coefficients then the fit is flagged
    as potentially failed.

    A third check is similar to the first, except that it compares the
    interquartile range from the fitted isophotes to those using just
    the global position angle and ellipticity values.

    A fourth check uses the first FFT coefficient to detect if the
    light is biased to one side of the galaxy. Typically this
    indicated either a failed center, or the galaxy has been disturbed
    and is not lopsided.

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'center'
    - 'init ellip'
    - 'init pa'
    - 'fit R' (optional)
    - 'fit ellip' (optional)
    - 'fit pa' (optional)
    - 'prof data' (optional)

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'checkfit': {'isophote variability': , # True if the test was passed, False if the test failed (bool)
                      'FFT coefficients': , # True if the test was passed, False if the test failed (bool)
                      'initial fit compare': , # True if the test was passed, False if the test failed (bool)
                      'Light symmetry': }, # True if the test was passed, False if the test failed (bool)

         'auxfile checkfit isophote variability': ,# optional aux file message for pass/fail of test (string)
         'auxfile checkfit FFT coefficients': ,# optional aux file message for pass/fail of test (string)
         'auxfile checkfit initial fit compare': ,# optional aux file message for pass/fail of test (string)
         'auxfile checkfit Light symmetry': ,# optional aux file message for pass/fail of test (string)

        }

    """
    tests = {}
    # subtract background from image during processing
    dat = IMG - results["background"]
    mask = results["mask"] if "mask" in results else None
    if mask is not None and not np.any(mask):
        mask = None

    # Compare variability of flux values along isophotes
    ######################################################################
    use_center = results["center"]
    count_variable = 0
    count_initrelative = 0
    count_checked = 0
    count_initrelative_checked = 0
    count_skipped = 0
    f2_compare = []
    f1_compare = []
    if "fit R" in results:
        checkson = {
            "R": results["fit R"],
            "pa": results["fit pa"],
            "ellip": results["fit ellip"],
        }
    else:
        checkson = {
            "R": results["prof data"]["R"],
            "pa": results["prof data"]["pa"],
            "ellip": results["prof data"]["ellip"],
        }

    for i in range(len(checkson["R"])):
        init_isovals = _iso_extract(
            dat,
            checkson["R"][i],
            {
                "ellip": results["init ellip"],
                "pa": results["init pa"],
            },
            use_center,
            mask=mask,
        )
        init_isovals = init_isovals[np.isfinite(init_isovals)]
        isovals = _iso_extract(
            dat,
            checkson["R"][i],
            {"ellip": checkson["ellip"][i], "pa": checkson["pa"][i]},
            use_center,
            mask=mask,
        )
        isovals = isovals[np.isfinite(isovals)]
        if len(isovals) <= 2:
            count_skipped += 1
            continue
        med_isovals = np.median(isovals)
        iqr_isovals = iqr(isovals)
        if not (np.isfinite(med_isovals) and np.isfinite(iqr_isovals)):
            count_skipped += 1
            continue
        coefs = fft(np.clip(isovals, a_max=np.quantile(isovals, 0.85), a_min=None))
        count_checked += 1

        if med_isovals < (iqr_isovals - results["background noise"]):
            count_variable += 1
        if len(init_isovals) > 0:
            med_init = np.median(init_isovals)
            iqr_init = iqr(init_isovals)
            denom_isovals = med_isovals + results["background noise"]
            denom_init = med_init + results["background noise"]
            if (
                np.isfinite(denom_isovals)
                and np.isfinite(denom_init)
                and denom_isovals != 0
                and denom_init != 0
            ):
                count_initrelative_checked += 1
                if ((iqr_isovals - results["background noise"]) / denom_isovals) > (
                    iqr_init / denom_init
                ):
                    count_initrelative += 1
        fft_denom = len(isovals) * (max(0, med_isovals) + results["background noise"])
        if not np.isfinite(fft_denom) or fft_denom <= 0:
            continue
        f2_compare.append(
            np.sum(np.abs(coefs[2]))
            / fft_denom
        )
        f1_compare.append(
            np.abs(coefs[1])
            / fft_denom
        )

    f1_compare = np.array(f1_compare)
    f2_compare = np.array(f2_compare)
    if count_skipped > 0:
        logging.info(
            "%s: checkfit skipped %i isophotes with too few usable samples"
            % (options["ap_name"], count_skipped)
        )
    if count_checked == 0:
        logging.warning(
            "%s: Possible failed fit! no isophotes had enough usable samples for checkfit"
            % options["ap_name"]
        )
        tests["isophote variability"] = False
        tests["initial fit compare"] = False
        tests["FFT coefficients"] = False
        tests["Light symmetry"] = False
    elif count_variable > (0.2 * count_checked):
        logging.warning(
            "%s: Possible failed fit! flux values highly variable along isophotes"
            % options["ap_name"]
        )
        tests["isophote variability"] = False
    else:
        tests["isophote variability"] = True
    if count_checked == 0:
        pass
    elif (
        count_initrelative_checked > 0
        and count_initrelative > (0.5 * count_initrelative_checked)
    ):
        logging.warning(
            "%s: Possible failed fit! flux values highly variable relative to initialization"
            % options["ap_name"]
        )
        tests["initial fit compare"] = False
    else:
        tests["initial fit compare"] = True
    if count_checked == 0:
        pass
    elif (
        len(f2_compare) == 0
        or np.sum(f2_compare > 0.2) > (0.1 * len(f2_compare))
        or np.sum(f2_compare > 0.1) > (0.3 * len(f2_compare))
        or np.sum(f2_compare > 0.05) > (0.8 * len(f2_compare))
    ):
        logging.warning(
            "%s: Possible failed fit! poor convergence of FFT coefficients"
            % options["ap_name"]
        )
        tests["FFT coefficients"] = False
    else:
        tests["FFT coefficients"] = True
    if count_checked == 0:
        pass
    elif (
        len(f1_compare) == 0
        or np.sum(f1_compare > 0.2) > (0.1 * len(f1_compare))
        or np.sum(f1_compare > 0.1) > (0.3 * len(f1_compare))
        or np.sum(f1_compare > 0.05) > (0.8 * len(f1_compare))
    ):
        logging.warning(
            "%s: Possible failed fit! possible failed center or lopsided galaxy"
            % options["ap_name"]
        )
        tests["Light symmetry"] = False
    else:
        tests["Light symmetry"] = True

    res = {"checkfit": tests}
    if count_skipped > 0:
        res["auxfile checkfit skipped"] = "checkfit skipped isophotes: %i" % count_skipped
    for t in tests:
        res["auxfile checkfit %s" % t] = "checkfit %s: %s" % (
            t,
            "pass" if tests[t] else "fail",
        )
    return IMG, res
