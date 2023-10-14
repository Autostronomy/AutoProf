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

    # Compare variability of flux values along isophotes
    ######################################################################
    use_center = results["center"]
    count_variable = 0
    count_initrelative = 0
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
                "ellip": results["init ellip"],  # fixme, use mask
                "pa": results["init pa"],
            },
            use_center,
        )
        isovals = _iso_extract(
            dat,
            checkson["R"][i],
            {"ellip": checkson["ellip"][i], "pa": checkson["pa"][i]},
            use_center,
        )
        coefs = fft(np.clip(isovals, a_max=np.quantile(isovals, 0.85), a_min=None))

        if np.median(isovals) < (iqr(isovals) - results["background noise"]):
            count_variable += 1
        if (
            (iqr(isovals) - results["background noise"])
            / (np.median(isovals) + results["background noise"])
        ) > (
            iqr(init_isovals) / (np.median(init_isovals) + results["background noise"])
        ):
            count_initrelative += 1
        f2_compare.append(
            np.sum(np.abs(coefs[2]))
            / (
                len(isovals)
                * (max(0, np.median(isovals)) + results["background noise"])
            )
        )
        f1_compare.append(
            np.abs(coefs[1])
            / (
                len(isovals)
                * (max(0, np.median(isovals)) + results["background noise"])
            )
        )

    f1_compare = np.array(f1_compare)
    f2_compare = np.array(f2_compare)
    if count_variable > (0.2 * len(checkson["R"])):
        logging.warning(
            "%s: Possible failed fit! flux values highly variable along isophotes"
            % options["ap_name"]
        )
        tests["isophote variability"] = False
    else:
        tests["isophote variability"] = True
    if count_initrelative > (0.5 * len(checkson["R"])):
        logging.warning(
            "%s: Possible failed fit! flux values highly variable relative to initialization"
            % options["ap_name"]
        )
        tests["initial fit compare"] = False
    else:
        tests["initial fit compare"] = True
    if (
        np.sum(f2_compare > 0.2) > (0.1 * len(checkson["R"]))
        or np.sum(f2_compare > 0.1) > (0.3 * len(checkson["R"]))
        or np.sum(f2_compare > 0.05) > (0.8 * len(checkson["R"]))
    ):
        logging.warning(
            "%s: Possible failed fit! poor convergence of FFT coefficients"
            % options["ap_name"]
        )
        tests["FFT coefficients"] = False
    else:
        tests["FFT coefficients"] = True
    if (
        np.sum(f1_compare > 0.2) > (0.1 * len(checkson["R"]))
        or np.sum(f1_compare > 0.1) > (0.3 * len(checkson["R"]))
        or np.sum(f1_compare > 0.05) > (0.8 * len(checkson["R"]))
    ):
        logging.warning(
            "%s: Possible failed fit! possible failed center or lopsided galaxy"
            % options["ap_name"]
        )
        tests["Light symmetry"] = False
    else:
        tests["Light symmetry"] = True

    res = {"checkfit": tests}
    for t in tests:
        res["auxfile checkfit %s" % t] = "checkfit %s: %s" % (
            t,
            "pass" if tests[t] else "fail",
        )
    return IMG, res
