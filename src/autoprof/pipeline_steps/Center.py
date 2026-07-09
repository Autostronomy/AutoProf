import numpy as np
import sys
import os

from ..autoprofutils.SharedFunctions import (
    _iso_extract,
    _iso_interpolate_radius,
    _has_enough_isophote_coverage,
    _interpolate_invalid_isophote_samples,
    AddLogo,
    Angle_Median,
    flux_to_sb,
)
from photutils.centroids import centroid_2dg, centroid_com, centroid_1dg
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy.fftpack import fft, ifft
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import logging
from copy import copy, deepcopy

__all__ = ("Center_Forced", "Center_2DGaussian", "Center_1DGaussian", "Center_OfMass", "Center_Peak", "Center_HillClimb", "Center_HillClimb_mean")


def _center_mask(dat, results, extra_mask=None):
    center_mask = np.logical_not(np.isfinite(dat))
    if results.get("mask", None) is not None:
        center_mask = np.logical_or(center_mask, results["mask"])
    if extra_mask is not None:
        center_mask = np.logical_or(center_mask, extra_mask)
    return center_mask


def Center_Forced(IMG, results, options):
    """Extracts previously fit center coordinates.

    Extracts the center coordinates from an aux file for a previous
    AutoProf fit. Can instead simply be given a set center value, just
    like other centering methods. A given center will override teh
    fitted aux file center.

    Parameters
    -----------------
    ap_guess_center : dict, default None
      user provided starting point for center fitting. Center should
      be formatted as:

      .. code-block:: python

        {'x':float, 'y': float}

      , where the floats are the center coordinates in pixels. If not
      given, Autoprof will default to a guess of the image center.

    ap_set_center : dict, default None
      user provided fixed center for rest of calculations. Center
      should be formatted as:

      .. code-block:: python

        {'x':float, 'y': float}

      , where the floats are the center coordinates in pixels. If not
      given, Autoprof will default to a guess of the image center.

    ap_forcing_profile : string, default None
      (required for forced photometry) file path to .prof file
      providing forced photometry PA and ellip values to apply to
      *ap_image_file*.

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'center': {'x': , # x coordinate of the center (pix)
                    'y': }, # y coordinate of the center (pix)

         'auxfile center': # optional, message for aux file to record galaxy center (string)
         'auxfile centeral sb': # optional, central surface brightness value (float)

        }

    """
    current_center = {"x": IMG.shape[1] / 2, "y": IMG.shape[0] / 2}
    dat = IMG - results["background"]
    if "ap_guess_center" in options:
        current_center = deepcopy(options["ap_guess_center"])
        logging.info(
            "%s: Center initialized by user: %s"
            % (options["ap_name"], str(current_center))
        )
    if "ap_set_center" in options:
        logging.info(
            "%s: Center set by user: %s"
            % (options["ap_name"], str(options["ap_set_center"]))
        )
        sb0 = _central_surface_brightness(dat, options["ap_set_center"], results, options)
        return IMG, {
            "center": deepcopy(options["ap_set_center"]),
            "auxfile central sb": "central surface brightness: %.4f mag arcsec^-2"
            % sb0,
        }

    try:
        with open(options["ap_forcing_profile"][:-4] + "aux", "r") as f:
            for line in f.readlines():
                if line[:6] == "center":
                    x_loc = line.find("x:")
                    y_loc = line.find("y:")
                    current_center = {
                        "x": float(line[x_loc + 3 : line.find("pix")]),
                        "y": float(line[y_loc + 3 : line.rfind("pix")]),
                    }
                    break
            else:
                logging.warning(
                    "%s: Forced center failed! Using image center (or guess)."
                    % options["ap_name"]
                )
    except:
        logging.warning(
            "%s: Forced center failed! Using image center (or guess)."
            % options["ap_name"]
        )
    sb0 = _central_surface_brightness(dat, current_center, results, options)
    return IMG, {
        "center": current_center,
        "auxfile center": "center x: %.2f pix, y: %.2f pix"
        % (current_center["x"], current_center["y"]),
        "auxfile central sb": "central surface brightness: %.4f mag arcsec^-2" % sb0,
    }


def Center_2DGaussian(IMG, results, options):
    """Find galaxy center with a 2D gaussian fit to the image..

    Compute the pixel location of the galaxy center by fitting a 2d
    Gaussian as implimented by the photutils package.

    Parameters
    -----------------
    ap_guess_center : dict, default None
      user provided starting point for center fitting. Center should
      be formatted as:

      .. code-block:: python

        {'x':float, 'y': float}

      , where the floats are the center coordinates in pixels. If not
      given, Autoprof will default to a guess of the image center.

    ap_set_center : dict, default None
      user provided fixed center for rest of calculations. Center
      should be formatted as:

      .. code-block:: python

        {'x':float, 'y': float}

      , where the floats are the center coordinates in pixels. If not
      given, Autoprof will default to a guess of the image center.

    ap_centeringring : int, default 50
      Size of ring to use when finding galaxy center, in units of
      PSF. Larger rings will give the 2D fit more data to work with
      and allow for the starting position to be further from the true
      galaxy center.  Smaller rings will include fewer spurious
      objects, and can stop the 2D fit from being distracted by larger
      nearby objects/galaxies.

    Notes
    ----------
    :References:
    - 'background'
    - 'psf fwhm'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'center': {'x': , # x coordinate of the center (pix)
                    'y': }, # y coordinate of the center (pix)

         'auxfile center': # optional, message for aux file to record galaxy center (string)

        }

    """

    current_center = {"x": IMG.shape[1] / 2, "y": IMG.shape[0] / 2}
    if "ap_guess_center" in options:
        current_center = deepcopy(options["ap_guess_center"])
        logging.info(
            "%s: Center initialized by user: %s"
            % (options["ap_name"], str(current_center))
        )
    if "ap_set_center" in options:
        logging.info(
            "%s: Center set by user: %s"
            % (options["ap_name"], str(options["ap_set_center"]))
        )
        return IMG, {"center": deepcopy(options["ap_set_center"])}

    dat = IMG - results["background"]

    # Create mask to focus centering algorithm on the center of the image
    ranges = [
        [
            max(
                0,
                int(
                    current_center["x"]
                    - (
                        options["ap_centeringring"]
                        if "ap_centeringring" in options
                        else 50
                    )
                    * results["psf fwhm"]
                ),
            ),
            min(
                IMG.shape[1],
                int(
                    current_center["x"]
                    + (
                        options["ap_centeringring"]
                        if "ap_centeringring" in options
                        else 50
                    )
                    * results["psf fwhm"]
                ),
            ),
        ],
        [
            max(
                0,
                int(
                    current_center["y"]
                    - (
                        options["ap_centeringring"]
                        if "ap_centeringring" in options
                        else 50
                    )
                    * results["psf fwhm"]
                ),
            ),
            min(
                IMG.shape[0],
                int(
                    current_center["y"]
                    + (
                        options["ap_centeringring"]
                        if "ap_centeringring" in options
                        else 50
                    )
                    * results["psf fwhm"]
                ),
            ),
        ],
    ]
    centralize_mask = np.ones(IMG.shape, dtype=bool)
    centralize_mask[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]] = False
    center_mask = _center_mask(dat, results, centralize_mask)

    try:
        if not np.any(np.logical_not(center_mask)):
            raise ValueError("all center pixels are masked")
        x, y = centroid_2dg(np.where(center_mask, 0.0, dat), mask=center_mask)
        if not np.all(np.isfinite([x, y])):
            raise ValueError("center is non-finite")
        current_center = {"x": x, "y": y}
    except Exception:
        logging.warning(
            "%s: 2D Gaussian center finding failed! using image center (or guess)."
            % options["ap_name"]
        )

    # Plot center value for diagnostic purposes
    if "ap_doplot" in options and options["ap_doplot"]:
        plt.imshow(
            np.clip(dat, a_min=0, a_max=None),
            origin="lower",
            cmap="Greys_r",
            norm=ImageNormalize(stretch=LogStretch()),
        )
        plt.plot(
            [current_center["x"]],
            [current_center["y"]],
            marker="x",
            markersize=10,
            color="y",
        )
        plt.savefig(
            f"{options.get('ap_plotpath','')}center_vis_{options['ap_name']}.{options.get('ap_plot_extension', 'jpg')}",
            dpi=options["ap_plotdpi"] if "ap_plotdpi" in options else 300,            
        )
        plt.close()
    logging.info(
        "%s Center found: x %.1f, y %.1f"
        % (options["ap_name"], current_center["x"], current_center["y"])
    )
    return IMG, {
        "center": current_center,
        "auxfile center": "center x: %.2f pix, y: %.2f pix"
        % (current_center["x"], current_center["y"]),
    }


def Center_1DGaussian(IMG, results, options):
    """Find galaxy center with many 1D gaussian fits to the image..

    Compute the pixel location of the galaxy center using a photutils
    method.  Looking at 100 seeing lengths around the center of the
    image (images should already be mostly centered), finds the galaxy
    center by fitting several 1d Gaussians.

    Parameters
    -----------------
    ap_guess_center : dict, default None
      user provided starting point for center fitting. Center should
      be formatted as:

      .. code-block:: python

        {'x':float, 'y': float}

      , where the floats are the center coordinates in pixels. If not
      given, Autoprof will default to a guess of the image center.

    ap_set_center : dict, default None
      user provided fixed center for rest of calculations. Center
      should be formatted as:

      .. code-block:: python

        {'x':float, 'y': float}

      , where the floats are the center coordinates in pixels. If not
      given, Autoprof will default to a guess of the image center.

    ap_centeringring : int, default 50
      Size of ring to use when finding galaxy center, in units of
      PSF. Larger rings will give the 1D fits more data to work with
      and allow for the starting position to be further from the true
      galaxy center.  Smaller rings will include fewer spurious
      objects, and can stop the 1D fits from being distracted by
      larger nearby objects/galaxies.

    Notes
    ----------
    :References:
    - 'background'
    - 'psf fwhm'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'center': {'x': , # x coordinate of the center (pix)
                    'y': }, # y coordinate of the center (pix)

         'auxfile center': # optional, message for aux file to record galaxy center (string)

        }

    """

    current_center = {"x": IMG.shape[1] / 2, "y": IMG.shape[0] / 2}
    if "ap_guess_center" in options:
        current_center = deepcopy(options["ap_guess_center"])
        logging.info(
            "%s: Center initialized by user: %s"
            % (options["ap_name"], str(current_center))
        )
    if "ap_set_center" in options:
        logging.info(
            "%s: Center set by user: %s"
            % (options["ap_name"], str(options["ap_set_center"]))
        )
        return IMG, {"center": deepcopy(options["ap_set_center"])}

    dat = IMG - results["background"]

    # Create mask to focus centering algorithm on the center of the image
    ranges = [
        [
            max(
                0,
                int(
                    current_center["x"]
                    - (
                        options["ap_centeringring"]
                        if "ap_centeringring" in options
                        else 50
                    )
                    * results["psf fwhm"]
                ),
            ),
            min(
                IMG.shape[1],
                int(
                    current_center["x"]
                    + (
                        options["ap_centeringring"]
                        if "ap_centeringring" in options
                        else 50
                    )
                    * results["psf fwhm"]
                ),
            ),
        ],
        [
            max(
                0,
                int(
                    current_center["y"]
                    - (
                        options["ap_centeringring"]
                        if "ap_centeringring" in options
                        else 50
                    )
                    * results["psf fwhm"]
                ),
            ),
            min(
                IMG.shape[0],
                int(
                    current_center["y"]
                    + (
                        options["ap_centeringring"]
                        if "ap_centeringring" in options
                        else 50
                    )
                    * results["psf fwhm"]
                ),
            ),
        ],
    ]
    centralize_mask = np.ones(IMG.shape, dtype=bool)
    centralize_mask[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]] = False
    center_mask = _center_mask(dat, results, centralize_mask)

    try:
        if not np.any(np.logical_not(center_mask)):
            raise ValueError("all center pixels are masked")
        x, y = centroid_1dg(np.where(center_mask, 0.0, dat), mask=center_mask)
        if not np.all(np.isfinite([x, y])):
            raise ValueError("center is non-finite")
        current_center = {"x": x, "y": y}
    except Exception:
        logging.warning(
            "%s: 1D Gaussian center finding failed! using image center (or guess)."
            % options["ap_name"]
        )

    # Plot center value for diagnostic purposes
    if "ap_doplot" in options and options["ap_doplot"]:
        plt.imshow(
            np.clip(dat, a_min=0, a_max=None),
            origin="lower",
            cmap="Greys_r",
            norm=ImageNormalize(stretch=LogStretch()),
        )
        plt.plot(
            [current_center["x"]],
            [current_center["y"]],
            marker="x",
            markersize=10,
            color="y",
        )
        plt.savefig(
            f"{options.get('ap_plotpath','')}center_vis_{options['ap_name']}.{options.get('ap_plot_extension', 'jpg')}",
            dpi=options["ap_plotdpi"] if "ap_plotdpi" in options else 300,            
        )
        plt.close()
    logging.info(
        "%s Center found: x %.1f, y %.1f"
        % (options["ap_name"], current_center["x"], current_center["y"])
    )
    return IMG, {
        "center": current_center,
        "auxfile center": "center x: %.2f pix, y: %.2f pix"
        % (current_center["x"], current_center["y"]),
    }


def Center_OfMass(IMG, results, options):
    """Find the light weighted galaxy center.

    Iteratively computes the light weighted centroid within a window,
    moves to the new center and computes the light weighted centroid
    again.  The size of the search area is 10PSF by default. The
    iterative process will continue until the center is updated by
    less than 1/10th of the PSF size or when too mny iterations have
    been reached.

    Parameters
    -----------------
    ap_guess_center : dict, default None
      user provided starting point for center fitting. Center should
      be formatted as:

      .. code-block:: python

        {'x':float, 'y': float}

      , where the floats are the center coordinates in pixels. If not
      given, Autoprof will default to a guess of the image center.

    ap_set_center : dict, default None
      user provided fixed center for rest of calculations. Center
      should be formatted as:

      .. code-block:: python

        {'x':float, 'y': float}

      , where the floats are the center coordinates in pixels. If not
      given, Autoprof will default to a guess of the image center.

    ap_centeringring : int, default 10
      Size of ring to use when finding galaxy center, in units of
      PSF. Larger rings will allow for the starting position to be
      further from the true galaxy center.  Smaller rings will include
      fewer spurious objects, and can stop the centroid from being
      distracted by larger nearby objects/galaxies.

    Notes
    ----------
    :References:
    - 'background'
    - 'psf fwhm'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'center': {'x': , # x coordinate of the center (pix)
                    'y': }, # y coordinate of the center (pix)

         'auxfile center': # optional, message for aux file to record galaxy center (string)
         'auxfile centeral sb': # optional, central surface brightness value (float)

        }

    """

    current_center = {"x": IMG.shape[1] / 2, "y": IMG.shape[0] / 2}
    dat = IMG - results["background"]
    if "ap_guess_center" in options:
        current_center = deepcopy(options["ap_guess_center"])
        logging.info(
            "%s: Center initialized by user: %s"
            % (options["ap_name"], str(current_center))
        )
    if "ap_set_center" in options:
        logging.info(
            "%s: Center set by user: %s"
            % (options["ap_name"], str(options["ap_set_center"]))
        )
        sb0 = _central_surface_brightness(dat, options["ap_set_center"], results, options)
        return IMG, {
            "center": deepcopy(options["ap_set_center"]),
            "auxfile central sb": "central surface brightness: %.4f mag arcsec^-2"
            % sb0,
        }

    searchring = int(
        (options["ap_centeringring"] if "ap_centeringring" in options else 10)
        * results["psf fwhm"]
    )
    center_mask = _center_mask(dat, results)
    N_updates = 0
    while N_updates < 100:
        N_updates += 1
        old_center = deepcopy(current_center)
        ranges = [
            [
                max(0, int(current_center["x"] - searchring / 2)),
                min(IMG.shape[1], int(current_center["x"] + searchring / 2)),
            ],
            [
                max(0, int(current_center["y"] - searchring / 2)),
                min(IMG.shape[0], int(current_center["y"] + searchring / 2)),
            ],
        ]
        chunk = dat[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]]
        chunk_mask = center_mask[
            ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]
        ]
        choose = np.logical_not(chunk_mask)
        if not np.any(choose):
            logging.warning(
                "%s: Center of mass failed because all center pixels are masked."
                % options["ap_name"]
            )
            break
        yy, xx = np.indices(chunk.shape)
        weights = np.where(choose, chunk, 0.0)
        denominator = np.sum(weights)
        if (not np.isfinite(denominator)) or denominator == 0:
            logging.warning(
                "%s: Center of mass failed because the center flux sum is invalid."
                % options["ap_name"]
            )
            current_center = old_center
            break
        new_center = {
            "x": ranges[0][0] + np.sum(weights * xx) / denominator,
            "y": ranges[1][0] + np.sum(weights * yy) / denominator,
        }
        if not np.all(np.isfinite([new_center["x"], new_center["y"]])):
            logging.warning(
                "%s: Center of mass failed because the center is non-finite."
                % options["ap_name"]
            )
            current_center = old_center
            break
        current_center = new_center
        if (
            abs(current_center["x"] - old_center["x"]) < 0.1 * results["psf fwhm"]
            and abs(current_center["y"] - old_center["y"]) < 0.1 * results["psf fwhm"]
        ):
            break

    sb0 = _central_surface_brightness(dat, current_center, results, options)
    return IMG, {
        "center": current_center,
        "auxfile center": "center x: %.2f pix, y: %.2f pix"
        % (current_center["x"], current_center["y"]),
        "auxfile central sb": "central surface brightness: %.4f mag arcsec^-2" % sb0,
    }


def Center_Peak(IMG, results, options):

    current_center = {"x": IMG.shape[1] / 2, "y": IMG.shape[0] / 2}
    dat = IMG - results["background"]
    if "ap_guess_center" in options:
        current_center = deepcopy(options["ap_guess_center"])
        logging.info(
            "%s: Center initialized by user: %s"
            % (options["ap_name"], str(current_center))
        )
    if "ap_set_center" in options:
        logging.info(
            "%s: Center set by user: %s"
            % (options["ap_name"], str(options["ap_set_center"]))
        )
        sb0 = _central_surface_brightness(dat, options["ap_set_center"], results, options)
        return IMG, {
            "center": deepcopy(options["ap_set_center"]),
            "auxfile central sb": "central surface brightness: %.4f mag arcsec^-2"
            % sb0,
        }

    searchring = int(
        (options["ap_centeringring"] if "ap_centeringring" in options else 10)
        * results["psf fwhm"]
    )
    ranges = [
        [
            max(0, int(current_center["x"] - searchring / 2)),
            min(IMG.shape[1], int(current_center["x"] + searchring / 2)),
        ],
        [
            max(0, int(current_center["y"] - searchring / 2)),
            min(IMG.shape[0], int(current_center["y"] + searchring / 2)),
        ],
    ]
    center_mask = _center_mask(dat, results)
    chunk = dat[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]]
    chunk_mask = center_mask[
        ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]
    ]
    yy, xx = np.indices(chunk.shape)
    xx = xx.flatten()
    yy = yy.flatten()
    choose = np.logical_not(chunk_mask.flatten())
    floor = max(results["background noise"] / 5, np.finfo(float).tiny)
    flux = np.clip(chunk.flatten()[choose], a_min=floor, a_max=None)

    try:
        if len(flux) < 9:
            raise ValueError("not enough unmasked pixels")
        A = np.array(
            [
                np.ones(xx[choose].shape),
                xx[choose],
                yy[choose],
                xx[choose] ** 2,
                yy[choose] ** 2,
                xx[choose] * yy[choose],
                xx[choose] * yy[choose] ** 2,
                yy[choose] * xx[choose] ** 2,
                xx[choose] ** 2 * yy[choose] ** 2,
            ]
        ).T
        poly2dfit = np.linalg.lstsq(A, np.log10(flux), rcond=None)
        new_center = {
            "x": -poly2dfit[0][1] / (2 * poly2dfit[0][3]) + ranges[0][0],
            "y": -poly2dfit[0][2] / (2 * poly2dfit[0][4]) + ranges[1][0],
        }
        if not np.all(np.isfinite([new_center["x"], new_center["y"]])):
            raise ValueError("center is non-finite")
        current_center = new_center
    except Exception:
        logging.warning(
            "%s: Peak center finding failed! using image center (or guess)."
            % options["ap_name"]
        )

    sb0 = _central_surface_brightness(dat, current_center, results, options)
    return IMG, {
        "center": current_center,
        "auxfile center": "center x: %.2f pix, y: %.2f pix"
        % (current_center["x"], current_center["y"]),
        "auxfile central sb": "central surface brightness: %.4f mag arcsec^-2" % sb0,
    }


def _central_surface_brightness(dat, center, results, options):
    isovals = _iso_extract(
        dat,
        0.0,
        {"ellip": 0.0, "pa": 0.0},
        center,
        mask=results.get("mask", None),
        rad_interp=_iso_interpolate_radius(options, results),
    )
    if len(isovals) == 0:
        return np.nan
    return flux_to_sb(
        isovals[0],
        options["ap_pixscale"],
        options["ap_zeropoint"] if "ap_zeropoint" in options else 22.5,
    )


def _extract_center_fft_samples(dat, radius, center, mask=None, **kwargs):
    kwargs.setdefault("interp_method", "bilinear")
    flux, theta, choose, _ = _iso_extract(
        dat,
        radius,
        {"ellip": 0.0, "pa": 0.0},
        center,
        more=True,
        mask=mask,
        return_choose=True,
        **kwargs,
    )
    if not _has_enough_isophote_coverage(theta, choose):
        return None
    return _interpolate_invalid_isophote_samples(flux, theta, choose)


def _center_sample_radii(psf_fwhm, searchring):
    sampleradii = np.linspace(1, searchring, searchring) * psf_fwhm / 2
    # Sub-pixel rings are dominated by interpolation rather than image structure.
    sampleradii = sampleradii[sampleradii >= 1.0]
    if len(sampleradii) == 0:
        return np.array([1.0])
    return sampleradii


def _hillclimb_loss(x, IMG, PSF, noise, rad_interp, mask=None):
    center_loss = 0
    valid_radii = 0
    for rr in range(3):
        RR = (rr + 1.0) * PSF / 2
        isovals = _extract_center_fft_samples(
            IMG,
            RR,
            {
                "x": np.clip(
                    x[0], a_min=np.ceil(3 + RR), a_max=np.floor(IMG.shape[1] - 4 - RR)
                ),
                "y": np.clip(
                    x[1], a_min=np.ceil(3 + RR), a_max=np.floor(IMG.shape[0] - 4 - RR)
                ),
            },
            mask=mask,
            rad_interp=rad_interp,
            interp_method="bilinear",
        )
        if isovals is None or len(isovals[0]) < 2:
            continue
        isovals = isovals[0]
        coefs = fft(isovals)
        denominator = len(isovals) * (max(0, np.median(isovals)) + noise)
        if (not np.isfinite(denominator)) or denominator <= 0:
            continue
        center_loss += np.abs(coefs[1]) / denominator
        valid_radii += 1
        if not np.isfinite(center_loss):
            return np.inf
    if valid_radii == 0:
        return np.inf
    return center_loss


def Center_HillClimb(IMG, results, options):
    """Follow locally increasing brightness (robust to PSF size objects) to find peak.

    Using 10 circular isophotes out to 10 times the PSF HWHM, the first FFT coefficient
    phases are averaged to find the direction of increasing flux. Flux values are sampled
    along this direction and a quadratic fit gives the maximum. This is iteratively
    repeated until the step size becomes very small.

    Parameters
    -----------------
    ap_guess_center : dict, default None
      user provided starting point for center fitting. Center should
      be formatted as:

      .. code-block:: python

        {'x':float, 'y': float}

      , where the floats are the center coordinates in pixels. If not
      given, Autoprof will default to a guess of the image center.

    ap_set_center : dict, default None
      user provided fixed center for rest of calculations. Center
      should be formatted as:

      .. code-block:: python

        {'x':float, 'y': float}

      , where the floats are the center coordinates in pixels. If not
      given, Autoprof will default to a guess of the image center.

    ap_centeringring : int, default 10
      Size of ring to use when finding galaxy center, in units of
      PSF HWHM. Larger rings will be robust to features (i.e., foreground
      stars), while smaller rings may be needed for small galaxies.

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'psf fwhm'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'center': {'x': , # x coordinate of the center (pix)
                    'y': }, # y coordinate of the center (pix)

         'auxfile center': # optional, message for aux file to record galaxy center (string)
         'auxfile centeral sb': # optional, central surface brightness value (float)

        }

    """

    current_center = {"x": IMG.shape[1] / 2, "y": IMG.shape[0] / 2}
    dat = IMG - results["background"]
    if "ap_guess_center" in options:
        current_center = deepcopy(options["ap_guess_center"])
        logging.info(
            "%s: Center initialized by user: %s"
            % (options["ap_name"], str(current_center))
        )
    if "ap_set_center" in options:
        logging.info(
            "%s: Center set by user: %s"
            % (options["ap_name"], str(options["ap_set_center"]))
        )
        sb0 = _central_surface_brightness(dat, options["ap_set_center"], results, options)
        return IMG, {
            "center": deepcopy(options["ap_set_center"]),
            "auxfile central sb": "central surface brightness: %.4f mag arcsec^-2"
            % sb0,
        }

    searchring = (
        int(options["ap_centeringring"]) if "ap_centeringring" in options else 10
    )
    sampleradii = _center_sample_radii(results["psf fwhm"], searchring)
    rad_interp = _iso_interpolate_radius(options, results)
    # Search rings can be far from the true center, so avoid nearest-neighbor sampling.
    search_rad_interp = np.inf

    track_centers = []
    small_update_count = 0
    total_count = 0
    refine_center = True
    while small_update_count <= 5 and total_count <= 100:
        total_count += 1
        phases = []
        isovals = []
        coefs = []
        sampled_radii = []
        for r in sampleradii:
            isovals_r = _extract_center_fft_samples(
                dat,
                r,
                current_center,
                mask=results.get("mask", None),
                rad_interp=search_rad_interp,
            )
            if isovals_r is None or len(isovals_r[0]) < 2:
                continue
            isovals.append(isovals_r)
            coefs.append(
                fft(
                    np.clip(
                        isovals[-1][0],
                        a_max=np.quantile(isovals[-1][0], 0.85),
                        a_min=None,
                    )
                )
            )
            phases.append((-np.angle(coefs[-1][1])) % (2 * np.pi))
            sampled_radii.append(r)
        if len(phases) == 0:
            logging.warning(
                "%s: Center finding stopped because all sampled isophotes were masked"
                % options["ap_name"]
            )
            refine_center = False
            break
        direction = Angle_Median(phases) % (2 * np.pi)
        if not np.isfinite(direction):
            logging.warning(
                "%s: Center finding stopped because the update direction is invalid"
                % options["ap_name"]
            )
            refine_center = False
            break
        levels = []
        level_locs = []
        for i, r in enumerate(sampled_radii):
            floc = np.argmin(np.abs((isovals[i][1] % (2 * np.pi)) - direction))
            rloc = np.argmin(
                np.abs(
                    (isovals[i][1] % (2 * np.pi)) - ((direction + np.pi) % (2 * np.pi))
                )
            )
            smooth = np.abs(ifft(coefs[i][: min(10, len(coefs[i]))], n=len(coefs[i])))
            levels.append(smooth[floc])
            level_locs.append(r)
            levels.insert(0, smooth[rloc])
            level_locs.insert(0, -r)
        try:
            p = np.polyfit(level_locs, levels, deg=2)
            if p[0] < 0 and len(levels) > 3:
                dist = np.clip(
                    -p[1] / (2 * p[0]), a_min=min(level_locs), a_max=max(level_locs)
                )
            else:
                dist = level_locs[np.argmax(levels)]
        except:
            dist = results["psf fwhm"]
        current_center["x"] += dist * np.cos(direction)
        current_center["y"] += dist * np.sin(direction)
        if abs(dist) < (0.5 * results["psf fwhm"]):
            small_update_count += 1
        else:
            small_update_count = 0
        track_centers.append([current_center["x"], current_center["y"]])

    if refine_center:
        # refine center
        ranges = [
            [
                max(0, int(current_center["x"] - results["psf fwhm"] * 5)),
                min(dat.shape[1], int(current_center["x"] + results["psf fwhm"] * 5)),
            ],
            [
                max(0, int(current_center["y"] - results["psf fwhm"] * 5)),
                min(dat.shape[0], int(current_center["y"] + results["psf fwhm"] * 5)),
            ],
        ]
        refine_mask = None
        if "mask" in results:
            refine_mask = results["mask"][
                ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]
            ]

        res = minimize(
            _hillclimb_loss,
            x0=[current_center["x"] - ranges[0][0], current_center["y"] - ranges[1][0]],
            args=(
                dat[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]],
                results["psf fwhm"],
                results["background noise"],
                rad_interp,
                refine_mask,
            ),
            method="Nelder-Mead",
        )
        if res.success and np.all(np.isfinite(res.x)) and np.isfinite(res.fun):
            current_center["x"] = res.x[0] + ranges[0][0]
            current_center["y"] = res.x[1] + ranges[1][0]
    track_centers.append([current_center["x"], current_center["y"]])

    sb0 = _central_surface_brightness(dat, current_center, results, options)
    return IMG, {
        "center": current_center,
        "auxfile center": "center x: %.2f pix, y: %.2f pix"
        % (current_center["x"], current_center["y"]),
        "auxfile central sb": "central surface brightness: %.4f mag arcsec^-2" % sb0,
    }


def _hillclimb_mean_loss(x, IMG, PSF, noise, rad_interp, mask=None):
    center_loss = 0
    valid_radii = 0
    for rr in range(3):
        isovals = _extract_center_fft_samples(
            IMG,
            (rr + 0.5) * PSF,
            {"x": x[0], "y": x[1]},
            rad_interp=rad_interp,
            mask=mask,
        )
        if isovals is None or len(isovals[0]) < 2:
            continue
        isovals = isovals[0]
        coefs = fft(isovals)
        denominator = len(isovals) * max(noise, np.mean(isovals))
        if (not np.isfinite(denominator)) or denominator <= 0:
            continue
        center_loss += np.abs(coefs[1]) / denominator
        valid_radii += 1
        if not np.isfinite(center_loss):
            return np.inf
    if valid_radii == 0:
        return np.inf
    return center_loss


def Center_HillClimb_mean(IMG, results, options):
    """Follow locally increasing brightness (robust to PSF size objects) to find peak.

    Using 10 circular isophotes out to 10 times the PSF HWHM, the
    first FFT coefficient phases are averaged to find the direction of
    increasing flux. Flux values are sampled along this direction and
    a quadratic fit gives the maximum. This is iteratively repeated
    until the step size becomes very small. This function is identical
    to :func:`~autoprofutils.Center.Center_HillClimb` except that all
    averages/scatters are mean/std based instead of median/iqr based.

    Parameters
    -----------------
    ap_guess_center : dict, default None
      user provided starting point for center fitting. Center should
      be formatted as:

      .. code-block:: python

        {'x':float, 'y': float}

      , where the floats are the center coordinates in pixels. If not
      given, Autoprof will default to a guess of the image center.

    ap_set_center : dict, default None
      user provided fixed center for rest of calculations. Center
      should be formatted as:

      .. code-block:: python

        {'x':float, 'y': float}

      , where the floats are the center coordinates in pixels. If not
      given, Autoprof will default to a guess of the image center.

    ap_centeringring : int, default 10
      Size of ring to use when finding galaxy center, in units of
      PSF HWHM. Larger rings will be robust to features (i.e., foreground
      stars), while smaller rings may be needed for small galaxies.

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'psf fwhm'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'center': {'x': , # x coordinate of the center (pix)
                    'y': }, # y coordinate of the center (pix)

         'auxfile center': # optional, message for aux file to record galaxy center (string)
         'auxfile centeral sb': # optional, central surface brightness value (float)

        }

    """
    current_center = {"x": IMG.shape[0] / 2, "y": IMG.shape[1] / 2}

    current_center = {"x": IMG.shape[1] / 2, "y": IMG.shape[0] / 2}
    if "ap_guess_center" in options:
        current_center = deepcopy(options["ap_guess_center"])
        logging.info(
            "%s: Center initialized by user: %s"
            % (options["ap_name"], str(current_center))
        )
    if "ap_set_center" in options:
        logging.info(
            "%s: Center set by user: %s"
            % (options["ap_name"], str(options["ap_set_center"]))
        )
        return IMG, {"center": deepcopy(options["ap_set_center"])}

    dat = IMG - results["background"]

    searchring = (
        int(options["ap_centeringring"]) if "ap_centeringring" in options else 10
    )
    sampleradii = _center_sample_radii(results["psf fwhm"], searchring)
    rad_interp = _iso_interpolate_radius(options, results)
    # Search rings can be far from the true center, so avoid nearest-neighbor sampling.
    search_rad_interp = np.inf

    track_centers = []
    small_update_count = 0
    total_count = 0
    refine_center = True
    while small_update_count <= 5 and total_count <= 100:
        total_count += 1
        phases = []
        isovals = []
        coefs = []
        sampled_radii = []
        for r in sampleradii:
            isovals_r = _extract_center_fft_samples(
                dat,
                r,
                current_center,
                mask=results.get("mask", None),
                rad_interp=search_rad_interp,
            )
            if isovals_r is None or len(isovals_r[0]) < 2:
                continue
            isovals.append(isovals_r)
            coefs.append(fft(isovals[-1][0]))
            phases.append((-np.angle(coefs[-1][1])) % (2 * np.pi))
            sampled_radii.append(r)
        if len(phases) == 0:
            logging.warning(
                "%s: Mean center finding stopped because all sampled isophotes were masked"
                % options["ap_name"]
            )
            refine_center = False
            break
        direction = Angle_Median(phases) % (2 * np.pi)
        if not np.isfinite(direction):
            logging.warning(
                "%s: Mean center finding stopped because the update direction is invalid"
                % options["ap_name"]
            )
            refine_center = False
            break
        levels = []
        level_locs = []
        for i, r in enumerate(sampled_radii):
            floc = np.argmin(np.abs(isovals[i][1] - direction))
            rloc = np.argmin(
                np.abs(isovals[i][1] - ((direction + np.pi) % (2 * np.pi)))
            )
            smooth = np.abs(ifft(coefs[i][: min(10, len(coefs[i]))], n=len(coefs[i])))
            levels.append(smooth[floc])
            level_locs.append(r)
            levels.insert(0, smooth[rloc])
            level_locs.insert(0, -r)
        try:
            p = np.polyfit(level_locs, levels, deg=2)
            if p[0] < 0 and len(levels) > 3:
                dist = np.clip(
                    -p[1] / (2 * p[0]), a_min=min(level_locs), a_max=max(level_locs)
                )
            else:
                dist = level_locs[np.argmax(levels)]
        except:
            dist = results["psf fwhm"]
        current_center["x"] += dist * np.cos(direction)
        current_center["y"] += dist * np.sin(direction)
        if abs(dist) < (0.25 * results["psf fwhm"]):
            small_update_count += 1
        else:
            small_update_count = 0
        track_centers.append([current_center["x"], current_center["y"]])

    if refine_center:
        # refine center
        res = minimize(
            _hillclimb_mean_loss,
            x0=[current_center["x"], current_center["y"]],
            args=(
                dat,
                results["psf fwhm"],
                results["background noise"],
                rad_interp,
                results.get("mask", None),
            ),
            method="Nelder-Mead",
        )
        if res.success and np.all(np.isfinite(res.x)) and np.isfinite(res.fun):
            current_center["x"] = res.x[0]
            current_center["y"] = res.x[1]

    return IMG, {
        "center": current_center,
        "auxfile center": "center x: %.2f pix, y: %.2f pix"
        % (current_center["x"], current_center["y"]),
    }
