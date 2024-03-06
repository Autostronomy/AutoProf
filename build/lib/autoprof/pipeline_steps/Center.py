import numpy as np
import sys
import os

from ..autoprofutils.SharedFunctions import (
    _iso_extract,
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
        sb0 = flux_to_sb(
            _iso_extract(
                IMG - results["background"],
                0.0,
                {"ellip": 0.0, "pa": 0.0},
                options["ap_set_center"],
            )[0],
            options["ap_pixscale"],
            options["ap_zeropoint"] if "ap_zeropoint" in options else 22.5,
        )
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
    sb0 = flux_to_sb(
        _iso_extract(
            IMG - results["background"], 0.0, {"ellip": 0.0, "pa": 0.0}, current_center
        )[0],
        options["ap_pixscale"],
        options["ap_zeropoint"] if "ap_zeropoint" in options else 22.5,
    )
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

    try:
        x, y = centroid_2dg(IMG - results["background"], mask=centralize_mask)
        current_center = {"x": x, "y": y}
    except:
        logging.warning(
            "%s: 2D Gaussian center finding failed! using image center (or guess)."
            % options["ap_name"]
        )

    # Plot center value for diagnostic purposes
    if "ap_doplot" in options and options["ap_doplot"]:
        plt.imshow(
            np.clip(IMG - results["background"], a_min=0, a_max=None),
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
    logging.info("%s Center found: x %.1f, y %.1f" % (options["ap_name"], x, y))
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

    try:
        x, y = centroid_1dg(IMG - results["background"], mask=centralize_mask)
        current_center = {"x": x, "y": y}
    except:
        logging.warning(
            "%s: 2D Gaussian center finding failed! using image center (or guess)."
            % options["ap_name"]
        )

    # Plot center value for diagnostic purposes
    if "ap_doplot" in options and options["ap_doplot"]:
        plt.imshow(
            np.clip(IMG - results["background"], a_min=0, a_max=None),
            origin="lower",
            cmap="Greys_r",
            norm=ImageNormalize(stretch=LogStretch()),
        )
        plt.plot([y], [x], marker="x", markersize=10, color="y")
        plt.savefig(
            f"{options.get('ap_plotpath','')}center_vis_{options['ap_name']}.{options.get('ap_plot_extension', 'jpg')}",
            dpi=options["ap_plotdpi"] if "ap_plotdpi" in options else 300,            
        )
        plt.close()
    logging.info("%s Center found: x %.1f, y %.1f" % (options["ap_name"], x, y))
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
        sb0 = flux_to_sb(
            _iso_extract(dat, 0.0, {"ellip": 0.0, "pa": 0.0}, options["ap_set_center"])[
                0
            ],
            options["ap_pixscale"],
            options["ap_zeropoint"] if "ap_zeropoint" in options else 22.5,
        )
        return IMG, {
            "center": deepcopy(options["ap_set_center"]),
            "auxfile central sb": "central surface brightness: %.4f mag arcsec^-2"
            % sb0,
        }

    searchring = int(
        (options["ap_centeringring"] if "ap_centeringring" in options else 10)
        * results["psf fwhm"]
    )
    xx, yy = np.meshgrid(np.arange(searchring), np.arange(searchring))
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
        current_center = {
            "x": ranges[0][0]
            + np.sum(
                (dat[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]] * xx)
            )
            / np.sum(dat[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]]),
            "y": ranges[1][0]
            + np.sum(
                (dat[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]] * yy)
            )
            / np.sum(dat[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]]),
        }
        if (
            abs(current_center["x"] - old_center["x"]) < 0.1 * results["psf fwhm"]
            and abs(current_center["y"] - old_center["y"]) < 0.1 * results["psf fwhm"]
        ):
            break

    sb0 = flux_to_sb(
        _iso_extract(dat, 0.0, {"ellip": 0.0, "pa": 0.0}, current_center)[0],
        options["ap_pixscale"],
        options["ap_zeropoint"] if "ap_zeropoint" in options else 22.5,
    )
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
        sb0 = flux_to_sb(
            _iso_extract(dat, 0.0, {"ellip": 0.0, "pa": 0.0}, options["ap_set_center"])[
                0
            ],
            options["ap_pixscale"],
            options["ap_zeropoint"] if "ap_zeropoint" in options else 22.5,
        )
        return IMG, {
            "center": deepcopy(options["ap_set_center"]),
            "auxfile central sb": "central surface brightness: %.4f mag arcsec^-2"
            % sb0,
        }

    searchring = int(
        (options["ap_centeringring"] if "ap_centeringring" in options else 10)
        * results["psf fwhm"]
    )
    xx, yy = np.meshgrid(np.arange(searchring), np.arange(searchring))
    xx = xx.flatten()
    yy = yy.flatten()
    A = np.array(
        [
            np.ones(xx.shape),
            xx,
            yy,
            xx ** 2,
            yy ** 2,
            xx * yy,
            xx * yy ** 2,
            yy * xx ** 2,
            xx ** 2 * yy ** 2,
        ]
    ).T
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
    chunk = np.clip(
        dat[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]].T,
        a_min=results["background noise"] / 5,
        a_max=None,
    )

    poly2dfit = np.linalg.lstsq(A, np.log10(chunk.flatten()), rcond=None)
    current_center = {
        "x": -poly2dfit[0][2] / (2 * poly2dfit[0][4]) + ranges[0][0],
        "y": -poly2dfit[0][1] / (2 * poly2dfit[0][3]) + ranges[1][0],
    }

    sb0 = flux_to_sb(
        _iso_extract(dat, 0.0, {"ellip": 0.0, "pa": 0.0}, current_center)[0],
        options["ap_pixscale"],
        options["ap_zeropoint"] if "ap_zeropoint" in options else 22.5,
    )
    return IMG, {
        "center": current_center,
        "auxfile center": "center x: %.2f pix, y: %.2f pix"
        % (current_center["x"], current_center["y"]),
        "auxfile central sb": "central surface brightness: %.4f mag arcsec^-2" % sb0,
    }


def _hillclimb_loss(x, IMG, PSF, noise):
    center_loss = 0
    for rr in range(3):
        RR = (rr + 1.0) * PSF / 2
        isovals = _iso_extract(
            IMG,
            RR,
            {"ellip": 0.0, "pa": 0.0},
            {
                "x": np.clip(
                    x[0], a_min=np.ceil(3 + RR), a_max=np.floor(IMG.shape[1] - 4 - RR)
                ),
                "y": np.clip(
                    x[1], a_min=np.ceil(3 + RR), a_max=np.floor(IMG.shape[0] - 4 - RR)
                ),
            },
            more=False,
            rad_interp=10 * PSF,
            interp_method="lanczos",
            interp_window=3,
        )
        coefs = fft(isovals)
        center_loss += np.abs(coefs[1]) / (
            len(isovals) * (max(0, np.median(isovals)) + noise)
        )
    return center_loss


def Center_HillClimb(IMG, results, options):
    """Follow locally increasing brightness (robust to PSF size objects) to find peak.

    Using 10 circular isophotes out to 10 times the PSF length, the first FFT coefficient
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
      PSF. Larger rings will be robust to features (i.e., foreground
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
        sb0 = flux_to_sb(
            _iso_extract(dat, 0.0, {"ellip": 0.0, "pa": 0.0}, options["ap_set_center"], mask = results.get("mask", None))[
                0
            ],
            options["ap_pixscale"],
            options["ap_zeropoint"] if "ap_zeropoint" in options else 22.5,
        )
        return IMG, {
            "center": deepcopy(options["ap_set_center"]),
            "auxfile central sb": "central surface brightness: %.4f mag arcsec^-2"
            % sb0,
        }

    searchring = (
        int(options["ap_centeringring"]) if "ap_centeringring" in options else 10
    )
    sampleradii = np.linspace(1, searchring, searchring) * results["psf fwhm"]

    track_centers = []
    small_update_count = 0
    total_count = 0
    while small_update_count <= 5 and total_count <= 100:
        total_count += 1
        phases = []
        isovals = []
        coefs = []
        for r in sampleradii:
            isovals.append(
                _iso_extract(
                    dat, r, {"ellip": 0.0, "pa": 0.0}, current_center, more=True,
                    mask = results.get("mask", None)
                )
            )
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
        direction = Angle_Median(phases) % (2 * np.pi)
        levels = []
        level_locs = []
        for i, r in enumerate(sampleradii):
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

    res = minimize(
        _hillclimb_loss,
        x0=[current_center["x"] - ranges[0][0], current_center["y"] - ranges[1][0]],
        args=(
            dat[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]],
            results["psf fwhm"],
            results["background noise"],
        ),
        method="Nelder-Mead",
    )
    if res.success:
        current_center["x"] = res.x[0] + ranges[0][0]
        current_center["y"] = res.x[1] + ranges[1][0]
    track_centers.append([current_center["x"], current_center["y"]])

    sb0 = flux_to_sb(
        _iso_extract(dat, 0.0, {"ellip": 0.0, "pa": 0.0}, current_center, mask = results.get("mask", None))[0],
        options["ap_pixscale"],
        options["ap_zeropoint"] if "ap_zeropoint" in options else 22.5,
    )
    return IMG, {
        "center": current_center,
        "auxfile center": "center x: %.2f pix, y: %.2f pix"
        % (current_center["x"], current_center["y"]),
        "auxfile central sb": "central surface brightness: %.4f mag arcsec^-2" % sb0,
    }


def _hillclimb_mean_loss(x, IMG, PSF, noise):
    center_loss = 0
    for rr in range(3):
        isovals = _iso_extract(
            IMG,
            (rr + 0.5) * PSF,
            {"ellip": 0.0, "pa": 0.0},
            {"x": x[0], "y": x[1]},
            more=False,
            rad_interp=10 * PSF,
        )
        coefs = fft(isovals)
        center_loss += np.abs(coefs[1]) / (
            len(isovals) * (max(noise, np.mean(isovals)))
        )
    return center_loss


def Center_HillClimb_mean(IMG, results, options):
    """Follow locally increasing brightness (robust to PSF size objects) to find peak.

    Using 10 circular isophotes out to 10 times the PSF length, the
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
      PSF. Larger rings will be robust to features (i.e., foreground
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
    sampleradii = np.linspace(1, searchring, searchring) * results["psf fwhm"]

    track_centers = []
    small_update_count = 0
    total_count = 0
    while small_update_count <= 5 and total_count <= 100:
        total_count += 1
        phases = []
        isovals = []
        coefs = []
        for r in sampleradii:
            isovals.append(
                _iso_extract(
                    dat, r, {"ellip": 0.0, "pa": 0.0}, current_center, more=True
                )
            )
            coefs.append(fft(isovals[-1][0]))
            phases.append((-np.angle(coefs[-1][1])) % (2 * np.pi))
        direction = Angle_Median(phases) % (2 * np.pi)
        levels = []
        level_locs = []
        for i, r in enumerate(sampleradii):
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

    # refine center
    res = minimize(
        _hillclimb_mean_loss,
        x0=[current_center["x"], current_center["y"]],
        args=(dat, results["psf fwhm"], results["background noise"]),
        method="Nelder-Mead",
    )
    if res.success:
        current_center["x"] = res.x[0]
        current_center["y"] = res.x[1]

    return IMG, {
        "center": current_center,
        "auxfile center": "center x: %.2f pix, y: %.2f pix"
        % (current_center["x"], current_center["y"]),
    }
