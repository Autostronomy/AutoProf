from photutils import DAOStarFinder, IRAFStarFinder
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy.stats import mode, iqr
import logging
import sys
import os

from ..autoprofutils.SharedFunctions import Read_Image, LSBImage, AddLogo, StarFind

__all__ = ("Bad_Pixel_Mask", "Mask_Segmentation_Map", "Star_Mask_IRAF", "Star_Mask")

def Bad_Pixel_Mask(IMG, results, options):
    """Simple masking routine to clip pixels based on thresholds.

    Creates a mask image using user provided limits on highest/lowest
    pixels values allowed. Also users can reject pixels with a
    specific value. This can be used on its own, or in combination
    with other masking routines. Multiple Mask calls with perform
    boolean-or operation.

    Parameters
    -----------------
    ap_badpixel_high : float, default None
      flux value that corresponds to a saturated pixel or bad pixel
      flag, all values above *ap_badpixel_high* will be masked if
      using the *Bad_Pixel_Mask* pipeline method.

    ap_badpixel_low : float, default None
      flux value that corresponds to a bad pixel flag, all values
      below *ap_badpixel_low* will be masked if using the
      *Bad_Pixel_Mask* pipeline method.

    ap_badpixel_exact : float, default None
      flux value that corresponds to a precise bad pixel flag, all
      values equal to *ap_badpixel_exact* will be masked if using the
      *Bad_Pixel_Mask* pipeline method.

    See Also
    --------
    ap_savemask : bool, default False
      indicates if the mask should be saved after fitting

    Notes
    ----------
    :References:
    - 'mask' (optional)

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'mask':  # 2d mask image with boolean datatype (ndarray)
        }

    """

    Mask = np.zeros(IMG.shape, dtype=bool)
    if "ap_badpixel_high" in options:
        Mask[IMG >= options["ap_badpixel_high"]] = True
    if "ap_badpixel_low" in options:
        Mask[IMG <= options["ap_badpixel_low"]] = True
    if "ap_badpixel_exact" in options:
        Mask[IMG == options["ap_badpixel_exact"]] = True
    if np.any(np.logical_not(np.isfinite(IMG))):
        Mask[np.logical_not(np.isfinite(IMG))] = True
        
    if "mask" in results:
        mask = np.logical_or(mask, results["mask"])

    logging.info("%s: masking %i bad pixels" % (options["ap_name"], np.sum(Mask)))
    return IMG, {"mask": Mask}


def Mask_Segmentation_Map(IMG, results, options):
    """Reads the results from other masking routines into AutoProf.

    Creates a mask from a supplied segmentation map. Such maps
    typically number each source with an integer. In such a case,
    AutoProf will check to see if the object center lands on one of
    these segments, if so it will zero out that source-id before
    converting the segmentation map into a mask. If the supplied image
    is just a 0, 1 mask then AutoProf will take it as is.

    Parameters
    -----------------
    ap_mask_file : string, default None
      path to fits file which is a mask for the image. Must have the same dimensions as the main image.

    See Also
    --------
    ap_savemask : bool, default False
      indicates if the mask should be saved after fitting

    Notes
    ----------
    :References:
    - 'background' (optional)
    - 'background noise' (optional)
    - 'center' (optional)
    - 'mask' (optional)

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'mask':  # 2d mask image with boolean datatype (ndarray)
        }

    """

    if "ap_mask_file" not in options or options["ap_mask_file"] is None:
        mask = np.zeros(IMG.shape, dtype=bool)
    else:
        mask = Read_Image(options["ap_mask_file"], options)

    if "center" in results:
        if mask[int(results["center"]["y"]), int(results["center"]["x"])] > 1.1:
            mask[
                mask == mask[int(results["center"]["y"]), int(results["center"]["x"])]
            ] = 0
    elif "ap_set_center" in options:
        if (
            mask[int(options["ap_set_center"]["y"]), int(options["ap_set_center"]["x"])]
            > 1.1
        ):
            mask[
                mask
                == mask[
                    int(options["ap_set_center"]["y"]),
                    int(options["ap_set_center"]["x"]),
                ]
            ] = 0
    elif "ap_guess_center" in options:
        if (
            mask[
                int(options["ap_guess_center"]["y"]),
                int(options["ap_guess_center"]["x"]),
            ]
            > 1.1
        ):
            mask[
                mask
                == mask[
                    int(options["ap_guess_center"]["y"]),
                    int(options["ap_guess_center"]["x"]),
                ]
            ] = 0
    elif mask[int(IMG.shape[0] / 2), int(IMG.shape[1] / 2)] > 1.1:
        mask[mask == mask[int(IMG.shape[0] / 2), int(IMG.shape[1] / 2)]] = 0

    if "mask" in results:
        mask = np.logical_or(mask, results["mask"])

    # Plot star mask for diagnostic purposes
    if "ap_doplot" in options and options["ap_doplot"]:
        bkgrnd = results["background"] if "background" in results else np.median(IMG)
        noise = (
            results["background noise"]
            if "background noise" in results
            else iqr(IMG, rng=[16, 84]) / 2
        )
        LSBImage(IMG - bkgrnd, noise)
        showmask = np.copy(mask)
        showmask[showmask > 1] = 1
        showmask[showmask < 1] = np.nan
        plt.imshow(showmask, origin="lower", cmap="Reds_r", alpha=0.5)
        plt.tight_layout()
        if not ("ap_nologo" in options and options["ap_nologo"]):
            AddLogo(plt.gcf())
        plt.savefig(
            f"{options.get('ap_plotpath','')}mask_{options['ap_name']}.{options.get('ap_plot_extension', 'jpg')}",
            dpi=options["ap_plotdpi"] if "ap_plotdpi" in options else 300,
        )
        plt.close()

    return IMG, {"mask": mask.astype(bool)}


def Star_Mask_IRAF(IMG, results, options):
    """Masking routine which identifies stars and masks a region around them.

    An IRAF star finder wrapper (from photutils) is applied to the
    image and then the identified sources are masked form the image.
    The size of the mask depends on the flux in the source roughly as
    sqrt(log(f)), thus an inverse of a Gaussian.

    See Also
    --------
    ap_savemask : bool, default False
      indicates if the mask should be saved after fitting

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'psf fwhm'
    - 'center'
    - 'mask' (optional)

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'mask':  # 2d mask image with boolean datatype (ndarray)
        }

    """

    fwhm = results["psf fwhm"]
    use_center = results["center"]

    # Find scale of bounding box for galaxy. Stars will only be found within this box
    smaj = results["fit R"][-1] if "fit R" in results else max(IMG.shape)
    xbox = int(1.5 * smaj)
    ybox = int(1.5 * smaj)
    xbounds = [
        max(0, int(use_center["x"] - xbox)),
        min(int(use_center["x"] + xbox), IMG.shape[1]),
    ]
    ybounds = [
        max(0, int(use_center["y"] - ybox)),
        min(int(use_center["y"] + ybox), IMG.shape[0]),
    ]

    # Run photutils wrapper for IRAF star finder
    dat = IMG - results["background"]
    iraffind = IRAFStarFinder(
        fwhm=fwhm, threshold=10.0 * results["background noise"], brightest=50
    )
    irafsources = iraffind(dat[ybounds[0] : ybounds[1], xbounds[0] : xbounds[1]])
    mask = np.zeros(IMG.shape, dtype=bool)
    # Mask star pixels and area around proportionate to their total flux
    XX, YY = np.meshgrid(range(IMG.shape[0]), range(IMG.shape[1]), indexing="ij")
    if irafsources:
        for x, y, f in zip(
            irafsources["xcentroid"], irafsources["ycentroid"], irafsources["flux"]
        ):
            if (
                np.sqrt(
                    (x - (xbounds[1] - xbounds[0]) / 2) ** 2
                    + (y - (ybounds[1] - ybounds[0]) / 2) ** 2
                )
                < 10 * results["psf fwhm"]
            ):
                continue
            # compute distance of every pixel to the identified star
            R = np.sqrt((YY - (x + xbounds[0])) ** 2 + (XX - (y + ybounds[0])) ** 2)
            # Compute the flux of the star
            # f = np.sum(IMG[R < 10*fwhm])
            # Compute radius to reach background noise level, assuming gaussian
            Rstar = (fwhm / 2.355) * np.sqrt(
                2
                * np.log(
                    f
                    / (np.sqrt(2 * np.pi * fwhm / 2.355) * results["background noise"])
                )
            )
            mask[R < Rstar] = True

    if "mask" in results:
        mask = np.logical_or(mask, results["mask"])

    # Plot star mask for diagnostic purposes
    if "ap_doplot" in options and options["ap_doplot"]:
        plt.imshow(
            np.clip(
                dat[ybounds[0] : ybounds[1], xbounds[0] : xbounds[1]],
                a_min=0,
                a_max=None,
            ),
            origin="lower",
            cmap="Greys_r",
            norm=ImageNormalize(stretch=LogStretch()),
        )
        dat = mask.astype(float)[ybounds[0] : ybounds[1], xbounds[0] : xbounds[1]]
        dat[dat == 0] = np.nan
        plt.imshow(dat, origin="lower", cmap="Reds_r", alpha=0.7)
        plt.savefig(
            f"{options.get('ap_plotpath','')}mask_{options['ap_name']}.{options.get('ap_plot_extension', 'jpg')}",
            dpi=options["ap_plotdpi"] if "ap_plotdpi" in options else 300,
        )
        plt.close()

    return IMG, {"mask": mask}


def Star_Mask(IMG, results, options):
    """Masking routine which identifies stars and masks a region around them.

    Using an edge detecting convolutional filter, sources are
    identified that are of similar scale as the PSF. These sources are
    masked with a region roughly 2 times the FWHM of the source.

    See Also
    --------
    ap_savemask : bool, default False
      indicates if the mask should be saved after fitting

    starfind
      :func:`autoprofutils.SharedFunctions.StarFind`

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'psf fwhm'
    - 'center'
    - 'mask' (optional)

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'mask':  # 2d mask image with boolean datatype (ndarray)
        }

    """

    fwhm = results["psf fwhm"]
    use_center = results["center"]

    # Find scale of bounding box for galaxy. Stars will only be found within this box
    smaj = results["fit R"][-1] if "fit R" in results else max(IMG.shape)
    xbox = int(1.5 * smaj)
    ybox = int(1.5 * smaj)
    xbounds = [
        max(0, int(use_center["x"] - xbox)),
        min(int(use_center["x"] + xbox), IMG.shape[1]),
    ]
    ybounds = [
        max(0, int(use_center["y"] - ybox)),
        min(int(use_center["y"] + ybox), IMG.shape[0]),
    ]

    # Run photutils wrapper for IRAF star finder
    dat = IMG - results["background"]

    all_stars = StarFind(
        dat[ybounds[0] : ybounds[1], xbounds[0] : xbounds[1]],
        fwhm,
        results["background noise"],
        detect_threshold=10,
        minsep=3,
        reject_size=3,
    )
    mask = np.zeros(IMG.shape, dtype=bool)
    # Mask star pixels and area around proportionate to their total flux
    XX, YY = np.meshgrid(range(IMG.shape[0]), range(IMG.shape[1]), indexing="ij")
    for x, y, f, d, p in zip(
        all_stars["x"],
        all_stars["y"],
        all_stars["fwhm"],
        all_stars["deformity"],
        all_stars["peak"],
    ):
        if (
            np.sqrt(
                (x - (xbounds[1] - xbounds[0]) / 2) ** 2
                + (y - (ybounds[1] - ybounds[0]) / 2) ** 2
            )
            < 10 * results["psf fwhm"]
        ):
            continue
        # compute distance of every pixel to the identified star
        R = np.sqrt((YY - (x + xbounds[0])) ** 2 + (XX - (y + ybounds[0])) ** 2)
        mask[R < (max(np.log10(p / results["background noise"]), 2) * f)] = True

    if "mask" in results:
        mask = np.logical_or(mask, results["mask"])

    # Plot star mask for diagnostic purposes
    if "ap_doplot" in options and options["ap_doplot"]:
        LSBImage(
            dat[ybounds[0] : ybounds[1], xbounds[0] : xbounds[1]],
            results["background noise"],
        )
        dat = mask.astype(float)[ybounds[0] : ybounds[1], xbounds[0] : xbounds[1]]
        dat[dat == 0] = np.nan
        plt.imshow(dat, origin="lower", cmap="Reds_r", alpha=0.7)
        plt.tight_layout()
        plt.savefig(
            f"{options.get('ap_plotpath','')}mask_{options['ap_name']}.{options.get('ap_plot_extension', 'jpg')}",
            dpi=options["ap_plotdpi"] if "ap_plotdpi" in options else 300,
        )
        plt.close()

    return IMG, {"mask": mask}
