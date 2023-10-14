from photutils import DAOStarFinder, IRAFStarFinder
import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch, LogStretch, HistEqStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import logging
from astropy.io import fits
from itertools import product
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal
from scipy.fftpack import fft, ifft
import sys
import os

from ..autoprofutils.SharedFunctions import (
    StarFind,
    AddLogo,
    LSBImage,
    autocolours,
    interpolate_Lanczos,
    interpolate_bicubic,
    Read_Image,
)
from ..autoprofutils.Diagnostic_Plots import Plot_PSF_Stars
from copy import deepcopy

__all__ = ("PSF_IRAF", "PSF_StarFind", "PSF_Image", "PSF_deconvolve")

def PSF_IRAF(IMG, results, options):
    """PSF routine which identifies stars and averages the FWHM.

    Uses the photutil IRAF wrapper to identify stars in the image and
    computes the average FWHM.

    Parameters
    -----------------
    ap_guess_psf : float, default None
      Initialization value for the PSF calculation in pixels. If not
      given, AutoProf will default with a guess of 1/*ap_pixscale*

    ap_set_psf : float, default None
      force AutoProf to use this PSF value (in pixels) instead of
      calculating its own.

    Notes
    ----------
    :References:
    - 'background' (float)
    - 'background noise' (float)

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'psf fwhm':  # FWHM of the average PSF for the image
        }

    """
    if "ap_set_psf" in options:
        logging.info(
            "%s: PSF set by user: %.4e" % (options["ap_name"], options["ap_set_psf"])
        )
        return IMG, {"psf fwhm": options["ap_set_psf"]}
    elif "ap_guess_psf" in options:
        logging.info(
            "%s: PSF initialized by user: %.4e"
            % (options["ap_name"], options["ap_guess_psf"])
        )
        fwhm_guess = options["ap_guess_psf"]
    else:
        fwhm_guess = max(1.0, 1.0 / options["ap_pixscale"])

    edge_mask = np.zeros(IMG.shape, dtype=bool)
    edge_mask[
        int(IMG.shape[0] / 5.0) : int(4.0 * IMG.shape[0] / 5.0),
        int(IMG.shape[1] / 5.0) : int(4.0 * IMG.shape[1] / 5.0),
    ] = True

    dat = IMG - results["background"]
    # photutils wrapper for IRAF star finder
    count = 0
    sources = 0
    psf_iter = deepcopy(psf_guess)
    try:
        while count < 5 and sources < 20:
            iraffind = IRAFStarFinder(
                fwhm=psf_iter, threshold=6.0 * results["background noise"], brightest=50
            )
            irafsources = iraffind.find_stars(dat, edge_mask)
            psf_iter = np.median(irafsources["fwhm"])
            if np.median(irafsources["sharpness"]) >= 0.95:
                break
            count += 1
            sources = len(irafsources["fwhm"])
    except:
        return IMG, {"psf fwhm": fwhm_guess}
    if len(irafsources) < 5:
        return IMG, {"psf fwhm": fwhm_guess}

    psf = np.median(irafsources["fwhm"])

    if "ap_doplot" in options and options["ap_doplot"]:
        Plot_PSF_Stars(
            IMG,
            irafsources["xcentroid"],
            irafsources["ycentroid"],
            irafsources["fwhm"],
            psf,
            results,
            options,
        )

    return IMG, {"psf fwhm": psf, "auxfile psf": "psf fwhm: %.3f pix" % psf}


def PSF_StarFind(IMG, results, options):
    """PSF routine which identifies stars and averages the FWHM.

    The PSF method uses an edge finding convolution filter to identify
    candidate star pixels, then averages their FWHM. Randomly iterates
    through the pixels and searches for a local maximum. An FFT is
    used to identify non-circular star candidate (artifacts or
    galaxies) which may have been picked up by the edge
    finder. Circular apertures are placed around the star until half
    the central flux value is reached, This is recorded as the FWHM
    for that star. A collection of 50 stars are identified and the
    most circular (by FFT coefficients) half are kept, a median is
    taken as the image PSF.

    Parameters
    -----------------
    ap_guess_psf : float, default None
      Initialization value for the PSF calculation in pixels. If not
      given, AutoProf will default with a guess of 1/*ap_pixscale*

    ap_set_psf : float, default None
      force AutoProf to use this PSF value (in pixels) instead of
      calculating its own.

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

        {'psf fwhm':  # FWHM of the average PSF for the image
        }

    """

    if "ap_set_psf" in options:
        logging.info(
            "%s: PSF set by user: %.4e" % (options["ap_name"], options["ap_set_psf"])
        )
        return IMG, {"psf fwhm": options["ap_set_psf"]}
    elif "ap_guess_psf" in options:
        logging.info(
            "%s: PSF initialized by user: %.4e"
            % (options["ap_name"], options["ap_guess_psf"])
        )
        fwhm_guess = options["ap_guess_psf"]
    else:
        fwhm_guess = max(1.0, 1.0 / options["ap_pixscale"])

    if "mask" in results:
        use_mask = results["mask"]
    else:
        use_mask = np.zeros(IMG.shape, dtype=bool)
        use_mask[
            int(IMG.shape[0] / 5.0) : int(4.0 * IMG.shape[0] / 5.0),
            int(IMG.shape[1] / 5.0) : int(4.0 * IMG.shape[1] / 5.0),
        ] = True
    
    stars = StarFind(
        IMG - results["background"],
        fwhm_guess,
        results["background noise"],
        use_mask,
        maxstars=50,
    )
    if len(stars["fwhm"]) <= 10:
        logging.error(
            "%s: unable to detect enough stars! PSF results not valid, using 1 arcsec estimate psf of %f"
            % (options["ap_name"], fwhm_guess)
        )
        return IMG, {"psf fwhm": fwhm_guess}

    def_clip = 0.1
    while np.sum(stars["deformity"] < def_clip) < max(10, len(stars["fwhm"]) / 2):
        def_clip += 0.1
    psf = np.median(stars["fwhm"][stars["deformity"] < def_clip])
    if "ap_doplot" in options and options["ap_doplot"]:
        Plot_PSF_Stars(
            IMG,
            stars["x"],
            stars["y"],
            stars["fwhm"],
            psf,
            results,
            options,
            flagstars=stars["deformity"] >= def_clip,
        )

    logging.info(
        "%s: found psf: %f with deformity clip of: %f"
        % (options["ap_name"], psf, def_clip)
    )
    return IMG, {"psf fwhm": psf, "auxfile psf": "psf fwhm: %.3f pix" % psf}


def PSF_Image(IMG, results, options):
    """PSF routine which identifies stars and averages the FWHM.

    Constructs an averaged PSF image. Extracts a window of pixels
    around each identified star (+-10 PSF) and normalizes the flux
    total to 1. All extraced normalized stars are median stacked. The
    final PSF is saved as "<name>_psf.fits" and added to the results
    dictionary. Also calculates the PSF FWHM and adds it to the
    results dictionary. This method is currently very slow.

    Parameters
    -----------------

    ap_guess_psf : float, default None
      Initialization value for the PSF calculation in pixels. If not
      given, AutoProf will default with a guess of 1/*ap_pixscale*

    ap_set_psf : float, default None
      force AutoProf to use this PSF value (in pixels) instead of
      calculating its own.

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

        {'psf fwhm':  # FWHM of the average PSF for the image
         'auxfile psf': # aux file message giving the PSF
         'psf img':   # image of the PSF as numpy array
        }

    """

    if "ap_set_psf" in options:
        logging.info(
            "%s: PSF set by user: %.4e" % (options["ap_name"], options["ap_set_psf"])
        )
        return IMG, {"psf fwhm": options["ap_set_psf"]}
    elif "ap_guess_psf" in options:
        logging.info(
            "%s: PSF initialized by user: %.4e"
            % (options["ap_name"], options["ap_guess_psf"])
        )
        fwhm_guess = options["ap_guess_psf"]
    else:
        fwhm_guess = max(1.0, 1.0 / options["ap_pixscale"])

    edge_mask = np.zeros(IMG.shape, dtype=bool)
    edge_mask[
        int(IMG.shape[0] / 4.0) : int(3.0 * IMG.shape[0] / 4.0),
        int(IMG.shape[1] / 4.0) : int(3.0 * IMG.shape[1] / 4.0),
    ] = True
    dat = IMG - results["background"]
    stars = StarFind(
        dat,
        fwhm_guess,
        results["background noise"],
        edge_mask,
        detect_threshold=5.0,
        maxstars=100,
    )
    if len(stars["fwhm"]) <= 10:
        logging.error(
            "%s: unable to detect enough stars! PSF results not valid, using 1 arcsec estimate psf of %f"
            % (options["ap_name"], fwhm_guess)
        )

    def_clip = 0.1
    while np.sum(stars["deformity"] < def_clip) < max(10, len(stars["fwhm"]) * 2 / 3):
        def_clip += 0.1
    psf = np.median(stars["fwhm"][stars["deformity"] < def_clip])
    psf_iqr = np.quantile(stars["fwhm"][stars["deformity"] < def_clip], [0.1, 0.9])
    psf_size = int(psf * 10)
    if psf_size % 2 == 0:  # make PSF odd for easier calculations
        psf_size += 1

    psf_img = None
    XX, YY = np.meshgrid(
        np.array(range(psf_size)) - psf_size // 2,
        np.array(range(psf_size)) - psf_size // 2,
    )
    XX, YY = np.ravel(XX), np.ravel(YY)

    for i in range(len(stars["x"])):
        # ignore objects that likely aren't stars
        if (
            stars["deformity"][i] > def_clip
            or stars["fwhm"][i] < psf_iqr[0]
            or stars["fwhm"][i] > psf_iqr[1]
        ):
            continue
        # ignore objects that are too close to the edge
        if (
            stars["x"][i] < psf_size // 2
            or (dat.shape[1] - stars["x"][i]) < psf_size // 2
            or stars["y"][i] < psf_size // 2
            or (dat.shape[1] - stars["y"][i]) < psf_size // 2
        ):
            continue
        flux = interpolate_Lanczos(
            dat, XX + stars["x"][i], YY + stars["y"][i], 10
        ).reshape((1, psf_size, psf_size))
        flux /= np.sum(flux)
        psf_img = flux if psf_img is None else np.concatenate((psf_img, flux))

    # stack the PSF
    psf_img = np.median(psf_img, axis=0)
    # normalize the PSF
    psf_img /= np.sum(psf_img)

    hdul = fits.HDUList([fits.PrimaryHDU(psf_img)])
    hdul.writeto(
        os.path.join(
            options["ap_saveto"] if "ap_saveto" in options else "",
            "%s_psf.fits" % options["ap_name"],
        ),
        overwrite=True,
    )

    if "ap_doplot" in options and options["ap_doplot"]:
        plt.imshow(
            psf_img,
            origin="lower",
            cmap="Greys",
            norm=ImageNormalize(stretch=HistEqStretch(psf_img)),
        )
        my_cmap = cm.Greys_r
        my_cmap.set_under("k", alpha=0)
        fluxpeak = psf_img[psf_size // 2 + 1, psf_size // 2 + 1] / 2
        plt.imshow(
            np.clip(psf_img, a_min=fluxpeak / 10, a_max=None),
            origin="lower",
            cmap=my_cmap,
            norm=ImageNormalize(stretch=LogStretch(), clip=False),
            clim=[fluxpeak / 9, None],
            vmin=fluxpeak / 9,
        )
        plt.axis("off")
        plt.tight_layout()
        if not ("ap_nologo" in options and options["ap_nologo"]):
            AddLogo(plt.gcf())
        plt.savefig(
            f"{options.get('ap_plotpath','')}PSF_{options['ap_name']}.{options.get('ap_plot_extension', 'jpg')}",
            dpi=options["ap_plotdpi"] if "ap_plotdpi" in options else 300,
        )
        plt.close()

    return IMG, {
        "psf fwhm": psf,
        "auxfile psf": "psf fwhm: %.3f pix" % psf,
        "psf img": psf_img,
    }


def PSF_deconvolve(IMG, results, options):
    """routine which deconvolves the PSF from the primary image.

    Performs Richardson-Lucy deconvolution on the primary galaxy image
    using the sci-kit image implementation (the user must have skimage
    in their python installation). This deconvolution procedure is
    more stable than standard FFT deconvolution. This method is
    currently very slow. If the user provides an image via
    'ap_psf_file' then that will be taken as the psf and deconvolved
    from the image. If there is no file given, but 'psf img' exists in
    the results dictionary (ie from the 'psf img' pipeline step) then
    that will be used. If no other option is available, the 'psf fwhm'
    will be taken from the results dictionary and a PSF image will be
    constructed using a Gaussian of the given PSF out to 20 times the
    PSF size.

    Parameters
    -----------------

    ap_psf_file : string, default None
      Optional argument. Path to PSF fits file. For best results the
      image should have an odd number of pixels with the PSF centered
      in the image.

    ap_psf_deconvolution_iterations : int, default 50
      number of itterations of the Richardson-Lucy deconvolution
      algorithm to perform.

    Notes
    ----------
    :References:
    - 'psf img' (optional)
    - 'psf fwhm' (optional)

    Returns
    -------
    IMG : ndarray
      deconvolved galaxy image

    results : dict
      .. code-block:: python

        {}

    """

    from skimage import restoration

    if "ap_psf_file" in options:
        psf_img = Read_Image(options["ap_psf_file"], options)
    elif "psf img" in results:
        psf_img = results["psf img"]
    else:
        psf_size = int(results["psf fwhm"] * 20)
        if psf_size % 2 == 0:  # make PSF odd for easier calculations
            psf_size += 1

        XX, YY = np.meshgrid(
            np.array(range(psf_size)) - psf_size // 2,
            np.array(range(psf_size)) - psf_size // 2,
        )
        psf_std = results["psf fwhm"] / np.sqrt(8 * np.log(2))
        psf_img = np.exp(-(XX ** 2 + YY ** 2) / (2 * psf_std ** 2)) / np.sqrt(
            2 * np.pi * psf_std ** 2
        )
        psf_img /= np.sum(psf_img)

    if np.abs(np.sum(psf_img) - 1) > 1e-7:
        logging.warn("PSF image not normalized! sum(PSF) = %.3e" % np.sum(psf_img))
    dmax = np.max(IMG)
    dmin = np.min(IMG)
    dat_deconv = restoration.richardson_lucy(
        (IMG - dmin) / (dmax - dmin) - 0.5,
        psf_img,
        iterations=options["ap_psf_deconvolution_iterations"]
        if "ap_psf_deconvolution_iterations" in options
        else 50,
    )
    dat_deconv = (dat_deconv + 0.5) * (dmax - dmin) + dmin

    if "ap_psf_deconvolve_save" in options and options["ap_psf_deconvolve_save"]:
        header = fits.Header()
        hdul = fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU(dat_deconv)])
        
        hdul.writeto(
            os.path.join(
                options["ap_saveto"] if "ap_saveto" in options else "",
                "%s_deconvolved.fits" % options["ap_name"],
            ),
            overwrite=True,
        )
        
    
    return dat_deconv, {}
