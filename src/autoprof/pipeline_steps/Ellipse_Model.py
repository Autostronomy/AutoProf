import numpy as np
from astropy.io import fits
from scipy.interpolate import SmoothBivariateSpline, interp2d, Rbf, UnivariateSpline
from copy import deepcopy
import matplotlib.pyplot as plt
import sys
import os

from ..autoprofutils.SharedFunctions import (
    AddLogo,
    autocmap,
    LSBImage,
    Rscale_SuperEllipse,
    Rscale_Fmodes,
)
from ..autoprofutils.Diagnostic_Plots import Plot_EllipseModel
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

__all__ = ("EllipseModel",)

def EllipseModel(IMG, results, options):
    """Construct a smooth model image of the galaxy with fixed PA/elliptiicty.

    Constructs a 2D model image of the galaxy based on the extracted
    surface brightness profile and the global ellipticity and position
    angle values. First the image is transformed by rotating and
    stretching until the global ellipse fit has been transformed into
    a circle.  The radial distance of every pixel from the galaxy
    center is then used on an interpolated SB profile to determine the
    corresponding SB value. The SB values are applied and converted
    from mag/arcsec^2 to flux units.

    Parameters
    -----------------
    ap_zeropoint : float, default 22.5
      Photometric zero point

    ap_ellipsemodel_resolution : float, default 1
      scale factor for the ellipse model resolution. Above 1 increases
      the precision of the ellipse model (and computation time),
      between 0 and 1 decreases the resolution (and computation
      time). Note that the ellipse model resolution is defined
      logarithmically, so the center will always be more resolved

    ap_ellipsemodel_replacemaskedpixels : bool, default False
      If True, a new galaxy image will be generated with masked pixels
      replaced by the ellipse model values.

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'ellipse model': # 2d image with flux values for smooth model of galaxy
        }

    """

    zeropoint = options["ap_zeropoint"] if "ap_zeropoint" in options else 22.5

    CHOOSE = np.array(results["prof data"]["SB_e"]) < 0.5
    R = np.array(results["prof data"]["R"])[CHOOSE] / options["ap_pixscale"]

    sb = UnivariateSpline(R, np.array(results["prof data"]["SB"])[CHOOSE], ext=3, s=0)
    pa = UnivariateSpline(
        R, np.array(results["prof data"]["pa"])[CHOOSE] * np.pi / 180, ext=3, s=0
    )
    q = UnivariateSpline(
        R, 1 - np.array(results["prof data"]["ellip"])[CHOOSE], ext=3, s=0
    )
    C = (
        UnivariateSpline(R, np.array(results["prof data"]["C"])[CHOOSE], ext=3, s=0)
        if "C" in results["prof data"]
        else (lambda r: 2)
    )
    fit_Fmodes = (
        options["ap_isofit_fitcoefs"] if "ap_isofit_fitcoefs" in options else None
    )
    if not fit_Fmodes is None:
        A = []
        Phi = []
        Rlimscale = 0.0
        for m in range(len(fit_Fmodes)):
            Rlimscale += np.abs(
                np.array(results["prof data"]["A%i" % fit_Fmodes[m]])[CHOOSE][-1]
            )
            A.append(
                UnivariateSpline(
                    R,
                    np.array(results["prof data"]["A%i" % fit_Fmodes[m]])[CHOOSE],
                    ext=3,
                    s=0,
                )
            )
            Phi.append(
                UnivariateSpline(
                    R,
                    np.array(results["prof data"]["Phi%i" % fit_Fmodes[m]])[CHOOSE]
                    * np.pi
                    / 180,
                    ext=3,
                    s=0,
                )
            )
        Rlimscale = np.exp(Rlimscale)
    else:
        Rlimscale = 1.0

    ranges = [
        [
            max(0, int(results["center"]["x"] - R[-1] * Rlimscale - 2)),
            min(IMG.shape[1], int(results["center"]["x"] + R[-1] * Rlimscale + 2)),
        ],
        [
            max(0, int(results["center"]["y"] - R[-1] * Rlimscale - 2)),
            min(IMG.shape[0], int(results["center"]["y"] + R[-1] * Rlimscale + 2)),
        ],
    ]

    XX, YY = np.meshgrid(
        np.arange(ranges[0][1] - ranges[0][0], dtype=np.float32)
        - results["center"]["x"]
        + float(ranges[0][0]),
        np.arange(ranges[1][1] - ranges[1][0], dtype=np.float32)
        - results["center"]["y"]
        + float(ranges[1][0]),
    )
    theta = np.arctan2(YY, XX) #np.arctan(YY / XX) + np.pi * (XX < 0)
    Radius = np.sqrt(XX ** 2 + YY ** 2)
    MM = np.zeros(XX.shape, dtype=np.float32)
    Prox = np.zeros(XX.shape, dtype=np.float32) + np.inf
    WINDOW = [[0, XX.shape[0]], [0, XX.shape[1]]]
    for r in reversed(
        np.logspace(
            np.log10(R[0] / 2),
            np.log10(R[-1]),
            int(
                len(R)
                * 2
                * (
                    options["ap_ellipsemodel_resolution"]
                    if "ap_ellipsemodel_resolution" in options
                    else 2.
                )
            ),
        )
    ):
        if (r * Rlimscale) < (np.max(XX.shape) / np.sqrt(2)):
            WINDOW = [
                [
                    max(
                        0,
                        int(
                            results["center"]["y"]
                            - float(ranges[1][0])
                            - r * Rlimscale * 1.2
                        ),
                    ),
                    min(
                        XX.shape[0],
                        int(
                            results["center"]["y"]
                            - float(ranges[1][0])
                            + r * Rlimscale * 1.2
                        ),
                    ),
                ],
                [
                    max(
                        0,
                        int(
                            results["center"]["x"]
                            - float(ranges[0][0])
                            - r * Rlimscale * 1.2
                        ),
                    ),
                    min(
                        XX.shape[1],
                        int(
                            results["center"]["x"]
                            - float(ranges[0][0])
                            + r * Rlimscale * 1.2
                        ),
                    ),
                ],
            ]

        Rscale = Rscale_SuperEllipse(
            theta[WINDOW[0][0] : WINDOW[0][1], WINDOW[1][0] : WINDOW[1][1]] - pa(r),
            1.0 - q(r),
            C(r),
        )
        if not fit_Fmodes is None:
            Rscale *= np.exp(
                sum(
                    A[m](r)
                    * np.cos(
                        fit_Fmodes[m]
                        * (
                            theta[
                                WINDOW[0][0] : WINDOW[0][1], WINDOW[1][0] : WINDOW[1][1]
                            ]
                            + (Phi[m](r) - pa(r))
                        )
                    )
                    for m in range(len(fit_Fmodes))
                )
            )
        RR = Radius[WINDOW[0][0] : WINDOW[0][1], WINDOW[1][0] : WINDOW[1][1]] / Rscale
        D = np.abs(RR - r)
        CLOSE = D < Prox[WINDOW[0][0] : WINDOW[0][1], WINDOW[1][0] : WINDOW[1][1]]
        MM[WINDOW[0][0] : WINDOW[0][1], WINDOW[1][0] : WINDOW[1][1]][CLOSE] = RR[CLOSE]
        Prox[WINDOW[0][0] : WINDOW[0][1], WINDOW[1][0] : WINDOW[1][1]][CLOSE] = D[CLOSE]
    MM = sb(MM)
    MM = 10 ** (-(MM - zeropoint - 5 * np.log10(options["ap_pixscale"])) / 2.5)

    Rscale = Rscale_SuperEllipse(theta - pa(R[-1]), 1.0 - q(R[-1]), C(R[-1]))
    if not fit_Fmodes is None:
        Rscale *= np.exp(
            sum(
                A[m](R[-1])
                * np.cos(fit_Fmodes[m] * (theta + (Phi[m](R[-1]) - pa(R[-1]))))
                for m in range(len(fit_Fmodes))
            )
        )
    RR = Radius / Rscale
    # RR = np.sqrt((XX*np.cos(-pa(R[-1])) - YY*np.sin(-pa(R[-1])))**2 + ((XX*np.sin(-pa(R[-1])) + YY*np.cos(-pa(R[-1])))/np.clip(q(R[-1]),a_min = 0.03,a_max = 1))**2)
    # if not fit_Fmodes is None:
    #     RR /= np.exp(sum(A[m](R[-1]) * np.cos(fit_Fmodes[m] * (theta + (Phi[m](R[-1]) - pa(R[-1])))) for m in range(len(fit_Fmodes))))
    MM[RR > R[-1]] = 0

    Model = np.zeros(IMG.shape, dtype=np.float32)
    Model[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]] = MM

    header = fits.Header()
    hdul = fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU(Model)])

    hdul.writeto(
        os.path.join(
            options["ap_plotpath"] if "ap_plotpath" in options else "",
            "%s_genmodel.fits" % options["ap_name"],
        ),
        overwrite=True,
    )

    if "mask" in results and not results["mask"] is None:
        mask = results["mask"]
    else:
        mask = None
    if (
        "ap_ellipsemodel_replacemaskedpixels" in options
        and options["ap_ellipsemodel_replacemaskedpixels"]
        and not mask is None
    ):
        header = fits.Header()
        newImage = np.copy(IMG)
        newImage[mask] = Model[mask] + results["background"]
        hdul = fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU(newImage)])

        hdul.writeto(
            os.path.join(
                options["ap_plotpath"] if "ap_plotpath" in options else "",
                "%s_maskreplace.fits" % options["ap_name"],
            ),
            overwrite=True,
        )

    if "ap_doplot" in options and options["ap_doplot"]:
        Plot_EllipseModel(IMG, Model, R, "gen", results, options)

    return IMG, {"ellipse model": Model}
