import sys
import os
from scipy.integrate import trapz, quad
from scipy.stats import iqr, norm
from scipy.interpolate import interp2d, SmoothBivariateSpline, Rbf, RectBivariateSpline
from scipy.fftpack import fft, ifft
from scipy.optimize import minimize
from scipy.signal import convolve2d
from astropy.visualization import SqrtStretch, LogStretch, HistEqStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
from astropy.io import fits
from itertools import compress
import numpy as np
from photutils.isophote import EllipseSample, EllipseGeometry, Isophote, IsophoteList
from photutils.isophote import Ellipse as Photutils_Ellipse
import logging
from copy import deepcopy, copy
from time import time
import matplotlib.cm as cm
from matplotlib.cbook import get_sample_data
from matplotlib.colors import LinearSegmentedColormap
import warnings
Abs_Mag_Sun = {
    "u": 6.39,
    "g": 5.11,
    "r": 4.65,
    "i": 4.53,
    "z": 4.50,
    "U": 6.33,
    "B": 5.31,
    "V": 4.80,
    "R": 4.60,
    "I": 4.51,
    "J": 4.54,
    "H": 4.66,
    "K": 5.08,
    "3.6um": 3.24,
}  # mips.as.arizona.edu/~cnaw/sun.html # also see: http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
Au_to_Pc = 4.84814e-6  # pc au^-1
Pc_to_m = 3.086e16  # m pc^-1
Pc_to_Au = 206265.0  # Au pc^-1

cmaplist = ["#000000", "#720026", "#A0213F", "#ce4257", "#E76154", "#ff9b54", "#ffd1b1"]
cdict = {"red": [], "green": [], "blue": []}
cpoints = np.linspace(0, 1, len(cmaplist))
for i in range(len(cmaplist)):
    cdict["red"].append(
        [cpoints[i], int(cmaplist[i][1:3], 16) / 256, int(cmaplist[i][1:3], 16) / 256]
    )
    cdict["green"].append(
        [cpoints[i], int(cmaplist[i][3:5], 16) / 256, int(cmaplist[i][3:5], 16) / 256]
    )
    cdict["blue"].append(
        [cpoints[i], int(cmaplist[i][5:7], 16) / 256, int(cmaplist[i][5:7], 16) / 256]
    )
autocmap = LinearSegmentedColormap("autocmap", cdict)
autocolours = {
    "red1": "#c33248",
    "blue1": "#84DCCF",
    "blue2": "#6F8AB7",
    "redrange": ["#720026", "#A0213F", "#ce4257", "#E76154", "#ff9b54", "#ffd1b1"],
}  # '#D95D39'


def LSBImage(dat, noise):
    plt.figure(figsize=(6, 6))
    plt.imshow(
        dat,
        origin="lower",
        cmap="Greys",
        norm=ImageNormalize(stretch=HistEqStretch(dat[dat <= 3*noise]), clip = False, vmax = 3*noise, vmin = np.min(dat)),
    )
    my_cmap = copy(cm.Greys_r)
    my_cmap.set_under("k", alpha=0)
    
    plt.imshow(
        np.ma.masked_where(dat < 3*noise, dat), 
        origin="lower",
        cmap=my_cmap,
        norm=ImageNormalize(stretch=LogStretch(),clip = False),
        clim=[3 * noise, None],
        interpolation = 'none',
    )
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.05)
    plt.xlim([0, dat.shape[1]])
    plt.ylim([0, dat.shape[0]])


def _display_time(seconds):
    intervals = (
        ('hours', 3600),    # 60 * 60
        ('arcminutes', 60),
        ('arcseconds', 1),
    )
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result)

    
def AddScale(ax, img_width, loc = 'lower right'):
    """
    ax: figure axis object
    img_width: image width in arcseconds
    loc: location to put hte scale bar
    """
    scale_width = int(img_width/6)
    
    if scale_width > 60 and scale_width % 60 <= 15:
        scale_width -= scale_width % 60
    if scale_width > 45 and scale_width % 60 >= 45:
        scale_width += 60 - (scale_width % 60)
    if 15 < scale_width % 60 < 45:
        scale_width += 30 - (scale_width % 60)
        
    label = _display_time(scale_width)

    xloc = 0.05 if 'left' in loc else 0.95
    yloc = 0.95 if 'upper' in loc else 0.05
    
    ax.text(xloc - 0.5*scale_width/img_width, yloc + 0.005, label,
            horizontalalignment = 'center', verticalalignment = 'bottom',
            transform=ax.transAxes,
            fontsize = 'x-small' if len(label) < 20 else 'xx-small',
            weight = 'bold', color = autocolours['red1'])
    ax.plot([xloc - scale_width/img_width,xloc], [yloc,yloc],
            transform=ax.transAxes, color = autocolours['red1'])
    
        
def AddLogo(fig, loc=[0.8, 0.01, 0.844 / 5, 0.185 / 5], white=False):
    im = plt.imread(
        get_sample_data(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "media/",
                ("AP_logo_white.png" if white else "AP_logo.png"),
            )
        )
    )
    newax = fig.add_axes(loc, zorder=1000)
    if white:
        newax.imshow(np.zeros(im.shape) + np.array([0, 0, 0, 1]))
    else:
        newax.imshow(np.ones(im.shape))
    newax.imshow(im)
    newax.axis("off")


def flux_to_sb(flux, pixscale, zeropoint):
    return -2.5 * np.log10(flux) + zeropoint + 5 * np.log10(pixscale)


def flux_to_mag(flux, zeropoint, fluxe=None):
    if fluxe is None:
        return -2.5 * np.log10(flux) + zeropoint
    else:
        return -2.5 * np.log10(flux) + zeropoint, 2.5 * fluxe / (np.log(10) * flux)


def sb_to_flux(sb, pixscale, zeropoint):
    return (pixscale ** 2) * 10 ** (-(sb - zeropoint) / 2.5)


def mag_to_flux(mag, zeropoint, mage=None):
    if mage is None:
        return 10 ** (-(mag - zeropoint) / 2.5)
    else:
        I = 10 ** (-(mag - zeropoint) / 2.5)
        return I, np.log(10) * I * mage / 2.5


def magperarcsec2_to_mag(mu, a=None, b=None, A=None):
    """
    Converts mag/arcsec^2 to mag
    mu: mag/arcsec^2
    a: semi major axis radius (arcsec)
    b: semi minor axis radius (arcsec)
    A: pre-calculated area (arcsec^2)
    returns: mag
    """
    assert (not A is None) or (not a is None and not b is None)
    if A is None:
        A = np.pi * a * b
    return mu - 2.5 * np.log10(
        A
    )  # https://en.wikipedia.org/wiki/Surface_brightness#Calculating_surface_brightness


def mag_to_magperarcsec2(m, a=None, b=None, R=None, A=None):
    """
    Converts mag to mag/arcsec^2
    m: mag
    a: semi major axis radius (arcsec)
    b: semi minor axis radius (arcsec)
    A: pre-calculated area (arcsec^2)
    returns: mag/arcsec^2
    """
    assert (not A is None) or (not a is None and not b is None) or (not R is None)
    if not R is None:
        A = np.pi * (R ** 2)
    elif A is None:
        A = np.pi * a * b
    return m + 2.5 * np.log10(
        A
    )  # https://en.wikipedia.org/wiki/Surface_brightness#Calculating_surface_brightness


def halfmag(mag):
    """
    Computes the magnitude corresponding to half in log space.
    Effectively, converts to luminosity, divides by 2, then
    converts back to magnitude. Distance is not needed as it
    cancels out. Here is a basic walk through::

      m_1 - m_ref = -2.5log10(I_1/I_ref)
      m_2 - m_ref = -2.5log10(I_1/2I_ref)
                  = -2.5log10(I_1/I_ref) + 2.5log10(2)
      m_2 = m_1 + 2.5log10(2)
    """

    return mag + 2.5 * np.log10(2)


def pc_to_arcsec(R, D, Re=0.0, De=0.0):
    """
    Converts a size in parsec to arcseconds

    R: length in pc
    D: distance in pc
    """

    theta = R / (D * Au_to_Pc)
    if np.all(Re == 0) and np.all(De == 0):
        return theta
    else:
        e = theta * np.sqrt((Re / R) ** 2 + (De / D) ** 2)
        return theta, e


def arcsec_to_pc(theta, D, thetae=0.0, De=0.0):
    """
    Converts a size in arcseconds to parsec

    theta: angle in arcsec
    D: distance in pc
    """
    r = theta * D * Au_to_Pc
    if np.all(thetae == 0) and np.all(De == 0):
        return r
    else:
        e = r * np.sqrt((thetae / theta) ** 2 + (De / D) ** 2)
        return r, e


def ISB_to_muSB(I, band, IE=None):
    """
    Converts surface brightness in Lsolar pc^-2 into mag arcsec^-2

    I: surface brightness, (L/Lsun) pc^-2
    band: Photometric band in which measurements were taken
    returns: surface brightness in mag arcsec^-2
    """

    muSB = 21.571 + Abs_Mag_Sun[band] - 2.5 * np.log10(I)
    if IE is None:
        return muSB
    else:
        return muSB, (2.5 / np.log(10)) * IE / I


def muSB_to_ISB(mu, band, muE=None):
    """
    Converts surface brightness in mag arcsec^-2 into Lsolar pc^-2

    mu: surface brightness, mag arcsec^-2
    band: Photometric band in which measurements were taken
    returns: surface brightness in (L/Lsun) pc^-2
    """

    ISB = 10 ** ((21.571 + Abs_Mag_Sun[band] - mu) / 2.5)
    if muE is None:
        return ISB
    else:
        return ISB, (np.log(10) / 2.5) * ISB * muE


def app_mag_to_abs_mag(m, D, me=0.0, De=0.0):
    """
    Converts an apparent magnitude to an absolute magnitude
    m: Apparent magnitude
    D: Distance to object in parsecs
    returns: Absolute magnitude at 10 parcecs
    """

    M = m - 5.0 * np.log10(D / 10.0)
    if np.all(me == 0) and np.all(De == 0):
        return M
    else:
        return M, np.sqrt(me ** 2 + (5.0 * De / (D * np.log(10))) ** 2)


def abs_mag_to_app_mag(M, D, Me=0.0, De=0.0):
    """
    Converts an absolute magnitude to an apparent magnitude
    M: Absolute magnitude at 10 parcecs
    D: Distance to object in parsecs
    returns: Apparent magnitude
    """

    m = M + 5.0 * np.log10(D / 10.0)
    if np.all(Me == 0) and np.all(De == 0):
        return m
    else:
        return m, np.sqrt(Me ** 2 + (5.0 * De / (D * np.log(10))) ** 2)


def mag_to_L(mag, band, mage=None, zeropoint=None):
    """
    Returns the luminosity (in solar luminosities) given the absolute magnitude and reference point.
    mag: Absolute magnitude
    band: Photometric band in which measurements were taken
    mage: uncertainty in absolute magnitude
    zeropoint: user defined zero point
    returns: Luminosity in solar luminosities
    """

    L = 10 ** (((Abs_Mag_Sun[band] if zeropoint is None else zeropoint) - mag) / 2.5)
    if mage is None:
        return L
    else:
        Le = np.abs(L * mage * np.log(10) / 2.5)
        return L, Le


def L_to_mag(L, band, Le=None, zeropoint=None):
    """
    Returns the Absolute magnitude of a star given its luminosity and distance
    L: Luminosity in solar luminosities
    band: Photometric band in which measurements were taken
    Le: Uncertainty in luminosity
    zeropoint: user defined zero point

    returns: Absolute magnitude
    """

    mag = (Abs_Mag_Sun[band] if zeropoint is None else zeropoint) - 2.5 * np.log10(L)

    if Le is None:
        return mag
    else:
        mage = np.abs(2.5 * Le / (L * np.log(10)))
        return mag, mage


def Sigma_Clip_Upper(v, iterations=10, nsigma=5):
    """
    Perform sigma clipping on the "v" array. Each iteration involves
    computing the median and 16-84 range, these are used to clip beyond
    "nsigma" number of sigma above the median. This is repeated for
    "iterations" number of iterations, or until convergence if None.
    """

    v2 = np.sort(v)
    i = 0
    old_lim = 0
    lim = np.inf
    while i < iterations and old_lim != lim:
        med = np.median(v2[v2 < lim])
        rng = iqr(v2[v2 < lim], rng=[16, 84]) / 2
        old_lim = lim
        lim = med + rng * nsigma
        i += 1
    return lim


def Smooth_Mode(v):
    # set the starting point for the optimization at the median
    start = np.median(v)
    # set the smoothing scale equal to roughly 0.5% of the width of the data
    scale = iqr(v) / max(1.0, np.log10(len(v)))  # /10
    # Fit the peak of the smoothed histogram
    res = minimize(
        lambda x: -np.sum(np.exp(-(((v - x) / scale) ** 2))),
        x0=[start],
        method="Nelder-Mead",
    )
    return res.x[0]


def _average(v, method="median"):
    if method == "mean":
        return np.mean(v)
    elif method == "mode":
        return Smooth_Mode(v)
    elif method == "median":
        return np.median(v)
    else:
        raise ValueError("Unrecognized average method: %s" % method)


def _scatter(v, method="median"):
    if method == "mean":
        return np.std(v)
    elif method == "mode":
        return iqr(v, rng=(31.731 / 2, 100 - 31.731 / 2)) / 2.0
    elif method == "median":
        return iqr(v, rng=(31.731 / 2, 100 - 31.731 / 2)) / 2.0
    else:
        raise ValueError("Unrecognized average method: %s" % method)


def Rscale_Fmodes(theta, modes, Am, Phim):
    return np.exp(
        sum(Am[m] * np.cos(modes[m] * (theta + Phim[m])) for m in range(len(modes)))
    )


def parametric_Fmodes(theta, modes, Am, Phim):
    x = np.cos(theta)
    y = np.sin(theta)
    Rscale = Rscale_Fmodes(theta, modes, Am, Phim)
    return x * Rscale, y * Rscale


def Rscale_SuperEllipse(theta, ellip, C=2):
    res = (1 - ellip) / (
        np.abs((1 - ellip) * np.cos(theta)) ** (C) + np.abs(np.sin(theta)) ** (C)
    ) ** (1.0 / C)
    return res


def parametric_SuperEllipse(theta, ellip, C=2):
    rs = Rscale_SuperEllipse(theta, ellip, C)
    return rs * np.cos(theta), rs * np.sin(theta)


def Rotate_Cartesian(theta, X, Y):
    return X * np.cos(theta) - Y * np.sin(theta), Y * np.cos(theta) + X * np.sin(theta)


def interpolate_bicubic(dat, X, Y):
    f_interp = RectBivariateSpline(
        np.arange(dat.shape[0], dtype=np.float32),
        np.arange(dat.shape[1], dtype=np.float32),
        dat,
    )
    return f_interp(Y, X, grid=False)


def interpolate_Lanczos(dat, X, Y, scale):
    """
    Perform Lanczos interpolation on an image.
    https://pixinsight.com/doc/docs/InterpolationAlgorithms/InterpolationAlgorithms.html
    """
    flux = []

    for i in range(len(X)):
        box = [
            [
                max(0, int(round(np.floor(X[i]) - scale + 1))),
                min(dat.shape[1], int(round(np.floor(X[i]) + scale + 1))),
            ],
            [
                max(0, int(round(np.floor(Y[i]) - scale + 1))),
                min(dat.shape[0], int(round(np.floor(Y[i]) + scale + 1))),
            ],
        ]
        chunk = dat[box[1][0] : box[1][1], box[0][0] : box[0][1]]
        XX = np.ones(chunk.shape)
        Lx = (
            np.sinc(np.arange(-scale + 1, scale + 1) - X[i] + np.floor(X[i]))
            * np.sinc(
                (np.arange(-scale + 1, scale + 1) - X[i] + np.floor(X[i])) / scale
            )
        )[
            box[0][0]
            - int(round(np.floor(X[i]) - scale + 1)) : 2 * scale
            + box[0][1]
            - int(round(np.floor(X[i]) + scale + 1))
        ]
        Ly = (
            np.sinc(np.arange(-scale + 1, scale + 1) - Y[i] + np.floor(Y[i]))
            * np.sinc(
                (np.arange(-scale + 1, scale + 1) - Y[i] + np.floor(Y[i])) / scale
            )
        )[
            box[1][0]
            - int(round(np.floor(Y[i]) - scale + 1)) : 2 * scale
            + box[1][1]
            - int(round(np.floor(Y[i]) + scale + 1))
        ]
        L = XX * Lx * Ly.reshape((Ly.size, -1))
        w = np.sum(L)
        flux.append(np.sum(chunk * L) / w)
    return np.array(flux)


def _iso_between(
    IMG,
    sma_low,
    sma_high,
    PARAMS,
    c,
    more=False,
    mask=None,
    sigmaclip=False,
    sclip_iterations=10,
    sclip_nsigma=5,
):

    if not "m" in PARAMS:
        PARAMS["m"] = None
    if not "C" in PARAMS:
        PARAMS["C"] = None
    Rlim = sma_high * (
        1.0
        if PARAMS["m"] is None
        else np.exp(sum(np.abs(PARAMS["Am"][m]) for m in range(len(PARAMS["m"]))))
    )
    ranges = [
        [max(0, int(c["x"] - Rlim - 2)), min(IMG.shape[1], int(c["x"] + Rlim + 2))],
        [max(0, int(c["y"] - Rlim - 2)), min(IMG.shape[0], int(c["y"] + Rlim + 2))],
    ]
    XX, YY = np.meshgrid(
        np.arange(ranges[0][1] - ranges[0][0], dtype=float)
        - c["x"]
        + float(ranges[0][0]),
        np.arange(ranges[1][1] - ranges[1][0], dtype=float)
        - c["y"]
        + float(ranges[1][0]),
    )
    theta = np.arctan2(YY, XX) #np.arctan(YY / XX) + np.pi * (XX < 0)
    RR = np.sqrt(XX ** 2 + YY ** 2)
    Fmode_Rscale = (
        1.0
        if PARAMS["m"] is None
        else Rscale_Fmodes(
            theta - PARAMS["pa"], PARAMS["m"], PARAMS["Am"], PARAMS["Phim"]
        )
    )
    SuperEllipse_Rscale = Rscale_SuperEllipse(
        theta - PARAMS["pa"], PARAMS["ellip"], 2 if PARAMS["C"] is None else PARAMS["C"]
    )
    RR /= SuperEllipse_Rscale * Fmode_Rscale
    rselect = np.logical_and(RR < sma_high, RR > sma_low)
    fluxes = IMG[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]][rselect]
    CHOOSE = None
    if not mask is None and sma_high > 5:
        CHOOSE = np.logical_not(
            mask[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]][rselect]
        )
    # Perform sigma clipping if requested
    if sigmaclip:
        sclim = Sigma_Clip_Upper(fluxes, sclip_iterations, sclip_nsigma)
        if CHOOSE is None:
            CHOOSE = fluxes < sclim
        else:
            CHOOSE = np.logical_or(CHOOSE, fluxes < sclim)
    if CHOOSE is not None and np.sum(CHOOSE) < 5:
        logging.warning(
            "Entire Isophote is Masked! R_l: %.3f, R_h: %.3f, PA: %.3f, ellip: %.3f"
            % (sma_low, sma_high, PARAMS["pa"] * 180 / np.pi, PARAMS["ellip"])
        )
        CHOOSE = np.ones(CHOOSE.shape).astype(bool)
    if CHOOSE is not None:
        countmasked = np.sum(np.logical_not(CHOOSE))
    else:
        countmasked = 0
    if more:
        if CHOOSE is not None and sma_high > 5:
            return fluxes[CHOOSE], theta[rselect][CHOOSE], countmasked
        else:
            return fluxes, theta[rselect], countmasked
    else:
        if CHOOSE is not None and sma_high > 5:
            return fluxes[CHOOSE]
        else:
            return fluxes


def _iso_extract(
    IMG,
    sma,
    PARAMS,
    c,
    more=False,
    minN=None,
    mask=None,
    interp_mask=False,
    rad_interp=30,
    interp_method="lanczos",
    interp_window=5,
    sigmaclip=False,
    sclip_iterations=10,
    sclip_nsigma=5,
):
    """
    Internal, basic function for extracting the pixel fluxes along an isophote
    """
    if not "m" in PARAMS:
        PARAMS["m"] = None
    if not "C" in PARAMS:
        PARAMS["C"] = None
    N = max(15, int(0.9 * 2 * np.pi * sma))
    if not minN is None:
        N = max(minN, N)
    # points along ellipse to evaluate
    theta = np.linspace(0, 2 * np.pi * (1.0 - 1.0 / N), N)
    theta = np.arctan((1.0 - PARAMS["ellip"]) * np.tan(theta)) + np.pi * (
        np.cos(theta) < 0
    )
    Fmode_Rscale = (
        1.0
        if PARAMS["m"] is None
        else Rscale_Fmodes(theta, PARAMS["m"], PARAMS["Am"], PARAMS["Phim"])
    )
    R = sma * Fmode_Rscale
    # Define ellipse
    X, Y = parametric_SuperEllipse(
        theta, PARAMS["ellip"], 2 if PARAMS["C"] is None else PARAMS["C"]
    )
    X, Y = R * X, R * Y
    # rotate ellipse by PA
    X, Y = Rotate_Cartesian(PARAMS["pa"], X, Y)
    theta = (theta + PARAMS["pa"]) % (2 * np.pi)
    # shift center
    X, Y = X + c["x"], Y + c["y"]

    # Reject samples from outside the image
    BORDER = np.logical_and(
        np.logical_and(X >= 0, X < (IMG.shape[1] - 1)),
        np.logical_and(Y >= 0, Y < (IMG.shape[0] - 1)),
    )
    if not np.all(BORDER):
        X = X[BORDER]
        Y = Y[BORDER]
        theta = theta[BORDER]

    Rlim = np.max(R)
    if Rlim < rad_interp:
        box = [
            [max(0, int(c["x"] - Rlim - 5)), min(IMG.shape[1], int(c["x"] + Rlim + 5))],
            [max(0, int(c["y"] - Rlim - 5)), min(IMG.shape[0], int(c["y"] + Rlim + 5))],
        ]
        if interp_method == "bicubic":
            flux = interpolate_bicubic(
                IMG[box[1][0] : box[1][1], box[0][0] : box[0][1]],
                X - box[0][0],
                Y - box[1][0],
            )
        elif interp_method == "lanczos":
            flux = interpolate_Lanczos(IMG, X, Y, interp_window)
        else:
            raise ValueError(
                "Unknown interpolate method %s. Should be one of lanczos or bicubic"
                % interp_method
            )
    else:
        # round to integers and sample pixels values
        flux = IMG[np.rint(Y).astype(np.int32), np.rint(X).astype(np.int32)]
    # CHOOSE holds bolean array for which flux values to keep, initialized as None for no clipping
    CHOOSE = None
    # Mask pixels if a mask is given
    if not mask is None:
        CHOOSE = np.logical_not(
            mask[np.rint(Y).astype(np.int32), np.rint(X).astype(np.int32)]
        )
    # Perform sigma clipping if requested
    if sigmaclip:
        sclim = Sigma_Clip_Upper(flux, sclip_iterations, sclip_nsigma)
        if CHOOSE is None:
            CHOOSE = flux < sclim
        else:
            CHOOSE = np.logical_or(CHOOSE, flux < sclim)
    # Dont clip pixels if that removes all of the pixels
    countmasked = 0
    if not CHOOSE is None and np.sum(CHOOSE) <= 0:
        logging.warning(
            "Entire Isophote is Masked! R: %.3f, PA: %.3f, ellip: %.3f"
            % (sma, PARAMS["pa"] * 180 / np.pi, PARAMS["ellip"])
        )
        # Interpolate clipped flux values if requested
    elif not CHOOSE is None and interp_mask:
        flux[np.logical_not(CHOOSE)] = np.interp(
            theta[np.logical_not(CHOOSE)], theta[CHOOSE], flux[CHOOSE], period=2 * np.pi
        )
        # simply remove all clipped pixels if user doesn't reqest another option
    elif not CHOOSE is None:
        flux = flux[CHOOSE]
        theta = theta[CHOOSE]
        countmasked = np.sum(np.logical_not(CHOOSE))

    # Return just the flux values, or flux and angle values
    if more:
        return flux, theta, countmasked
    else:
        return flux


def _iso_line(IMG, length, width, pa, c, more=False):

    start = np.array([c["x"], c["y"]])
    end = start + length * np.array([np.cos(pa), np.sin(pa)])

    ranges = [
        [
            max(0, int(min(start[0], end[0]) - 2)),
            min(IMG.shape[1], int(max(start[0], end[0]) + 2)),
        ],
        [
            max(0, int(min(start[1], end[1]) - 2)),
            min(IMG.shape[0], int(max(start[1], end[1]) + 2)),
        ],
    ]
    XX, YY = np.meshgrid(
        np.arange(ranges[0][1] - ranges[0][0], dtype=float),
        np.arange(ranges[1][1] - ranges[1][0], dtype=float),
    )
    XX -= c["x"] - float(ranges[0][0])
    YY -= c["y"] - float(ranges[1][0])
    XX, YY = (XX * np.cos(-pa) - YY * np.sin(-pa), XX * np.sin(-pa) + YY * np.cos(-pa))

    lselect = np.logical_and.reduce(
        (XX >= -0.5, XX <= length, np.abs(YY) <= (width / 2))
    )
    flux = IMG[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]][lselect]

    if more:
        return flux, XX[lselect], YY[lselect]
    else:
        return flux, XX[lselect]


def StarFind(
    IMG,
    fwhm_guess,
    background_noise,
    mask=None,
    peakmax=None,
    detect_threshold=20.0,
    minsep=10.0,
    reject_size=10.0,
    maxstars=np.inf,
):
    """
    Find stars in an image, determine their fwhm and peak flux values.

    IMG: image data as numpy 2D array
    fwhm_guess: A guess at the PSF fwhm, can be within a factor of 2 and everything should work
    background_noise: image background flux noise
    mask: masked pixels as numpy 2D array with same dimensions as IMG
    peakmax: maximum allowed peak flux value for a star, used to remove saturated pixels
    detect_threshold: threshold (in units of sigma) value for convolved image to consider a pixel as a star candidate.
    minsep: minimum allowed separation between stars, in units of fwhm_guess
    reject_size: reject stars with fitted FWHM greater than this times the fwhm_guess
    maxstars: stop once this number of stars have been found, this is for speed purposes
    """

    # Convolve edge detector with image
    S = 3 ** np.array([1, 2, 3, 4, 5])
    S = int(S[np.argmin(np.abs(S / 3 - fwhm_guess))])
    zz = np.ones((S, S)) * -1
    zz[int(S / 3) : int(2 * S / 3), int(S / 3) : int(2 * S / 3)] = 8

    new = convolve2d(IMG, zz, mode="same")

    centers = np.array([])
    deformities = []
    fwhms = []
    peaks = []
    # Select pixels which edge detector identifies
    if mask is None:
        highpixels = np.argwhere(new > detect_threshold * iqr(new))
    else:
        use_mask = np.logical_and(np.logical_not(mask), np.isfinite(new))
        highpixels = np.argwhere(
            np.logical_and(new > detect_threshold * iqr(new[use_mask]), use_mask)
        )
    np.random.shuffle(highpixels)
    # meshgrid for 2D polynomial fit (pre-built for efficiency)
    xx, yy = np.meshgrid(np.arange(6), np.arange(6))
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

    for i in range(len(highpixels)):
        # reject if near an existing center
        if len(centers) != 0 and np.any(
            np.sqrt(np.sum((highpixels[i] - centers) ** 2, axis=1))
            < minsep * fwhm_guess
        ):
            continue
        # reject if near edge
        if np.any(highpixels[i] < 5 * fwhm_guess) or np.any(
            highpixels[i] > (np.array(IMG.shape) - 5 * fwhm_guess)
        ):
            continue
        # set starting point at local maximum pixel
        newcenter = np.array([highpixels[i][1], highpixels[i][0]])
        ranges = [
            [
                max(0, int(newcenter[0] - fwhm_guess * 5)),
                min(IMG.shape[1], int(newcenter[0] + fwhm_guess * 5)),
            ],
            [
                max(0, int(newcenter[1] - fwhm_guess * 5)),
                min(IMG.shape[0], int(newcenter[1] + fwhm_guess * 5)),
            ],
        ]
        newcenter = np.unravel_index(
            np.argmax(IMG[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]].T),
            IMG[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]].T.shape,
        )
        newcenter += np.array([ranges[0][0], ranges[1][0]])
        if np.any(newcenter < 5 * fwhm_guess) or np.any(
            newcenter > (np.array(IMG.shape) - 5 * fwhm_guess)
        ):
            continue
        # update star center with 2D polynomial fit
        ranges = [
            [max(0, int(newcenter[0] - 3)), min(IMG.shape[1], int(newcenter[0] + 3))],
            [max(0, int(newcenter[1] - 3)), min(IMG.shape[0], int(newcenter[1] + 3))],
        ]
        chunk = np.clip(
            IMG[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]].T,
            a_min=background_noise / 3,
            a_max=None,
        )
        poly2dfit = np.linalg.lstsq(A, np.log10(chunk.flatten()), rcond=None)
        newcenter = np.array(
            [
                -poly2dfit[0][2] / (2 * poly2dfit[0][4]),
                -poly2dfit[0][1] / (2 * poly2dfit[0][3]),
            ]
        )
        # reject if 2D polynomial maximum is outside the fitting region
        if np.any(newcenter < 0) or np.any(newcenter > 5):
            continue
        newcenter += np.array([ranges[0][0], ranges[1][0]])

        # reject centers that are outside the image
        if np.any(newcenter < 5 * fwhm_guess) or np.any(
            newcenter > (np.array(list(reversed(IMG.shape))) - 5 * fwhm_guess)
        ):
            continue
        # reject stars with too high flux
        if (not peakmax is None) and np.any(
            IMG[
                int(newcenter[1] - minsep * fwhm_guess) : int(
                    newcenter[1] + minsep * fwhm_guess
                ),
                int(newcenter[0] - minsep * fwhm_guess) : int(
                    newcenter[0] + minsep * fwhm_guess
                ),
            ]
            >= peakmax
        ):
            continue
        # reject if near existing center
        if len(centers) != 0 and np.any(
            np.sqrt(np.sum((newcenter - centers) ** 2, axis=1)) < minsep * fwhm_guess
        ):
            continue
        if not np.all(np.isfinite(newcenter)):
            continue

        # Extract flux as a function of radius
        local_flux = np.median(
            _iso_extract(
                IMG,
                reject_size * fwhm_guess,
                {"ellip": 0.0, "pa": 0.0},
                {"x": newcenter[0], "y": newcenter[1]},
                mask = mask,
                interp_method = 'bicubic',
            )
        )
        flux = [
            np.median(
                _iso_extract(
                    IMG,
                    0.0,
                    {"ellip": 0.0, "pa": 0.0},
                    {"x": newcenter[0], "y": newcenter[1]},
                    mask = mask,
                    interp_method = 'bicubic',
                )
            )
            - local_flux
        ]
        if (flux[0] - local_flux) < (detect_threshold * background_noise):
            continue
        R = [0.0]
        deformity = [0.0]
        badcount = 0
        while (flux[-1] > max(flux[0] / 2, background_noise) or len(R) < 5) and len(
            R
        ) < 50:  # len(R) < 50 and (flux[-1] > background_noise or len(R) <= 5):
            R.append(R[-1] + fwhm_guess / 10)
            try:
                isovals = _iso_extract(
                    IMG,
                    R[-1],
                    {"ellip": 0.0, "pa": 0.0},
                    {"x": newcenter[0], "y": newcenter[1]},
                    mask = mask,
                    interp_method = 'bicubic',
                )
            except:
                R = np.zeros(101)  # cause finder to skip this star
                break
            coefs = fft(isovals)
            deformity.append(
                np.sum(np.abs(coefs[1:5]))
                / (len(isovals) * (max(np.median(isovals), 0) + background_noise))
            )  # np.sqrt(np.abs(coefs[0]))
            # if np.sum(np.abs(coefs[1:5])) > np.sqrt(np.abs(coefs[0])):
            #     badcount += 1
            flux.append(np.median(isovals) - local_flux)
        if len(R) >= 50:
            continue
        fwhm_fit = np.interp(flux[0] / 2, list(reversed(flux)), list(reversed(R))) * 2

        # reject if fitted FWHM unrealistically large
        if fwhm_fit > reject_size * fwhm_guess or fwhm_fit < 0 or not np.isfinite(fwhm_fit):
            continue
        # Add star to list
        if len(centers) == 0:
            centers = np.array([deepcopy(newcenter)])
        else:
            centers = np.concatenate((centers, [newcenter]), axis=0)
        deformities.append(deformity[-1])
        fwhms.append(deepcopy(fwhm_fit))
        peaks.append(flux[0])
        # stop if max N stars reached
        if len(fwhms) >= maxstars:
            break

        # ranges = [[max(0,int(newcenter[0]-fwhm_guess*5)), min(IMG.shape[1],int(newcenter[0]+fwhm_guess*5))],
        #           [max(0,int(newcenter[1]-fwhm_guess*5)), min(IMG.shape[0],int(newcenter[1]+fwhm_guess*5))]]
        # plt.imshow(np.clip(IMG[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]],
        #                    a_min = 0,a_max = None), origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        # plt.scatter([newcenter[0] - ranges[0][0]], [newcenter[1] - ranges[1][0]], color = 'r', marker = 'x')
        # plt.scatter([com_center[0] - ranges[0][0]], [com_center[1] - ranges[1][0]], color = 'b', marker = 'x')
        # #plt.scatter([highpixels[i][1] - ranges[0][0]], [highpixels[i][0] - ranges[1][0]], color = 'g', marker = 'x')
        # plt.savefig('test/PSF_test_%i_center.jpg' % randid)
        # plt.close()
    return {
        "x": centers[:, 0],
        "y": centers[:, 1],
        "fwhm": np.array(fwhms),
        "peak": np.array(peaks),
        "deformity": np.array(deformities),
    }


def _x_to_pa(x):
    """
    Internal, basic function to ensure position angles remain
    in the proper parameter space (0, pi)
    """
    return x % np.pi  # np.pi / (1. + np.exp(-(x-np.pi/2)))


def _inv_x_to_pa(pa):
    """
    Internal, reverse _x_to_pa, although this is just a filler
    function as _x_to_pa is non reversible.
    """
    return pa % np.pi


def PA_shift_convention(pa, deg=False):
    """
    Alternates between standard mathematical convention for angles, and astronomical position angle convention.
    The standard convention is to measure angles counter-clockwise relative to the positive x-axis
    The astronomical convention is to measure angles counter-clockwise relative to the positive y-axis
    """

    if deg:
        shift = 180.0
    else:
        shift = np.pi
    return (pa - (shift / 2)) % shift


def _x_to_eps(x):
    """
    Internal, function to map the reals to the range (0.0,1.0)
    as the range of reasonable ellipticity values.
    """
    return 0.5 + np.arctan(x - 0.5) / np.pi  # 0.02 + 0.96/(1. + np.exp(-(x - 0.5)))


def _inv_x_to_eps(eps):
    """
    Internal, inverse of _x_to_eps function
    """
    return 0.5 + np.tan(np.pi * (eps - 0.5))  # 0.5 - np.log(0.96/(eps - 0.02) - 1.)


def Read_Image(filename, options):
    """
    Reads a galaxy image given a file name. In a fits image the data is assumed to exist in the
    primary HDU unless given 'hdulelement'. In a numpy file, it is assumed that only one image
    is in the file.

    filename: A string containing the full path to an image file

    returns: Extracted image data as numpy 2D array
    """

    # Read a fits file
    if filename[filename.rfind(".") + 1 :].lower() == "fits":
        hdul = fits.open(filename, uint = False)
        dat = hdul[options["ap_hdulelement"] if "ap_hdulelement" in options else 0].data
    # Read a numpy array file
    if filename[filename.rfind(".") + 1 :].lower() == "npy":
        dat = np.load(filename)

    return np.require(dat, dtype=float)


def Angle_TwoAngles_sin(a1, a2):
    """
    Compute the angle between two vectors at angles a1 and a2
    """

    return np.arcsin(np.sin(a1 - a2))


def Angle_TwoAngles_cos(a1, a2):
    """
    Compute the angle between two vectors at angles a1 and a2
    """

    return np.arccos(np.cos(a1 - a2))


def Angle_Average(a):
    """
    Compute the average for a list of angles, which may wrap around a cyclic boundary.

    a: list of angles in the range [0,2pi]
    """
    i = np.cos(a) + 1j * np.sin(a)
    return np.angle(np.mean(i))


def Angle_Median(a):
    """
    Compute the median for a list of angles, which may wrap around a cyclic boundary.

    a: list of angles in the range [0,2pi]
    """
    i = np.median(np.cos(a)) + 1j * np.median(np.sin(a))
    return np.angle(i)


def Angle_Scatter(a):
    """
    Compute the scatter for a list of angles, which may wrap around a cyclic boundary.

    a: list of angles in the range [0,2pi]
    """
    i = np.cos(a) + 1j * np.sin(a)
    return iqr(np.angle(1j * i / np.mean(i)), rng=[16, 84])


def GetOptions(c):
    """
    Extract all of the AutoProf user optionional parameters form the config file.
    User options are identified as any python object that starts with "ap\_" in the
    variable name.
    """
    newoptions = {}
    for var in dir(c):
        if var.startswith("ap_"):
            val = getattr(c, var)
            if not val is None:
                newoptions[var] = val

    return newoptions


def fluxdens_to_fluxsum(R, I, axisratio):
    """
    Integrate a flux density profile

    R: semi-major axis length (arcsec)
    I: flux density (flux/arcsec^2)
    axisratio: b/a profile
    """

    S = np.zeros(len(R))
    S[0] = I[0] * np.pi * axisratio[0] * (R[0] ** 2)
    for i in range(1, len(R)):
        S[i] = (
            trapz(2 * np.pi * I[: i + 1] * R[: i + 1] * axisratio[: i + 1], R[: i + 1])
            + S[0]
        )
    return S


def fluxdens_to_fluxsum_errorprop(
    R, I, IE, axisratio, axisratioE=None, N=100, symmetric_error=True
):
    """
    Integrate a flux density profile

    R: semi-major axis length (arcsec)
    I: flux density (flux/arcsec^2)
    axisratio: b/a profile
    """
    if axisratioE is None:
        axisratioE = np.zeros(len(R))

    # Create container for the monte-carlo iterations
    sum_results = np.zeros((N, len(R))) - 99.999
    I_CHOOSE = np.logical_and(np.isfinite(I), I > 0)
    if np.sum(I_CHOOSE) < 5:
        return (None, None) if symmetric_error else (None, None, None)
    sum_results[0][I_CHOOSE] = fluxdens_to_fluxsum(
        R[I_CHOOSE], I[I_CHOOSE], axisratio[I_CHOOSE]
    )
    for i in range(1, N):
        # Randomly sampled SB profile
        tempI = np.random.normal(loc=I, scale=np.abs(IE))
        # Randomly sampled axis ratio profile
        tempq = np.clip(
            np.random.normal(loc=axisratio, scale=np.abs(axisratioE)),
            a_min=1e-3,
            a_max=1 - 1e-3,
        )
        # Compute COG with sampled data
        sum_results[i][I_CHOOSE] = fluxdens_to_fluxsum(
            R[I_CHOOSE], tempI[I_CHOOSE], tempq[I_CHOOSE]
        )

    # Condense monte-carlo evaluations into profile and uncertainty envelope
    sum_lower = sum_results[0] - np.quantile(sum_results, 0.317310507863 / 2, axis=0)
    sum_upper = (
        np.quantile(sum_results, 1.0 - 0.317310507863 / 2, axis=0) - sum_results[0]
    )

    # Return requested uncertainty format
    if symmetric_error:
        return sum_results[0], np.abs(sum_lower + sum_upper) / 2
    else:
        return sum_results[0], sum_lower, sum_upper


def _Fmode_integrand(t, parameters):
    fsum = sum(
        parameters["Am"][m] * np.cos(parameters["m"][m] * (t + parameters["Phim"][m]))
        for m in range(len(parameters["m"]))
    )
    dfsum = sum(
        parameters["m"][m]
        * parameters["Am"][m]
        * np.sin(parameters["m"][m] * (t + parameters["Phim"][m]))
        for m in range(len(parameters["m"]))
    )
    return (np.sin(t) ** 2) * np.exp(2 * fsum) + np.sin(t) * np.cos(t) * np.exp(
        fsum
    ) * dfsum


def Fmode_Areas(R, parameters):

    A = []
    for i in range(len(R)):
        A.append(
            (R[i] ** 2) * quad(_Fmode_integrand, 0, 2 * np.pi, args=(parameters[i],))[0]
        )
    return np.array(A)


def Fmode_fluxdens_to_fluxsum(R, I, parameters, A=None):
    """
    Integrate a flux density profile, with isophotes including Fourier perturbations.

    Arguments
    ---------
    R: arcsec
      semi-major axis length

    I: flux/arcsec^2
      flux density

    parameters: list of dictionaries
      list of dictionary of isophote shape parameters for each radius.
      formatted as

      .. code-block:: python

        {'ellip': ellipticity,
         'm': list of modes used,
         'Am': list of mode powers,
         'Phim': list of mode phases

      }

      entries for each radius.
    """
    if all(parameters[p]["m"] is None for p in range(len(parameters))):
        return fluxdens_to_fluxsum(
            R,
            I,
            1.0
            - np.array(list(parameters[p]["ellip"] for p in range(len(parameters)))),
        )

    S = np.zeros(len(R))
    if A is None:
        A = Fmode_Areas(R, parameters)
    # update the Area calculation to be scaled by the ellipticity
    Aq = A * np.array(list((1 - parameters[i]["ellip"]) for i in range(len(R))))
    S[0] = I[0] * Aq[0]
    Adiff = np.array([Aq[0]] + list(Aq[1:] - Aq[:-1]))
    for i in range(1, len(R)):
        S[i] = trapz(I[: i + 1] * Adiff[: i + 1], R[: i + 1]) + S[0]
    return S


def Fmode_fluxdens_to_fluxsum_errorprop(
    R, I, IE, parameters, N=100, symmetric_error=True
):
    """
    Integrate a flux density profile, with isophotes including Fourier perturbations.

    Arguments
    ---------
    R: arcsec
      semi-major axis length

    I: flux/arcsec^2
      flux density

    parameters: list of dictionaries
      list of dictionary of isophote shape parameters for each radius.
      formatted as

      .. code-block:: python

        {'ellip': ellipticity,
         'm': list of modes used,
         'Am': list of mode powers,
         'Phim': list of mode phases

      }

      entries for each radius.
    """

    for i in range(len(R)):
        if not "ellip err" in parameters[i]:
            parameters[i]["ellip err"] = np.zeros(len(R))
    if all(parameters[p]["m"] is None for p in range(len(parameters))):
        return fluxdens_to_fluxsum_errorprop(
            R,
            I,
            IE,
            1.0
            - np.array(list(parameters[p]["ellip"] for p in range(len(parameters)))),
            np.array(list(parameters[p]["ellip err"] for p in range(len(parameters)))),
            N=N,
            symmetric_error=symmetric_error,
        )

    # Create container for the monte-carlo iterations
    sum_results = np.zeros((N, len(R))) - 99.999
    I_CHOOSE = np.logical_and(np.isfinite(I), I > 0)
    if np.sum(I_CHOOSE) < 5:
        return (None, None) if symmetric_error else (None, None, None)
    cut_parameters = list(compress(parameters, I_CHOOSE))
    A = Fmode_Areas(R[I_CHOOSE], cut_parameters)
    sum_results[0][I_CHOOSE] = Fmode_fluxdens_to_fluxsum(
        R[I_CHOOSE], I[I_CHOOSE], cut_parameters, A
    )
    for i in range(1, N):
        # Randomly sampled SB profile
        tempI = np.random.normal(loc=I, scale=np.abs(IE))
        # Randomly sampled axis ratio profile
        temp_parameters = deepcopy(cut_parameters)
        for p in range(len(cut_parameters)):
            temp_parameters[p]["ellip"] = np.clip(
                np.random.normal(
                    loc=cut_parameters[p]["ellip"],
                    scale=np.abs(cut_parameters[p]["ellip err"]),
                ),
                a_min=1e-3,
                a_max=1 - 1e-3,
            )
        # Compute COG with sampled data
        sum_results[i][I_CHOOSE] = Fmode_fluxdens_to_fluxsum(
            R[I_CHOOSE], tempI[I_CHOOSE], temp_parameters, A
        )

    # Condense monte-carlo evaluations into profile and uncertainty envelope
    sum_lower = sum_results[0] - np.quantile(sum_results, 0.317310507863 / 2, axis=0)
    sum_upper = (
        np.quantile(sum_results, 1.0 - 0.317310507863 / 2, axis=0) - sum_results[0]
    )

    # Return requested uncertainty format
    if symmetric_error:
        return sum_results[0], np.abs(sum_lower + sum_upper) / 2
    else:
        return sum_results[0], sum_lower, sum_upper


def SBprof_to_COG(R, SB, parameters):
    """
    Converts a surface brightness profile to a curve of growth by integrating
    the SB profile in flux units then converting back to mag units. Two methods
    are implemented, one using the trapezoid method and one assuming constant
    SB between isophotes. Trapezoid method is in principle more accurate, but
    may become unstable with erratic data.

    R: Radius in arcsec
    SB: surface brightness in mag arcsec^-2
    axisratio: b/a indicating degree of isophote ellipticity
    method: 0 for trapezoid, 1 for constant

    returns: magnitude values at each radius of the profile in mag
    """

    # m = np.zeros(len(R))
    # # Dummy band for conversions, cancelled out on return
    # band = 'r'

    # # Method 0 uses trapezoid method to integrate in flux space
    # if method == 0:
    #     # Compute the starting point assuming constant SB within first isophote
    #     m[0] = magperarcsec2_to_mag(SB[0], A = np.pi*axisratio[0]*(R[0]**2))
    #     # Dummy distance value for integral
    #     D = 10
    #     # Convert to flux space
    #     I = muSB_to_ISB(np.array(SB), band)
    #     # Ensure numpy array
    #     axisratio = np.array(axisratio)
    #     # Convert to physical radius using dummy distance
    #     R = arcsec_to_pc(np.array(R), D)
    #     # Integrate up to each radius in the profile
    #     for i in range(1,len(R)):
    #         m[i] = abs_mag_to_app_mag(L_to_mag(trapz(2*np.pi*I[:i+1]*R[:i+1]*axisratio[:i+1],R[:i+1]) + \
    #                                            mag_to_L(app_mag_to_abs_mag(m[0], D), band), band),D)
    # elif method == 1:
    #     # Compute the starting point assuming constant SB within first isophote
    #     m[0] = magperarcsec2_to_mag(SB[0], A = np.pi*axisratio[0]*(R[0]**2))
    #     # Progress through each radius and add the contribution from each isophote individually
    #     for i in range(1,len(R)):
    #         m[i] = L_to_mag(mag_to_L(magperarcsec2_to_mag(SB[i], A = np.pi*axisratio[i]*(R[i]**2)), band) - \
    #                         mag_to_L(magperarcsec2_to_mag(SB[i], A = np.pi*axisratio[i-1]*(R[i-1]**2)), band) + \
    #                         mag_to_L(m[i-1], band), band)

    return flux_to_mag(
        Fmode_fluxdens_to_fluxsum(R, mag_to_flux(SB, 20.), parameters), 20.
    )


def SBprof_to_COG_errorprop(R, SB, SBE, parameters, N=100, symmetric_error=True):
    """
    Converts a surface brightness profile to a curve of growth by integrating
    the SB profile in flux units then converting back to mag units. Two methods
    are implemented, one using the trapezoid method and one assuming constant
    SB between isophotes. Trapezoid method is in principle more accurate, but
    may become unstable with erratic data. An uncertainty profile is also
    computed, from a given SB uncertainty profile and optional axisratio
    uncertainty profile.

    R: Radius in arcsec
    SB: surface brightness in mag arcsec^-2
    SBE: surface brightness uncertainty relative mag arcsec^-2
    axisratio: b/a indicating degree of isophote ellipticity
    axisratioE: uncertainty in b/a
    N: number of iterations for computing uncertainty
    method: 0 for trapezoid, 1 for constant

    returns: magnitude and uncertainty profile in mag
    """

    # # If not provided, axis ratio error is assumed to be zero
    # if axisratioE is None:
    #     axisratioE = np.zeros(len(R))

    # # Create container for the monte-carlo iterations
    # COG_results = np.zeros((N, len(R))) + 99.999
    # SB_CHOOSE = np.logical_and(np.isfinite(SB), SB < 99)
    # if np.sum(SB_CHOOSE) < 5:
    #     return (None, None) if symmetric_error else (None, None, None)
    # COG_results[0][SB_CHOOSE] = SBprof_to_COG(R[SB_CHOOSE], SB[SB_CHOOSE], axisratio[SB_CHOOSE], method = method)
    # for i in range(1,N):
    #     # Randomly sampled SB profile
    #     tempSB = np.random.normal(loc = SB, scale = np.abs(SBE))
    #     # Randomly sampled axis ratio profile
    #     tempq = np.random.normal(loc = axisratio, scale = np.abs(axisratioE))
    #     # Compute COG with sampled data
    #     COG_results[i][SB_CHOOSE] = SBprof_to_COG(R[SB_CHOOSE], tempSB[SB_CHOOSE], tempq[SB_CHOOSE], method = method)

    # # Condense monte-carlo evaluations into profile and uncertainty envelope
    # COG_profile = COG_results[0]
    # COG_lower = COG_profile - np.quantile(COG_results, 0.317310507863/2, axis = 0)
    # COG_upper = np.quantile(COG_results, 1. - 0.317310507863/2, axis = 0) - COG_profile

    # # Return requested uncertainty format
    # if symmetric_error:
    #     return COG_profile, np.abs(COG_lower + COG_upper)/2
    # else:
    #     return COG_profile, COG_lower, COG_upper
    I, Ie = mag_to_flux(SB, 20., SBE)
    res = Fmode_fluxdens_to_fluxsum_errorprop(
        R, I, Ie, parameters, N=N, symmetric_error=symmetric_error
    )
    if symmetric_error:
        return flux_to_mag(res[0], 20., res[1])
    else:
        lower = flux_to_mag(res[0], 20., res[1])
        upper = flux_to_mag(res[0], 20., res[2])
        return lower[0], lower[1], upper[1]
