from photutils import DAOStarFinder, IRAFStarFinder
import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import logging


def Calculate_PSF(IMG, pixscale, name, results, **kwargs):
    """
    Idenitfy the location of stars in the image and calculate
    their average PSF.

    IMG: numpy 2d array of pixel values
    pixscale: conversion factor from pixels to arcsec (arcsec pixel^-1)
    """

    # Guess for PSF based on user provided seeing value
    fwhm_guess = max(1. / pixscale, 1)

    # photutils wrapper for IRAF star finder
    iraffind = IRAFStarFinder(fwhm = fwhm_guess, threshold = 20.*results['background']['iqr'])
    irafsources = iraffind(IMG - results['background']['median'])

    logging.info('%s: found psf: %f' % (name,np.median(irafsources['fwhm'])))
    
    # Return PSF statistics
    return {'median': np.median(irafsources['fwhm']),
            'mean': np.mean(irafsources['fwhm']),
            'std': np.std(irafsources['fwhm']),
            'iqr': iqr(irafsources['fwhm'])}
    
def Given_PSF(IMG, pixscale, name, results, **kwargs):
    """
    Uses the kwarg "given_psf" to return a user inputted psf.
    The given_psf object should be a float representing all images or
    a dictionary where each key is a galaxy name (as in the names
    given for each galaxy) and the value is a dictionary structured
    as {'median': float, 'mean': float, 'std': float, 'iqr': float}
    where the values give psf statistics in pixel units. If the
    galaxy name is not in the given_psf dictionary it will return
    the standard "Calculate_PSF" result.

    IMG: numpy 2d array of pixel values
    pixelscale: conversion factor from pixels to arcsec (arcsec pixel^-1)
    """

    try:
        if type(kwargs['given_psf']) == float:
            return kwargs['given_psf']
        else:
            return kwargs['given_psf'][name]
    except:
        return Calculate_PSF(IMG, pixscale, name, results, **kwargs)
