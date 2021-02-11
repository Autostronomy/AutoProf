from photutils import DAOStarFinder, IRAFStarFinder
import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib.patches import Ellipse
import logging
from itertools import product
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal

def _GaussFit(x, dat, xx, yy, noise, fwhm_guess):

    loss = []
    for i in range(len(xx)):
        coords = np.array(list(product(range(max(0,int(xx[i] - 6*fwhm_guess)), min(dat.shape[0]-1,int(xx[i] + 6*fwhm_guess))),
                                       range(max(0,int(yy[i] - 6*fwhm_guess)), min(dat.shape[1]-1,int(yy[i] + 6*fwhm_guess))))))
        chunk = dat[coords[:,1], coords[:,0]]
        CHOOSE = chunk > 3*noise
        if np.sum(CHOOSE) < 10:
            continue
        fluxsum = np.sum(chunk)
        model_vals = fluxsum*multivariate_normal.pdf(coords[CHOOSE], mean = [xx[i],yy[i]], cov = [[x[0],0.],[0.,x[0]]]) + noise
        loss.append(np.sum(np.abs((model_vals - chunk[CHOOSE])**2/chunk[CHOOSE])))
    
    return np.mean(sorted(loss)[:-5])

def PSF_2DGaussFit(IMG, pixscale, name, results, **kwargs):

    fwhm_guess = max(1. / pixscale, 1)
    edge_mask = np.zeros(IMG.shape, dtype = bool)
    edge_mask[int(IMG.shape[0]/5.):int(4.*IMG.shape[0]/5.),
              int(IMG.shape[1]/5.):int(4.*IMG.shape[1]/5.)] = True

    # photutils wrapper for IRAF star finder
    count = 0
    while count < 5:
        iraffind = IRAFStarFinder(fwhm = fwhm_guess, threshold = 6.*results['background']['noise'], roundlo = 0.01, brightest = 20)
        irafsources = iraffind.find_stars(IMG - results['background']['background'], edge_mask)
        fwhm_guess = np.median(irafsources['fwhm'])
        if np.median(irafsources['sharpness']) >= 0.95:
            break
        count += 1
        
    if 'doplot' in kwargs and kwargs['doplot']:    
        plt.imshow(np.clip(IMG - results['background']['background'], a_min = 0, a_max = None), origin = 'lower',
                   cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        for i in range(len(irafsources['fwhm'])):
            plt.gca().add_patch(Ellipse((irafsources['xcentroid'][i],irafsources['ycentroid'][i]), 16/pixscale, 16/pixscale,
                                        0, fill = False, linewidth = 0.5, color = 'y'))
        plt.savefig('%sPSF_Stars_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), dpi = 600)
        plt.clf()
    
    res = minimize(_GaussFit, x0 = [(fwhm_guess/2.355)**2], args = (IMG - results['background']['background'], irafsources['xcentroid'], irafsources['ycentroid'], results['background']['noise'], fwhm_guess))
    logging.info('%s: found psf: %f' % (name,np.sqrt(res.x[0])*2.355))
    return {'fwhm': np.sqrt(res.x[0])*2.355}


def Calculate_PSF(IMG, pixscale, name, results, **kwargs):
    """
    Idenitfy the location of stars in the image and calculate
    their average PSF.

    IMG: numpy 2d array of pixel values
    pixscale: conversion factor from pixels to arcsec (arcsec pixel^-1)
    """
    
    # Guess for PSF based on user provided seeing value
    fwhm_guess = max(1. / pixscale, 1)
    edge_mask = np.zeros(IMG.shape, dtype = bool)
    edge_mask[int(IMG.shape[0]/5.):int(4.*IMG.shape[0]/5.),
              int(IMG.shape[1]/5.):int(4.*IMG.shape[1]/5.)] = True

    # photutils wrapper for IRAF star finder
    count = 0
    while count < 5:
        iraffind = IRAFStarFinder(fwhm = fwhm_guess, threshold = 6.*results['background']['noise'], roundlo = 0.01)
        irafsources = iraffind.find_stars(IMG - results['background']['background'], edge_mask)
        logging.info('%s: psf found %i objects (min sharpness %.2e, med sherp %.2e, max sharp %.2e)' % (name, len(irafsources['fwhm']), np.min(irafsources['sharpness']), np.median(irafsources['sharpness']), np.max(irafsources['sharpness'])))
        fwhm_guess = np.median(irafsources['fwhm'])
        hist,bins = np.histogram(irafsources['fwhm'], bins = 25)
        plt.bar(bins[:-1], hist, width = bins[1] - bins[0], align = 'edge')
        plt.savefig('plots/psftest_%s_%i.png' % (name,count))
        plt.clf()
        if np.median(irafsources['sharpness']) >= 0.95:
            break
        count += 1
    if 'doplot' in kwargs and kwargs['doplot']:    
        plt.imshow(np.clip(IMG - results['background']['background'], a_min = 0, a_max = None), origin = 'lower',
                   cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        for i in range(len(irafsources['fwhm'])):
            plt.gca().add_patch(Ellipse((irafsources['xcentroid'][i],irafsources['ycentroid'][i]), 16/pixscale, 16/pixscale,
                                    0, fill = False, linewidth = 0.5, color = 'y'))
        plt.savefig('%sPSF_Stars_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), dpi = 600)
        plt.clf()                
    
    logging.info('%s: found psf: %f' % (name,np.median(irafsources['fwhm'])))
    
    # Return PSF statistics
    return {'fwhm': fwhm_guess}
    
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
