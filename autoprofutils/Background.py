from photutils import make_source_mask
from photutils.isophote import EllipseSample, Ellipse, EllipseGeometry, Isophote, IsophoteList
from astropy.stats import sigma_clipped_stats
from scipy.stats import iqr
from scipy.optimize import minimize
from scipy.fftpack import fft2, ifft2
from time import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize


def Background_Mode(IMG, pixscale, name, results, **kwargs):
    """
    Compute background by finding the peak in a smoothed histogram of flux values.
    This should correspond to the peak of the noise floor.
    
    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """
    # Mask main body of image so only outer 1/5th is used
    # for background calculation.
    if 'mask' in results and not results['mask'] is None and np.any(results['mask']):
        mask = np.logical_not(results['mask'])
        logging.info('%s: Background using segmentation map mask. Masking %i pixels' % (name, np.sum(results['mask'])))
    else:
        mask = np.ones(IMG.shape, dtype = bool)
        mask[int(IMG.shape[0]/5.):int(4.*IMG.shape[0]/5.),
             int(IMG.shape[1]/5.):int(4.*IMG.shape[1]/5.)] = False
    values = IMG[mask].flatten()
    values = values[np.isfinite(values)]

    if 'background' in kwargs:
        bkgrnd = kwargs['background']
        logging.info('%s: Background set by user: %.4e' % (name, bkgrnd))
    else:
        # set the starting point for the sky level optimization at the median pixel flux
        start = np.quantile(values, 0.40)
        # set the smoothing scale equal to roughly 0.5% of the width of the data
        scale = iqr(values,rng = [30,70])/10
        
        # Fit the peak of the smoothed histogram
        res = minimize(lambda x: -np.sum(np.exp(-((values - x)/scale)**2)), x0 = [start], method = 'Nelder-Mead')
        bkgrnd = res.x[0]
    # Compute the 1sigma range using negative flux values, which should almost exclusively be sky noise
    if 'background_noise' in kwargs:
        noise = kwargs['background_noise']
        logging.info('%s: Background Noise set by user: %.4e' % (name, noise))
    else:
        noise = iqr(values[(values-bkgrnd) < 0], rng = [100 - 68.2689492137,100])
    uncertainty = noise / np.sqrt(np.sum((values-bkgrnd) < 0))
    
    if 'doplot' in kwargs and kwargs['doplot']:    
        hist, bins = np.histogram(values[np.logical_and((values-bkgrnd) < 20*noise, (values-bkgrnd) > -3*noise)], bins = 1000)
        plt.bar(bins[:-1], np.log10(hist), width = bins[1] - bins[0], color = 'k', label = 'pixel values')
        plt.axvline(bkgrnd, color = 'r', label = 'sky level: %.5e' % bkgrnd)
        plt.axvline(bkgrnd - noise, color = 'r', linestyle = '--', label = '1$\\sigma$ noise/pix: %.5e' % noise)
        plt.axvline(bkgrnd + noise, color = 'r', linestyle = '--')
        plt.legend()
        plt.xlabel('flux')
        plt.ylabel('log$_{10}$(count)')
        plt.savefig('%sBackground_hist_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
        plt.close()        
        
    return {'background': bkgrnd,
            'background noise': noise,
            'background uncertainty': uncertainty}

def Background_DilatedSources(IMG, pixscale, name, results, **kwargs):
    """
    Compute a global background value for an image. Performed by
    identifying pixels which are beyond 3 sigma above the average
    signal and masking them, also further masking a boarder
    of 20 pixels around the initial masked pixels. Returns a
    dictionary of parameters describing the background level.
    """

    # Mask main body of image so only outer 1/5th is used
    # for background calculation.
    if 'mask' in results and not results['mask'] is None:
        mask = results['mask']
    else:
        mask = np.zeros(IMG.shape)
        mask[int(IMG.shape[0]/5.):int(4.*IMG.shape[0]/5.),
             int(IMG.shape[1]/5.):int(4.*IMG.shape[1]/5.)] = 1

    # Run photutils source mask to remove pixels with sources
    # such as stars and galaxies, including a boarder
    # around each source.
    source_mask = make_source_mask(IMG,
                                   nsigma = 3,
                                   npixels = int(1./pixscale),
                                   dilate_size = 40,
                                   filter_fwhm = 1./pixscale,
                                   filter_size = int(3./pixscale),
                                   sigclip_iters = 5)
    mask = np.logical_or(mask, source_mask)

    # Return statistics from background sky
    noise = iqr(IMG[np.logical_not(mask)],rng = [16,84])/2
    return {'background': np.median(IMG[np.logical_not(mask)]),
            'background noise': noise,
            'background uncertainty': noise/np.sqrt(np.sum(np.logical_not(mask)))}

def Background_Unsharp(IMG, pixscale, name, results, **kwargs):

    coefs = fft2(IMG)

    unsharp = 3
    coefs[unsharp:-unsharp] = 0
    coefs[:,unsharp:-unsharp] = 0

    stats = Background_Mode(IMG, pixscale, name, results, **kwargs)
    stats.update({'background': ifft2(coefs).real})
    return stats
            

    
