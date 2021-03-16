from photutils import make_source_mask
from photutils.isophote import EllipseSample, Ellipse, EllipseGeometry, Isophote, IsophoteList
from astropy.stats import sigma_clipped_stats
from scipy.stats import iqr
from scipy.optimize import minimize
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
    edge_mask = np.ones(IMG.shape, dtype = bool)
    edge_mask[int(IMG.shape[0]/5.):int(4.*IMG.shape[0]/5.),
              int(IMG.shape[1]/5.):int(4.*IMG.shape[1]/5.)] = False
    values = IMG[edge_mask].flatten()
    values = values[np.isfinite(values)]
    # set the starting point for the sky level optimization at the median pixel flux
    start = np.median(values)
    # set the smoothing scale equal to roughly 1% of the width of the data
    scale = iqr(values,rng = [30,70])/40

    # Fit the peak of the smoothed histogram
    res = minimize(lambda x: -np.sum(np.exp(-((values - x)/scale)**2)), x0 = [start], method = 'Nelder-Mead')

    # Compute the 1sigma range using negative flux values, which should almost exclusively be sky noise 
    noise = iqr(values[(values-res.x[0]) < 0], rng = [100 - 68.2689492137,100])
    
    if 'doplot' in kwargs and kwargs['doplot']:    
        hist, bins = np.histogram(values[np.logical_and((values-res.x[0]) < 20*noise, (values-res.x[0]) > -3*noise)], bins = 1000)
        plt.bar(bins[:-1], np.log10(hist), width = bins[1] - bins[0], color = 'k', label = 'pixel values')
        plt.axvline(res.x[0], color = 'r', label = 'sky level: %.5e' % res.x[0])
        plt.axvline(res.x[0] - noise, color = 'r', linestyle = '--', label = '1$\\sigma$ noise/pix: %.5e' % noise)
        plt.axvline(res.x[0] + noise, color = 'r', linestyle = '--')
        plt.legend()
        plt.xlabel('flux')
        plt.ylabel('log$_{10}$(count)')
        plt.savefig('%sBackground_hist_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
        plt.close()
        
    return {'background': res.x[0],
            'background noise': noise}

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
    edge_mask = np.zeros(IMG.shape)
    edge_mask[int(IMG.shape[0]/5.):int(4.*IMG.shape[0]/5.),
              int(IMG.shape[1]/5.):int(4.*IMG.shape[1]/5.)] = 1

    # Run photutils source mask to remove pixels with sources
    # such as stars and galaxies, including a boarder
    # around each source.
    mask = make_source_mask(IMG,
                            nsigma = 3,
                            npixels = int(1./pixscale),
                            dilate_size = 40,
                            filter_fwhm = 1./pixscale,
                            filter_size = int(3./pixscale),
                            sigclip_iters = 5)
    mask = np.logical_or(mask, edge_mask)

    # Return statistics from background sky
    return {'background': np.median(IMG[np.logical_not(mask)]),
            'background noise': iqr(IMG[np.logical_not(mask)],rng = [16,84])/2}
