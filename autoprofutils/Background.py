from photutils import make_source_mask
from photutils.isophote import EllipseSample, Ellipse, EllipseGeometry, Isophote, IsophoteList
from astropy.stats import sigma_clipped_stats
from scipy.stats import iqr
from scipy.optimize import minimize
from time import time
import logging
import numpy as np

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
    values = IMG[edge_mask].ravel()
    start = np.median(values)
    scale = iqr(values,rng = [30,70])

    res = minimize(lambda x: -np.sum(np.exp(-((values - x)/scale)**2)), x0 = [start], method = 'Nelder-Mead')

    clip_above = 3*np.sqrt(np.mean(values - res.x[0])**2)
    for i in range(10):
        clip_above = 3*np.sqrt(np.mean(values[values < clip_above] - res.x[0])**2)

    return {'background': res.x[0], 'noise': iqr(values[values < clip_above], rng = [16,84])/2}

def Background_Global(IMG, pixscale, name, results, **kwargs):
    """
    Compute a global background value for an image. Performed by
    identifying pixels which are beyond 3 sigma above the average
    signal and masking them, also further masking a boarder
    of 20 pixels around the initial masked pixels. Returns a
    dictionary of parameters describing the background level.

    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
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
    return {'mean': np.mean(IMG[np.logical_not(mask)]),
            'background': np.median(IMG[np.logical_not(mask)]),
            'std': np.std(IMG[np.logical_not(mask)]),
            'noise': iqr(IMG[np.logical_not(mask)],rng = [16,84])/2}

def Background_ByPatches(IMG, pixscale, name, results, **kwargs):
    """
    Compute a global background value for an image. Done by
    evaluating statistics on various patches of sky near the
    boarder of an image. Patches include corner squares and
    edge bars.

    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """

    # Make the slicing commands to grab patches
    patches = [[[None,int(IMG.shape[0]/5.)],[int(IMG.shape[1]/5.),int(4*IMG.shape[1]/5.)]],
               [[int(4*IMG.shape[0]/5.),None],[int(IMG.shape[1]/5.),int(4*IMG.shape[1]/5.)]],
               [[int(IMG.shape[0]/5.),int(4*IMG.shape[0]/5.)],[None,int(IMG.shape[1]/5.)]],
               [[int(IMG.shape[0]/5.),int(4*IMG.shape[0]/5.)],[int(4*IMG.shape[1]/5.),None]],
               [[None,int(IMG.shape[0]/4.)],[None,int(IMG.shape[1]/4.)]],
               [[int(3*IMG.shape[0]/4.),None],[None,int(IMG.shape[1]/4.)]],
               [[None,int(IMG.shape[0]/4.)],[int(3*IMG.shape[1]/4.),None]],
               [[int(3*IMG.shape[0]/4.), None],[int(3*IMG.shape[1]/4.),None]]]
    clip_at = 4 * iqr(IMG)

    # Loop through the patches and compute statistics on each
    stats = {'mean':[], 'median': [], 'std': [], 'iqr': []}
    for p in patches:
        vals = IMG[p[0][0]:p[0][1],
                   p[1][0]:p[1][1]]
        stats['mean'].append(np.mean(vals[vals < clip_at]))
        stats['median'].append(np.median(vals[vals < clip_at]))
        stats['std'].append(np.std(vals[vals < clip_at]))
        stats['iqr'].append(iqr(vals[vals < clip_at],rng=[16,84])/2)

    # Compute statistics on the patches, instead of on image
    mean = np.median(stats['mean'])
    median = np.median(stats['median'])
    std = np.median(stats['std'])
    img_iqr = np.median(stats['iqr'])

    return {'mean': mean,
            'background': median,
            'std': std,
            'noise': img_iqr}
    

def Background_ByIsophote(IMG, pixscale, name, results, **kwargs):
    """
    Compute circular isophotes at large radii of the image. The
    flux space surface brightness is used as a background measurement

    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """
    isophote_SBs = []

    R = [1./pixscale]
    
    while R[-1] < IMG.shape[0]/2:
        R.append(R[-1] * 1.5)
        geo = EllipseGeometry(sma = R[-1],
                              x0 = int(IMG.shape[0]/2), y0 = int(IMG.shape[1]/2),
                              eps = 0.,
                              pa = 0.)
        ES = EllipseSample(IMG,
                           sma = R[-1],
                           geometry = geo)
        ES.extract()
        isophote_SBs.append(np.median(ES.values[2]))
    
    return {'mean': np.mean(isophote_SBs[min(np.argmin(isophote_SBs), len(isophote_SBs)-2):]),
            'background': np.min(isophote_SBs),
            'std': np.std(isophote_SBs[min(np.argmin(isophote_SBs), len(isophote_SBs)-2):]),
            'noise': iqr(isophote_SBs[min(np.argmin(isophote_SBs), len(isophote_SBs)-2):],rng=[16,84])/2}

def Background_All(IMG, pixscale, name, results, **kwargs):
    """
    Run all the background calculation algorithms and compare the results

    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """

    byisophote = Background_ByIsophote(IMG, pixscale, name, results, **kwargs)
    bypatches = Background_ByPatches(IMG, pixscale, name, results, **kwargs)
    byglobal = Background_Global(IMG, pixscale, name, results, **kwargs)
    start = time()
    bymode = Background_Mode(IMG, pixscale, name, results, **kwargs)
    
    logging.info('BACKGROUNDTEST %s|%f|%f|%f|%f, %.2f' % (name, byisophote['background'], bypatches['background'], byglobal['background'], bymode['background'], time() - start))

    return bymode
