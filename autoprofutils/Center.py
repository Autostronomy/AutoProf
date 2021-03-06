import numpy as np
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _iso_extract
from photutils.centroids import centroid_2dg, centroid_com, centroid_1dg
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import logging
from copy import copy

def Center_Null(IMG, pixscale, name, results, **kwargs):
    """
    Simply returns the center of the image.

    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """

    current_center = {'x': IMG.shape[0]/2, 'y': IMG.shape[1]/2}
    if 'given_center' in kwargs:
        current_center = kwargs['given_center']
        
    return {'center': current_center}

def Center_Forced(IMG, pixscale, name, results, **kwargs):
    """
    Takes the center from an aux file, or given value.
    
    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """
    center = {'x': IMG.shape[0]/2, 'y': IMG.shape[1]/2}
    if 'given_center' in kwargs:
        return {'center': kwargs['given_center']}
    
    with open(kwargs['forcing_profile'][:-4] + 'aux', 'r') as f:
        for line in f.readlines():
            if line[:6] == 'center':
                x_loc = line.find('x:')
                y_loc = line.find('y:')
                try:
                    center = {'x': float(line[x_loc+3:line.find('pix')]),
                              'y': float(line[y_loc+3:line.rfind('pix')])}
                    break
                except:
                    pass
        else:
            logging.warning('%s: Forced center failed! Using image center.' % name)
    return {'center': center}


def Center_Given(IMG, pixscale, name, results, **kwargs):
    """
    Uses the kwarg "given_center" to return a user inputted center.

    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """

    try:
        return {'center': kwargs['given_center']}
    except:
        logging.warning('%s: No center given! using image center.' % name)
        return {'center': {'x': IMG.shape[0]/2.,
                           'y': IMG.shape[1]/2.}}
    
    
def Center_Centroid(IMG, pixscale, name, results, **kwargs):
    """
    Compute the pixel location of the galaxy center using a centroid method.
    Looking at 50 seeing lengths around the center of the image (images
    should already be mostly centered), finds the galaxy center by fitting
    a 2d Gaussian.
    
    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """
    
    current_center = {'x': IMG.shape[0]/2, 'y': IMG.shape[1]/2}
    if 'given_center' in kwargs:
        current_center = kwargs['given_center']
    if 'fit_center' in kwargs and not kwargs['fit_center']:
        return {'center': current_center}
    
    # Create mask to focus centering algorithm on the center of the image
    centralize_mask = np.ones(IMG.shape)
    centralize_mask[int(IMG.shape[0]/2 - 30 * results['psf fwhm'] / pixscale):int(IMG.shape[0]/2 + 30 * results['psf fwhm'] / pixscale),
                    int(IMG.shape[1]/2 - 30 * results['psf fwhm'] / pixscale):int(IMG.shape[1]/2 + 30 * results['psf fwhm'] / pixscale)] = 0
    decentralize_mask = np.ones(IMG.shape)
    decentralize_mask[int(IMG.shape[0]/2 - 5 * results['psf fwhm'] / pixscale):int(IMG.shape[0]/2 + 5 * results['psf fwhm'] / pixscale),
                      int(IMG.shape[1]/2 - 5 * results['psf fwhm'] / pixscale):int(IMG.shape[1]/2 + 5 * results['psf fwhm'] / pixscale)] = 0
    
    x, y = centroid_2dg(IMG - results['background'])

    # Plot center value for diagnostic purposes
    if 'doplot' in kwargs and kwargs['doplot']:    
        plt.imshow(np.clip(IMG - results['background'],a_min = 0, a_max = None),
                   origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        plt.plot([y],[x], marker = 'x', markersize = 10, color = 'y')
        plt.savefig('%scenter_vis_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
        plt.clf()
    logging.info('%s Center found: x %.1f, y %.1f' % (name, x, y))    
    return {'center': {'x': x,
                       'y': y}}

def Center_1DGaussian(IMG, pixscale, name, results, **kwargs):
    """
    Compute the pixel location of the galaxy center using a photutils method.
    Looking at 100 seeing lengths around the center of the image (images
    should already be mostly centered), finds the galaxy center by fitting
    several 1d Gaussians.
    
    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """
    
    current_center = {'x': IMG.shape[0]/2, 'y': IMG.shape[1]/2}
    if 'given_center' in kwargs:
        current_center = kwargs['given_center']
    if 'fit_center' in kwargs and not kwargs['fit_center']:
        return {'center': current_center}
    
    # mask image to focus algorithm on the center of the image
    centralize_mask = np.ones(IMG.shape, dtype = bool)
    centralize_mask[int(IMG.shape[0]/2 - 100 * results['psf fwhm'] / pixscale):int(IMG.shape[0]/2 + 100 * results['psf fwhm'] / pixscale),
                    int(IMG.shape[1]/2 - 100 * results['psf fwhm'] / pixscale):int(IMG.shape[1]/2 + 100 * results['psf fwhm'] / pixscale)] = False
    
    x, y = centroid_1dg(IMG - results['background'],
                        mask = centralize_mask) # np.logical_or(mask['mask'], centralize_mask)
    
    # Plot center value for diagnostic purposes
    if 'doplot' in kwargs and kwargs['doplot']:    
        plt.imshow(np.clip(IMG - results['background'],a_min = 0, a_max = None),
                   origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        plt.plot([y],[x], marker = 'x', markersize = 10, color = 'y')
        plt.savefig('%scenter_vis_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
        plt.clf()
    logging.info('%s Center found: x %.1f, y %.1f' % (name, x, y))    
    return {'center': {'x': x,
                       'y': y}}

def Center_OfMass(IMG, pixscale, name, results, **kwargs):
    """
    Compute the pixel location of the galaxy center using a light weighted
    center of mass. Looking at 100 seeing lengths around the center of the
    image (images should already be mostly centered), finds the average
    light weighted center of the image.
    
    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """
    
    current_center = {'x': IMG.shape[0]/2, 'y': IMG.shape[1]/2}
    if 'given_center' in kwargs:
        current_center = kwargs['given_center']
    if 'fit_center' in kwargs and not kwargs['fit_center']:
        return {'center': current_center}
    
    # mask image to focus algorithm on the center of the image
    centralize_mask = np.ones(IMG.shape)
    centralize_mask[int(IMG.shape[0]/2 - 50 * results['psf fwhm'] / pixscale):int(IMG.shape[0]/2 + 50 * results['psf fwhm'] / pixscale),
                    int(IMG.shape[1]/2 - 50 * results['psf fwhm'] / pixscale):int(IMG.shape[1]/2 + 50 * results['psf fwhm'] / pixscale)] = 0
    
    x, y = centroid_com(IMG - results['background'],
                        mask = centralize_mask) # np.logical_or(mask['mask'], centralize_mask)
    
    # Plot center value for diagnostic purposes
    if 'doplot' in kwargs and kwargs['doplot']:    
        plt.imshow(np.clip(IMG - results['background'],a_min = 0, a_max = None),
                   origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        plt.plot([y],[x], marker = 'x', markersize = 10, color = 'y')
        plt.savefig('%scenter_vis_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
        plt.clf()
    logging.info('%s Center found: x %.1f, y %.1f' % (name, x, y))    
    return {'center': {'x': x,
                       'y': y}}

def Center_Bright(IMG, pixscale, name, results, **kwargs):
    """
    simply takes the brightest pixel within a region around the center of the image.
    
    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """
    current_center = {'x': IMG.shape[0]/2, 'y': IMG.shape[1]/2}
    if 'given_center' in kwargs:
        current_center = kwargs['given_center']
    if 'fit_center' in kwargs and not kwargs['fit_center']:
        return {'center': current_center}
    
    subdat = IMG[int(IMG.shape[0]/2 - 20*results['psf fwhm']):int(IMG.shape[0]/2 + 20*results['psf fwhm']),
                 int(IMG.shape[1]/2 - 20*results['psf fwhm']):int(IMG.shape[1]/2 + 20*results['psf fwhm'])]

    locmax = np.unravel_index(np.argmax(subdat), subdat.shape)

    return {'center': {'x': locmax[0] + IMG.shape[0]/2 - 20*results['psf fwhm'],
                       'y': locmax[1] + IMG.shape[1]/2 - 20*results['psf fwhm']}}

def Center_HillClimb(IMG, pixscale, name, results, **kwargs):
    """
    Using 10 circular isophotes out to 10 times the PSF length, the first FFT coefficient
    phases are averaged to find the direction of increasing flux. Flux values are sampled
    along this direction and a quadratic fit gives the maximum. This is iteratively
    repeated until the step size becomes very small.
    
    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """
    current_center = {'x': IMG.shape[0]/2, 'y': IMG.shape[1]/2}
    if 'given_center' in kwargs:
        current_center = kwargs['given_center']
    if 'fit_center' in kwargs and not kwargs['fit_center']:
        return {'center': current_center}

    dat = IMG - results['background']

    sampleradii = np.linspace(1,10,10) * results['psf fwhm']

    track_centers = []
    small_update_count = 0
    total_count = 0
    while small_update_count <= 5 and total_count <= 100:
        total_count += 1
        phases = []
        isovals = []
        coefs = []
        for r in sampleradii:
            isovals.append(_iso_extract(dat,r,0.,0.,current_center, more = True))
            coefs.append(fft(np.clip(isovals[-1][0], a_max = np.quantile(isovals[-1][0],0.85), a_min = None)))
            phases.append((-np.angle(coefs[-1][1])) % (2*np.pi))
        complexphase = np.array(np.cos(phases) + np.sin(phases)*1j,dtype = np.complex_)
        direction = np.angle(np.mean(complexphase)) % (2*np.pi) 
        levels = []
        level_locs = []
        for i, r in enumerate(sampleradii):
            floc = np.argmin(np.abs(isovals[i][1] - direction))
            rloc = np.argmin(np.abs(isovals[i][1] - ((direction+np.pi) % (2*np.pi))))
            smooth = np.abs(ifft(coefs[i][:min(10,len(coefs[i]))],n = len(coefs[i])))
            if smooth[floc] > (3*results['background noise']):
                levels.append(smooth[floc])
                level_locs.append(r)
            if smooth[rloc] > (3*results['background noise']):
                levels.insert(0,smooth[rloc])
                level_locs.insert(0,-r)
        try:
            p = np.polyfit(level_locs, levels, deg = 2)
            if p[0] < 0 and len(levels) > 3:
                dist = np.clip(-p[1]/(2*p[0]), a_min = min(level_locs), a_max = max(level_locs))
            else:
                dist = level_locs[np.argmax(levels)]
        except:
            dist = 1.
        current_center['x'] += dist*np.cos(direction)
        current_center['y'] += dist*np.sin(direction)
        if abs(dist) < (0.25*results['psf fwhm']):
            small_update_count += 1
        else:
            small_update_count = 0
        track_centers.append([current_center['x'], current_center['y']])

    # refine center
    nochange_count = 0
    while nochange_count < 5:
        center_update = []
        center_loss = []
        for i in range(10):
            if i == 0:
                center_update.append(current_center)
            else:
                center_update.append({'x': current_center['x'] + np.random.normal(loc = 0, scale = results['psf fwhm']/4),
                                      'y': current_center['y'] + np.random.normal(loc = 0, scale = results['psf fwhm']/4)})
            center_loss.append(0)
            for rr in range(3):
                isovals = _iso_extract(dat,(rr+0.5)*results['psf fwhm'],0.05,
                                       0.,center_update[-1], more = True)
                coefs = fft(np.clip(isovals[0], a_max = np.quantile(isovals[0],0.85), a_min = None))
                center_loss[-1] += np.abs(coefs[1])/np.median(isovals[0])
        ci = np.argmin(center_loss)
        if ci == 0:
            nochange_count += 1
        else:
            nochange_count = 0
            current_center = copy(center_update[ci])
            track_centers.append([current_center['x'], current_center['y']])
    track_centers = np.array(track_centers)

    # paper plot
    # if 'doplot' in kwargs and kwargs['doplot']:    
    #     plt.imshow(np.clip(dat,a_min = 0, a_max = None), origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
    #     plt.plot([IMG.shape[0]/2],[IMG.shape[1]/2], marker = 'x', markersize = 2, color = 'y')
    #     plt.plot([current_center['x']],[current_center['y']], marker = 'x', markersize = 3, color = 'r')
    #     plt.savefig('%stest_center_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
    #     plt.clf()

    #     plt.plot(track_centers[:,0], track_centers[:,1], color = 'k')
    #     plt.scatter(track_centers[:,0], track_centers[:,1], c = range(len(track_centers)), cmap = 'Reds')
    #     plt.xlabel('x coordinate [pix]')
    #     plt.ylabel('y coordinate [pix]')
    #     plt.savefig('%sCenter_path_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
    #     plt.clf()
        
    return {'center': current_center}

def Center_Multi_Method(IMG, pixscale, name, results, **kwargs):
    """
    Compute the pixel location of the galaxy center using the other methods
    included here, determines the best one to use.
    
    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """

    # Centering algorithm order for multi-method
    cents = ['Centroid', 'COM', '1D']
    cent_fun = [Center_Centroid, Center_OfMass, Center_1DGaussian]
    x_colours = ['y', 'cyan', 'magenta', 'lime']

    # Meshgrid from relative pixel locations
    XX, YY = np.meshgrid(range(IMG.shape[0]), range(IMG.shape[1]), indexing = 'xy')
    CenterR = np.sqrt((XX - IMG.shape[0]/2)**2 + (YY - IMG.shape[1]/2)**2)
    
    cent_vals = []
    for i in range(len(cents)):
        # Run the centering algorithm
        try:
            cent = cent_fun[i](IMG, pixscale, name, results, **kwargs)
        except:
            cent = {'x': np.nan, 'y': np.nan}
        cent_vals.append(cent)

        # Check that the centering algorithm didn't crash
        if not (np.isfinite(cent['x']) and np.isfinite(cent['y']) and 0 <= cent['x'] < IMG.shape[0] and 0 <= cent['y'] < IMG.shape[1]):
            continue
        # Evaluate relative pixel locations to center
        R = np.sqrt((XX - cent['x'])**2 + (YY - cent['y'])**2)
        # Check how the algorithm center brightness compares to the image center brightness
        if np.sqrt((cent['x'] - IMG.shape[0]/2)**2 + (cent['y'] - IMG.shape[1]/2)**2) < 50*results['psf fwhm']:
            # Plot center for diagnostic purposes
            if 'doplot' in kwargs and kwargs['doplot']:    
                plt.imshow(np.clip(IMG,a_min = 0, a_max = None), origin = 'lower',
                           cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
                for vi,v in enumerate(cent_vals):
                    plt.plot([v['x']],[v['y']], marker = 'x', markersize = 10, color = x_colours[vi], label = cents[vi])
                plt.legend()
                plt.savefig('%scenter_vis_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
                plt.clf()
            return {'center': cent}
    
    logging.warning('%s Centering failed, using center of image' % name)
    return {'center': {'x':int(IMG.shape[0]/2.),
                       'y': int(IMG.shape[1]/2.)}}

