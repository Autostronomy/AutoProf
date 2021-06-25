from photutils import make_source_mask
from photutils.isophote import EllipseSample, Ellipse, EllipseGeometry, Isophote, IsophoteList
from astropy.stats import sigma_clipped_stats
from scipy.stats import iqr
from scipy.optimize import minimize
from scipy.fftpack import fft2, ifft2
from scipy.integrate import trapz
from time import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import AddLogo, Smooth_Mode
from autoprofutils.Diagnostic_Plots import Plot_Background

def Background_Mode(IMG, results, options):
    """
    Compute background by finding the peak in a smoothed histogram of flux values.
    This should correspond to the peak of the noise floor.
    """
    # Mask main body of image so only outer 1/5th is used
    # for background calculation.
    if 'mask' in results and not results['mask'] is None and np.any(results['mask']):
        mask = np.logical_not(results['mask'])
        logging.info('%s: Background using mask. Masking %i pixels' % (options['ap_name'], np.sum(results['mask'])))
    else:
        mask = np.ones(IMG.shape, dtype = bool)
        mask[int(IMG.shape[0]/5.):int(4.*IMG.shape[0]/5.),
             int(IMG.shape[1]/5.):int(4.*IMG.shape[1]/5.)] = False
    values = IMG[mask].flatten()
    if len(values) < 1e5:
        values = IMG.flatten()
    if 'ap_background_speedup' in options and int(options['ap_background_speedup']) > 1:
        values = values[::int(options['ap_background_speedup'])]
    values = values[np.isfinite(values)]

    if 'ap_set_background' in options:
        bkgrnd = options['ap_set_background']
        logging.info('%s: Background set by user: %.4e' % (options['ap_name'], bkgrnd))
    else:
        # # Fit the peak of the smoothed histogram
        bkgrnd = Smooth_Mode(values)
        
    # Compute the 1sigma range using negative flux values, which should almost exclusively be sky noise
    if 'ap_set_background_noise' in options:
        noise = options['ap_set_background_noise']
        logging.info('%s: Background Noise set by user: %.4e' % (options['ap_name'], noise))
    else:
        noise = iqr(values[(values-bkgrnd) < 0], rng = [100 - 68.2689492137,100])
        if not np.isfinite(noise):
            noise = iqr(values,rng = [16,84])/2.
    uncertainty = noise / np.sqrt(np.sum((values-bkgrnd) < 0))
    if not np.isfinite(uncertainty):
        uncertainty = noise / np.sqrt(len(values))
    
    if 'ap_doplot' in options and options['ap_doplot']:
        Plot_Background(values, bkgrnd, noise, results, options)
        
    return IMG, {'background': bkgrnd,
                 'background noise': noise,
                 'background uncertainty': uncertainty,
                 'auxfile background': 'background: %.5e +- %.2e flux/pix, noise: %.5e flux/pix' % (bkgrnd, uncertainty, noise)}

def Background_DilatedSources(IMG, results, options):
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
    if not ('ap_set_background' in options and 'ap_set_background_noise' in options):
        source_mask = make_source_mask(IMG, nsigma = 3,
                                       npixels = int(1./options['ap_pixscale']),
                                       dilate_size = 40,
                                       filter_fwhm = 1./options['ap_pixscale'],
                                       filter_size = int(3./options['ap_pixscale']),
                                       sigclip_iters = 5)
        mask = np.logical_or(mask, source_mask)

    # Return statistics from background sky
    bkgrnd = options['ap_set_background'] if 'ap_set_background' in options else np.median(IMG[np.logical_not(mask)])
    noise = options['ap_set_background_noise'] if 'ap_set_background_noise' in options else iqr(IMG[np.logical_not(mask)],rng = [16,84])/2
    uncertainty = noise/np.sqrt(np.sum(np.logical_not(mask)))
    
    if 'ap_doplot' in options and options['ap_doplot']:
        Plot_Background(IMG[np.logical_not(mask)].ravel(), bkgrnd, noise, results, options)
    return IMG, {'background': bkgrnd,
                 'background noise': noise,
                 'background uncertainty': uncertainty,
                 'auxfile background': 'background: %.5e +- %.2e flux/pix, noise: %.5e flux/pix' % (bkgrnd, uncertainty, noise)}

def Background_Basic(IMG, results, options):
    mask = np.ones(IMG.shape, dtype = bool)
    mask[int(IMG.shape[0]/4.):int(3.*IMG.shape[0]/4.),
         int(IMG.shape[1]/4.):int(3.*IMG.shape[1]/4.)] = False

    bkgrnd = options['ap_set_background'] if 'ap_set_background' in options else np.mean(IMG[mask])
    noise = options['ap_set_background_noise'] if 'ap_set_background_noise' in options else np.std(IMG[mask])
    uncertainty = noise / np.sqrt(len(IMG[mask].ravel()))
    if 'ap_doplot' in options and options['ap_doplot']:
        Plot_Background(IMG[mask].ravel(), bkgrnd, noise, results, options)
    return IMG, {'background': bkgrnd,
                 'background noise': noise,
                 'background uncertainty': uncertainty,
                 'auxfile background': 'background: %.5e +- %.2e flux/pix, noise: %.5e flux/pix' % (bkgrnd, uncertainty, noise)}

def Background_Unsharp(IMG, results, options):

    coefs = fft2(IMG)

    unsharp = 3
    coefs[unsharp:-unsharp] = 0
    coefs[:,unsharp:-unsharp] = 0

    dumy, stats = Background_Mode(IMG, results, options)
    stats.update({'background': ifft2(coefs).real})
    return IMG, stats
            

    
