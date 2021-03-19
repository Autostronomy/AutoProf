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
from scipy.fftpack import fft, ifft
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _iso_extract, StarFind
from copy import deepcopy

def PSF_IRAF(IMG, pixscale, name, results, **kwargs):
    """
    Apply the photutils IRAF wrapper to the image to extract a PSF fwhm
    """
    if 'psf_set' in kwargs:
        return {'psf fwhm': kwargs['psf_set']}
    elif 'psf_guess' in kwargs:
        fwhm_guess = kwargs['psf_guess']
    else:
        fwhm_guess = max(1., 1./pixscale)

    edge_mask = np.zeros(IMG.shape, dtype = bool)
    edge_mask[int(IMG.shape[0]/4.):int(3.*IMG.shape[0]/4.),
              int(IMG.shape[1]/4.):int(3.*IMG.shape[1]/4.)] = True
    
    dat = IMG - results['background']
    # photutils wrapper for IRAF star finder
    count = 0
    sources = 0
    psf_iter = deepcopy(psf_guess)
    try:
        while count < 5 and sources < 20:
            iraffind = IRAFStarFinder(fwhm = psf_iter, threshold = 6.*results['background noise'], brightest = 50)
            irafsources = iraffind.find_stars(dat, edge_mask)
            psf_iter = np.median(irafsources['fwhm'])
            if np.median(irafsources['sharpness']) >= 0.95:
                break
            count += 1
            sources = len(irafsources['fwhm'])
    except:
        return {'psf fwhm': fwhm_guess}
    if len(irafsources) < 5:
        return {'psf fwhm': fwhm_guess}
    
    if 'doplot' in kwargs and kwargs['doplot']:    
        plt.imshow(np.clip(IMG - results['background'], a_min = 0, a_max = None), origin = 'lower',
                   cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        for i in range(len(irafsources['fwhm'])):
            plt.gca().add_patch(Ellipse((irafsources['xcentroid'][i],irafsources['ycentroid'][i]), 16/pixscale, 16/pixscale,
                                        0, fill = False, linewidth = 0.5, color = 'y'))
        plt.savefig('%sPSF_Stars_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), dpi = 600)
        plt.close()
        
    return {'psf fwhm': np.median(irafsources['fwhm'])}


def PSF_StarFind(IMG, pixscale, name, results, **kwargs):

    if 'psf_set' in kwargs:
        return {'psf fwhm': kwargs['psf_set']}
    elif 'psf_guess' in kwargs:
        fwhm_guess = kwargs['psf_guess']
    else:
        fwhm_guess = max(1., 1./pixscale)

    edge_mask = np.zeros(IMG.shape, dtype = bool)
    edge_mask[int(IMG.shape[0]/4.):int(3.*IMG.shape[0]/4.),
              int(IMG.shape[1]/4.):int(3.*IMG.shape[1]/4.)] = True
    stars = StarFind(IMG - results['background'], fwhm_guess, results['background noise'],
                     edge_mask, # peakmax = (kwargs['overflowval']-results['background'])*0.95 if 'overflowval' in kwargs else None,
                     maxstars = 50)
    if len(stars['fwhm']) <= 10:
        return {'psf fwhm': fwhm_guess}
    def_clip = 0.1
    while np.sum(stars['deformity'] < def_clip) < max(10,2*len(stars['fwhm'])/3):
        def_clip += 0.1
    if 'doplot' in kwargs and kwargs['doplot']:
        plt.imshow(np.clip(IMG - results['background'], a_min = 0, a_max = None), origin = 'lower',
                   cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        for i in range(len(stars['fwhm'])):
            plt.gca().add_patch(Ellipse((stars['x'][i],stars['y'][i]), 16/pixscale, 16/pixscale,
                                        0, fill = False, linewidth = 0.5, color = 'r' if stars['deformity'][i] >= def_clip else 'y'))
        plt.savefig('%sPSF_Stars_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), dpi = 600)
        plt.close()

    logging.info('%s: found psf: %f with deformity clip of: %f' % (name,np.median(stars['fwhm'][stars['deformity'] < def_clip]), def_clip))
    return {'psf fwhm': np.median(stars['fwhm'][stars['deformity'] < def_clip])}
