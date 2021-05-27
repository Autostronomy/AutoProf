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
from autoprofutils.SharedFunctions import _iso_extract, StarFind, AddLogo, LSBImage, autocolours
from copy import deepcopy

def PSF_IRAF(IMG, results, options):
    """
    Apply the photutils IRAF wrapper to the image to extract a PSF fwhm
    """
    if 'ap_set_psf' in options:
        logging.info('%s: PSF set by user: %.4e' % (options['ap_name'], options['ap_set_psf']))
        return IMG, {'psf fwhm': options['ap_set_psf']}
    elif 'ap_guess_psf' in options:
        logging.info('%s: PSF initialized by user: %.4e' % (options['ap_name'], options['ap_guess_psf']))
        fwhm_guess = options['ap_guess_psf']
    else:
        fwhm_guess = max(1., 1./options['ap_pixscale'])

    edge_mask = np.zeros(IMG.shape, dtype = bool)
    edge_mask[int(IMG.shape[0]/5.):int(4.*IMG.shape[0]/5.),
              int(IMG.shape[1]/5.):int(4.*IMG.shape[1]/5.)] = True
    
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
        return IMG, {'psf fwhm': fwhm_guess}
    if len(irafsources) < 5:
        return IMG, {'psf fwhm': fwhm_guess}
    
    if 'ap_doplot' in options and options['ap_doplot']:    
        plt.imshow(np.clip(IMG - results['background'], a_min = 0, a_max = None), origin = 'lower',
                   cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        for i in range(len(irafsources['fwhm'])):
            plt.gca().add_patch(Ellipse((irafsources['xcentroid'][i],irafsources['ycentroid'][i]), 16/options['ap_pixscale'], 16/options['ap_pixscale'],
                                        0, fill = False, linewidth = 0.5, color = 'y'))
        plt.savefig('%sPSF_Stars_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = 600)
        plt.close()

    psf = np.median(irafsources['fwhm'])
    return IMG, {'psf fwhm': psf, 'auxfile psf': 'psf fwhm: %.3f pix' % psf}

def PSF_StarFind(IMG, results, options):

    if 'ap_set_psf' in options:
        logging.info('%s: PSF set by user: %.4e' % (options['ap_name'], options['ap_set_psf']))
        return IMG, {'psf fwhm': options['ap_set_psf']}
    elif 'ap_guess_psf' in options:
        logging.info('%s: PSF initialized by user: %.4e' % (options['ap_name'], options['ap_guess_psf']))
        fwhm_guess = options['ap_guess_psf']
    else:
        fwhm_guess = max(1., 1./options['ap_pixscale'])

    edge_mask = np.zeros(IMG.shape, dtype = bool)
    edge_mask[int(IMG.shape[0]/5.):int(4.*IMG.shape[0]/5.),
              int(IMG.shape[1]/5.):int(4.*IMG.shape[1]/5.)] = True
    stars = StarFind(IMG - results['background'], fwhm_guess, results['background noise'],
                     edge_mask,  maxstars = 50)
    if len(stars['fwhm']) <= 10:
        return IMG, {'psf fwhm': fwhm_guess}
    def_clip = 0.1
    while np.sum(stars['deformity'] < def_clip) < max(10,len(stars['fwhm'])/2):
        def_clip += 0.1
    psf = np.median(stars['fwhm'][stars['deformity'] < def_clip])
    if 'ap_doplot' in options and options['ap_doplot']:
        LSBImage(IMG - results['background'], results['background noise'])
        for i in range(len(stars['fwhm'])):
            plt.gca().add_patch(Ellipse((stars['x'][i],stars['y'][i]), 20*psf, 20*psf,
                                        0, fill = False, linewidth = 1.5, color = autocolours['red1'] if stars['deformity'][i] >= def_clip else autocolours['blue1']))
        plt.tight_layout()
        if not ('ap_nologo' in options and options['ap_nologo']):
            AddLogo(plt.gcf())
        plt.savefig('%sPSF_Stars_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
        plt.close()

    logging.info('%s: found psf: %f with deformity clip of: %f' % (options['ap_name'],psf, def_clip))
    return IMG, {'psf fwhm': psf, 'auxfile psf': 'psf fwhm: %.3f pix' % psf}

