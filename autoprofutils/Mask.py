from photutils import DAOStarFinder, IRAFStarFinder
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy.stats import mode
import logging
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import Read_Image

def Overflow_Mask(IMG, pixscale, name, results, **kwargs):
    """
    Identify parts of the image where the CCD has overflowed and maxed
    out the sensor. These are characterized by large areas with high
    and identical pixel values
    """
    
    if 'autodetectoverflow' in kwargs and kwargs['autodetectoverflow']:
        # Set the overflowval to the most common pixel value, since overflow
        # pixels all have the same value with no noise.
        overflowval = mode(IMG, axis = None, nan_policy = 'omit')
        # If less than 10 pixels have the mode value, assume no pixels have
        # overflowed and the value is just random.
        if np.sum(IMG == overflowval) < 100:
            return np.zeros(IMG.shape,dtype = bool)
    if (not 'overflowval' in kwargs) or kwargs['overflowval'] is None:
        logging.info('%s: not masking overflow %s' % name)
        return np.zeros(IMG.shape)

    Mask = np.logical_and(IMG > (kwargs['overflowval'] - 1e-3), IMG < (kwargs['overflowval'] + 1e-3)).astype(bool)

    # eliminate places where no data is recorded
    Mask[IMG == 0] = True
    logging.info('%s: masking %i overflow pixels' % (name, np.sum(Mask)))
    return Mask

def Star_Mask_Given(IMG, pixscale, name, results, **kwargs):

    mask = np.zeros(IMG.shape) if kwargs['mask_file'] is None else Read_Image(kwargs['mask_file'], **kwargs)
    # Run separate code to find overflow pixels from very bright stars
    overflow_mask = Overflow_Mask(IMG, pixscale, name, results, **kwargs)

    return {'mask': mask, 'overflow mask': overflow_mask}

def Star_Mask_IRAF(IMG, pixscale, name, results, **kwargs):
    """
    Idenitfy the location of stars in the image and create a mask around
    each star of pixels to be avoided in further processing.

    fixme note: this can be optimized with slicing by only ever working
    on a patch of sky around the star. So long as the paths is definately
    larger than the circle to be masked.

    IMG: numpy 2d array of pixel values
    pixscale: conversion factor from pixels to arcsec (arcsec pixel^-1)
    background: output from a image background signal calculation (dict)
    psf: point spread function statistics
    overflowval: optional pixel flux value for overflowed pixels

    returns: collection of mask information
    """

    fwhm = results['psf fwhm']
    use_center = results['center']

    # Find scale of bounding box for galaxy. Stars will only be found within this box
    smaj = results['fit R'][-1]
    xbox = int(1.5*smaj)
    ybox = int(1.5*smaj)
    xbounds = [max(0,int(use_center['x'] - xbox)),min(int(use_center['x'] + xbox),IMG.shape[1])]
    ybounds = [max(0,int(use_center['y'] - ybox)),min(int(use_center['y'] + ybox),IMG.shape[0])]    
    
    # Run photutils wrapper for IRAF star finder
    iraffind = IRAFStarFinder(fwhm = 2*fwhm, threshold = 10.*results['background noise'], brightest = 50)
    irafsources = iraffind((IMG - results['background'])[ybounds[0]:ybounds[1],
                                                                   xbounds[0]:xbounds[1]])
    mask = np.zeros(IMG.shape, dtype = bool)
    # Mask star pixels and area around proportionate to their total flux
    XX,YY = np.meshgrid(range(IMG.shape[0]),range(IMG.shape[1]), indexing = 'ij')
    if irafsources:
        for x,y,f in zip(irafsources['xcentroid'], irafsources['ycentroid'], irafsources['flux']):
            if np.sqrt((x - (xbounds[1] - xbounds[0])/2)**2 + (y - (ybounds[1] - ybounds[0])/2)**2) < 10*results['psf fwhm']:
                continue
            # compute distance of every pixel to the identified star
            R = np.sqrt((XX-(x + xbounds[0]))**2 + (YY-(y + ybounds[0]))**2)
            # Compute the flux of the star
            #f = np.sum(IMG[R < 10*fwhm])
            # Compute radius to reach background noise level, assuming gaussian
            Rstar = (fwhm/2.355)*np.sqrt(2*np.log(f/(np.sqrt(2*np.pi*fwhm/2.355)*results['background noise']))) # fixme double check
            mask[R < Rstar] = True 

    # Include user defined mask if any
    if 'mask_file' in kwargs and not kwargs['mask_file'] is None:
        mask  = np.logical_or(mask, Read_Image(mask_file, **kwargs))
        
    # Run separate code to find overflow pixels from very bright stars
    overflow_mask = Overflow_Mask(IMG, pixscale, name, results, **kwargs)
    
    # Plot star mask for diagnostic purposes
    if 'doplot' in kwargs and kwargs['doplot']:
        plt.imshow(np.clip(IMG[max(0,int(use_center['y']-smaj*1.2)): min(IMG.shape[0],int(use_center['y']+smaj*1.2)),
                               max(0,int(use_center['x']-smaj*1.2)): min(IMG.shape[1],int(use_center['x']+smaj*1.2))],
                           a_min = 0, a_max = None), origin = 'lower',
                   cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        dat = np.logical_or(mask, overflow_mask).astype(float)[max(0,int(use_center['y']-smaj*1.2)): min(IMG.shape[0],int(use_center['y']+smaj*1.2)),
                                                               max(0,int(use_center['x']-smaj*1.2)): min(IMG.shape[1],int(use_center['x']+smaj*1.2))]
        dat[dat == 0] = np.nan
        plt.imshow(dat, origin = 'lower', cmap = 'Reds_r', alpha = 0.7)
        plt.savefig('%sMask_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
        plt.close()
    
    return {'mask':mask,
            'overflow mask': overflow_mask}

def NoMask(IMG, pixscale, name, results, **kwargs):
    """
    Dont mask stars. Still mask overflowed values if given.
    """
    overflow_mask = Overflow_Mask(IMG, pixscale, name, results, **kwargs)

    # Include user defined mask if any
    if 'mask_file' in kwargs and not kwargs['mask_file'] is None:
        mask = np.array(Read_Image(mask_file, **kwargs),dtype=bool)
    else:
        mask = np.zeros(IMG.shape,dtype = bool)
        
    return {'mask': mask,
            'overflow mask': overflow_mask}
    
