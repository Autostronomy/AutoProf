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

def Overflow_Mask(IMG, results, options):
    """
    Identify parts of the image where the CCD has overflowed and maxed
    out the sensor. These are characterized by large areas with high
    and identical pixel values
    """
    
    if 'autodetectoverflow' in options and options['autodetectoverflow']:
        # Set the overflowval to the most common pixel value, since overflow
        # pixels all have the same value with no noise.
        overflowval = mode(IMG, axis = None, nan_policy = 'omit')
        # If less than 10 pixels have the mode value, assume no pixels have
        # overflowed and the value is just random.
        if np.sum(IMG == overflowval) < 100:
            return IMG, {'mask': np.zeros(IMG.shape,dtype = bool)}
    if (not 'overflowval' in options) or options['overflowval'] is None:
        logging.info('%s: not masking overflow %s' % options['name'])
        return IMG, {'mask': np.zeros(IMG.shape)}

    Mask = np.logical_and(IMG > (options['overflowval'] - 1e-3), IMG < (options['overflowval'] + 1e-3)).astype(bool)

    # eliminate places where no data is recorded
    Mask[IMG == 0] = True
    logging.info('%s: masking %i overflow pixels' % (options['name'], np.sum(Mask)))
    return IMG, {'mask': Mask}

def Mask_Segmentation_Map(IMG, results, options):

    if options['mask_file'] is None:
        mask = np.zeros(IMG.shape) 
    else:
        mask = Read_Image(options['mask_file'], options)
        if 'preprocess' in options and 'preprocess_all' in options and options['preprocess_all']: 
            mask = options['preprocess'](mask)
            
    if 'center' in results:
        if mask[int(results['center']['y']),int(results['center']['x'])] > 1.1:
            mask[mask == mask[int(results['center']['y']),int(results['center']['x'])]] = 0
    elif 'given_center' in options:
        if mask[int(options['given_center']['y']),int(options['given_center']['x'])] > 1.1:
            mask[mask == mask[int(options['given_center']['y']),int(options['given_center']['x'])]] = 0
    elif mask[int(IMG.shape[0]/2),int(IMG.shape[1]/2)] > 1.1:
        mask[mask == mask[int(IMG.shape[0]/2),int(IMG.shape[1]/2)]] = 0

    return IMG, {'mask': mask.astype(bool)}

def Star_Mask_IRAF(IMG, results, options):
    """
    Idenitfy the location of stars in the image and create a mask around
    each star of pixels to be avoided in further processing.
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
    if 'mask_file' in options and not options['mask_file'] is None:
        mask  = np.logical_or(mask, Read_Image(mask_file, options))
        
    # Run separate code to find overflow pixels from very bright stars
    overflow_mask = Overflow_Mask(IMG, results, options)[1]['mask']
    
    # Plot star mask for diagnostic purposes
    if 'doplot' in options and options['doplot']:
        plt.imshow(np.clip(IMG[max(0,int(use_center['y']-smaj*1.2)): min(IMG.shape[0],int(use_center['y']+smaj*1.2)),
                               max(0,int(use_center['x']-smaj*1.2)): min(IMG.shape[1],int(use_center['x']+smaj*1.2))],
                           a_min = 0, a_max = None), origin = 'lower',
                   cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        dat = np.logical_or(mask, overflow_mask).astype(float)[max(0,int(use_center['y']-smaj*1.2)): min(IMG.shape[0],int(use_center['y']+smaj*1.2)),
                                                               max(0,int(use_center['x']-smaj*1.2)): min(IMG.shape[1],int(use_center['x']+smaj*1.2))]
        dat[dat == 0] = np.nan
        plt.imshow(dat, origin = 'lower', cmap = 'Reds_r', alpha = 0.7)
        plt.savefig('%sMask_%s.jpg' % (options['plotpath'] if 'plotpath' in options else '', options['name']))
        plt.close()
    
    return IMG, {'mask': np.logical_or(mask, overflow_mask)}
