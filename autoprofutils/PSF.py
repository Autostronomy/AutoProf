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
from autoprofutils.SharedFunctions import _iso_extract
from copy import deepcopy

def _2DGaussFit(x, dat, xx, yy, noise, fwhm_guess):

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
    """
    Identify 20 bright stars and simultaneously fit a 2D gaussian to all of the stars.
    The standard deivation of the fitted gaussian is then turned into a FWHM.
    
    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """
    fwhm_guess = max(1. / pixscale, 1)
    edge_mask = np.zeros(IMG.shape, dtype = bool)
    edge_mask[int(IMG.shape[0]/5.):int(4.*IMG.shape[0]/5.),
              int(IMG.shape[1]/5.):int(4.*IMG.shape[1]/5.)] = True

    # photutils wrapper for IRAF star finder
    count = 0
    while count < 5:
        iraffind = IRAFStarFinder(fwhm = fwhm_guess, threshold = 6.*results['background noise'], brightest = 50)
        irafsources = iraffind.find_stars(IMG - results['background'], edge_mask)
        fwhm_guess = np.median(irafsources['fwhm'])
        if np.median(irafsources['sharpness']) >= 0.95:
            break
        count += 1
        
    if 'doplot' in kwargs and kwargs['doplot']:    
        plt.imshow(np.clip(IMG - results['background'], a_min = 0, a_max = None), origin = 'lower',
                   cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        for i in range(len(irafsources['fwhm'])):
            plt.gca().add_patch(Ellipse((irafsources['xcentroid'][i],irafsources['ycentroid'][i]), 16/pixscale, 16/pixscale,
                                        0, fill = False, linewidth = 0.5, color = 'y'))
        plt.savefig('%sPSF_Stars_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), dpi = 600)
        plt.clf()

    res = minimize(_2DGaussFit, x0 = [(fwhm_guess/2.355)**2], args = (IMG - results['background'], irafsources['xcentroid'], irafsources['ycentroid'], results['background noise'], fwhm_guess))
    logging.info('%s: found psf: %f' % (name,np.sqrt(res.x[0])*2.355))
    return {'psf fwhm': np.sqrt(res.x[0])*2.355}

def _GaussFit(x, sr, sf):
    return np.mean((sf - x[1]*norm.pdf(sr, loc = 0, scale = x[0]))**2)

def PSF_GaussFit(IMG, pixscale, name, results, **kwargs):
    """
    Identify 20 bright stars and simultaneously fit a 2D gaussian to all of the stars.
    The standard deivation of the fitted gaussian is then turned into a FWHM.
    
    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """
    fwhm_guess = max(1. / pixscale, 1)
    edge_mask = np.zeros(IMG.shape, dtype = bool)
    edge_mask[int(IMG.shape[0]/5.):int(4.*IMG.shape[0]/5.),
              int(IMG.shape[1]/5.):int(4.*IMG.shape[1]/5.)] = True

    dat = IMG - results['background']
    # photutils wrapper for IRAF star finder
    count = 0
    sources = 0
    while count < 5 and sources < 20:
        iraffind = IRAFStarFinder(fwhm = fwhm_guess, threshold = 6.*results['background noise'], brightest = 50)
        irafsources = iraffind.find_stars(dat, edge_mask)
        fwhm_guess = np.median(irafsources['fwhm'])
        if np.median(irafsources['sharpness']) >= 0.95:
            break
        count += 1
        sources = len(irafsources['fwhm'])
        
    if 'doplot' in kwargs and kwargs['doplot']:    
        plt.imshow(np.clip(IMG - results['background'], a_min = 0, a_max = None), origin = 'lower',
                   cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        for i in range(len(irafsources['fwhm'])):
            plt.gca().add_patch(Ellipse((irafsources['xcentroid'][i],irafsources['ycentroid'][i]), 16/pixscale, 16/pixscale,
                                        0, fill = False, linewidth = 0.5, color = 'y'))
        plt.savefig('%sPSF_Stars_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), dpi = 600)
        plt.clf()

    # Get set of stars
    psf_estimates = []
    minbadcount = 0
    while len(psf_estimates) <= 5 and minbadcount < 5:
        for i in range(sources):
            sf = []
            sr = []
            count_badcoefs = 0
            medflux = np.inf
            while medflux > 2*results['background noise']:
                sr.append(sr[-1]*1.1 if len(sr) > 0 else 1)
                isovals = _iso_extract(dat,sr[-1],0.,0.,{'x': irafsources['xcentroid'][i], 'y': irafsources['ycentroid'][i]})
                medflux = np.median(isovals)
                sf.append(np.median(isovals))
                coefs = fft(np.clip(isovals, a_max = np.quantile(isovals,0.85), a_min = None))
                if np.abs(coefs[1]) > np.sqrt(coefs[0]) or np.abs(coefs[2]) > np.sqrt(coefs[0]):
                    count_badcoefs += 1
            if count_badcoefs <= minbadcount:
                res = minimize(_GaussFit, x0 = [fwhm_guess/2.355, sf[0]/norm.pdf(0,loc = 0,scale = fwhm_guess/2.355)], args = (np.array(sr), np.array(sf)), method = 'Nelder-Mead')
                # Paper plot
                # plt.scatter(sr,sf,color = 'k', label = 'star data')
                # plt.plot(np.linspace(0,sr[-1]*1.1, 100), res.x[1]*norm.pdf(np.linspace(0,sr[-1]*1.1, 100), loc = 0, scale = res.x[0]), color = 'r', label = 'PSF model')
                # plt.ylabel('flux')
                # plt.xlabel('radius [pix]')
                # plt.legend()
                # plt.savefig('%sPSF_Fit_%s_%i.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name,i))
                # plt.clf()
                psf_estimates.append(res.x[0]*2.355)
            if len(psf_estimates) > 30:
                break
        minbadcount += 1
        
    logging.info('%s: found psf: %f' % (name,np.median(psf_estimates)))
    return {'psf fwhm': np.median(psf_estimates) if len(psf_estimates) >= 5 else fwhm_guess}


def Calculate_PSF(IMG, pixscale, name, results, **kwargs):
    """
    Idenitfy the location of stars in the image and calculate
    their average PSF.

    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """
    
    # Guess for PSF based on user provided seeing value
    fwhm_guess = max(1. / pixscale, 1)
    edge_mask = np.zeros(IMG.shape, dtype = bool)
    edge_mask[int(IMG.shape[0]/5.):int(4.*IMG.shape[0]/5.),
              int(IMG.shape[1]/5.):int(4.*IMG.shape[1]/5.)] = True

    # photutils wrapper for IRAF star finder
    count = 0
    while count < 5:
        iraffind = IRAFStarFinder(fwhm = fwhm_guess, threshold = 6.*results['background noise'], roundlo = 0.01)
        irafsources = iraffind.find_stars(IMG - results['background'], edge_mask)
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
        plt.imshow(np.clip(IMG - results['background'], a_min = 0, a_max = None), origin = 'lower',
                   cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        for i in range(len(irafsources['fwhm'])):
            plt.gca().add_patch(Ellipse((irafsources['xcentroid'][i],irafsources['ycentroid'][i]), 16/pixscale, 16/pixscale,
                                    0, fill = False, linewidth = 0.5, color = 'y'))
        plt.savefig('%sPSF_Stars_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), dpi = 600)
        plt.clf()                
    
    logging.info('%s: found psf: %f' % (name,np.median(irafsources['fwhm'])))
    
    # Return PSF statistics
    return {'psf fwhm': fwhm_guess}
    
def Given_PSF(IMG, pixscale, name, results, **kwargs):
    """
    Uses the kwarg "given_psf" to return a user inputted psf.

    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """

    try:
        if type(kwargs['given_psf']) == float:
            return {'psf fwhm': kwargs['given_psf']}
        else:
            return {'psf fwhm': kwargs['given_psf'][name]}
    except:
        return Calculate_PSF(IMG, pixscale, name, results, **kwargs)
