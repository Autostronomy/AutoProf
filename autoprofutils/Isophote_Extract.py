import numpy as np
from photutils.isophote import EllipseSample, Ellipse, EllipseGeometry, Isophote, IsophoteList
from scipy.optimize import minimize
from scipy.stats import iqr
from scipy.fftpack import fft, ifft
from time import time
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from copy import copy
import logging
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _x_to_pa, _x_to_eps, _inv_x_to_eps, _inv_x_to_pa, SBprof_to_COG_errorprop

def Simple_Isophote_Extract(IMG, mask, background_level, center, R, E, PA, name = ''):
    """
    Extracts the specified isophotes using photutils ellipsesample function.
    Applies mask and backgorund level to image.    
    """

    # Create image array with background and mask applied
    if np.any(mask):
        logging.info('%s: is masked' % (name))
        dat = np.ma.masked_array(IMG - background_level, mask)
    else:
        logging.info('%s: is not masked' % (name))
        dat = IMG - background_level

    # Store isophotes
    iso_list = []
    
    for i in range(len(R)):
        # Container for ellipse geometry
        geo = EllipseGeometry(sma = R[i],
                              x0 = center['x'], y0 = center['y'],
                              eps = E[i], pa = PA[i])
        # Extract the isophote information
        ES = EllipseSample(dat, sma = R[i], geometry = geo)
        ES.extract()
        iso_list.append(Isophote(ES, niter = 30, valid = True, stop_code = 0))
        
    return IsophoteList(iso_list)

def Generate_Profile(IMG, pixscale, mask, background, background_noise, center, R, E, Ee, PA, PAe, global_fit, name, **kwargs):    

    isolist =  Simple_Isophote_Extract(IMG, mask, background,
                                       center, R, E, PA, name)
    isolistglob =  Simple_Isophote_Extract(IMG, mask, background,
                                           center, R, np.ones(len(R))*global_fit['ellip'],
                                           np.ones(len(R))*global_fit['pa'], name)
    
    # Compute surface brightness in mag arcsec^-2 from flux
    zeropoint = kwargs['zeropoint'] if 'zeropoint' in kwargs else 22.5
    clippedflux = list(np.clip(isolist.sample[i].values[2], a_min = None, a_max = np.quantile(isolist.sample[i].values[2],0.9)) for i in range(len(isolist.sample)))
    sb = np.array(list((-2.5*np.log10(np.median(isolist.sample[i].values[2]))\
                        + zeropoint + 2.5*np.log10(pixscale**2)) \
                       for i in range(len(isolist.sma))))
    sb[np.logical_not(np.isfinite(sb))] = 99.999

    sbE = list((iqr(clippedflux[i], #  - ifft(fft(clippedflux[i])[:int(len(clippedflux[i])/2)], n = len(clippedflux[i]))
                    rng = (31.7310507863/2,
                           100 - 31.7310507863/2)) / (2*np.sqrt(len(isolist.sample[i].values[2])))) \
               for i in range(len(isolist.sma)))

    sbE = np.array(list((np.abs(2.5*sbE[i]/(np.median(isolist.sample[i].values[2]) * np.log(10)))) \
                        for i in range(len(isolist.sma))))

    sb[np.logical_not(np.isfinite(sbE))] = 99.999
    sbE[np.logical_or(np.logical_not(np.isfinite(sbE)), sb > 90)] = 99.999

    # Compute Curve of Growth from SB profile
    cog, cogE = SBprof_to_COG_errorprop(isolist.sma * pixscale, sb, sbE, 1. - isolist.eps,
                                        isolist.ellip_err, N = 100, method = 0, symmetric_error = True)

    # Compute global profile values
    sbglob = np.array(list((-2.5*np.log10(np.median(isolistglob.sample[i].values[2]))\
                            + zeropoint + 2.5*np.log10(pixscale**2)) \
                           for i in range(len(isolistglob.sma))))
    sbglob[np.logical_not(np.isfinite(sbglob))] = 99.999

    sbglobE = list((iqr(isolistglob.sample[i].values[2],
                        rng = (31.7310507863/2,
                               100 - 31.7310507863/2)) / (2*np.sqrt(len(isolistglob.sample[i].values[2])))) \
                   for i in range(len(isolistglob.sma)))

    sbglobE = np.array(list((np.abs(2.5*sbglobE[i]/(np.median(isolistglob.sample[i].values[2]) * np.log(10)))) \
                            for i in range(len(isolistglob.sma))))

    sbglob[np.logical_not(np.isfinite(sbglobE))] = 99.999
    sbglobE[np.logical_or(np.logical_not(np.isfinite(sbglobE)), sbglob > 90)] = 99.999

    # Compute Curve of Growth from SB profile
    cogglob, cogglobE = SBprof_to_COG_errorprop(isolistglob.sma * pixscale, sbglob, sbglobE, 1. - isolistglob.eps,
                                                isolistglob.ellip_err, N = 100, method = 0, symmetric_error = True)
    
    # For each radius evaluation, write the profile parameters
    params = ['R', 'SB', 'SB_e', 'totmag', 'totmag_e', 'ellip', 'ellip_e', 'pa', 'pa_e', 'totmag_direct', 'totmag_direct_e', 'SB_fix', 'SB_fix_e', 'totmag_fix', 'totmag_fix_e'] # , 'x0', 'y0'
        
    SBprof_data = dict((h,[]) for h in params)
    SBprof_units = {'R': 'arcsec', 'SB': 'mag arcsec^-2', 'SB_e': 'mag arcsec^-2', 'totmag': 'mag', 'totmag_e': 'mag',
                    'ellip': 'unitless', 'ellip_e': 'unitless', 'pa': 'deg', 'pa_e': 'deg', 'totmag_direct': 'mag', 'totmag_direct_e': 'mag',
                    'SB_fix': 'mag arcsec^-2', 'SB_fix_e': 'mag arcsec^-2', 'totmag_fix': 'mag', 'totmag_fix_e': 'mag'}
    SBprof_format = {'R': '%.4f', 'SB': '%.4f', 'SB_e': '%.4f', 'totmag': '%.4f', 'totmag_e': '%.4f',
                    'ellip': '%.3f', 'ellip_e': '%.3f', 'pa': '%.2f', 'pa_e': '%.2f', 'totmag_direct': '%.4f', 'totmag_direct_e': '%.4f',
                     'SB_fix': '%.4f', 'SB_fix_e': '%.4f', 'totmag_fix': '%.4f', 'totmag_fix_e': '%.4f'}
    for i in range(len(isolist.sma)):
        tflux_e_err = isolist.rms[i] / (np.sqrt(isolist.npix_e[i]))
        SBprof_data['R'].append(isolist.sma[i] * pixscale)
        SBprof_data['SB'].append(sb[i])
        SBprof_data['SB_e'].append(sbE[i])
        SBprof_data['totmag'].append(cog[i])
        SBprof_data['totmag_e'].append(cogE[i])
        SBprof_data['ellip'].append(isolist.eps[i])
        SBprof_data['ellip_e'].append(Ee[i])
        SBprof_data['pa'].append(isolist.pa[i]*180/np.pi)
        SBprof_data['pa_e'].append(PAe[i]*180/np.pi)
        SBprof_data['totmag_direct'].append(zeropoint - 2.5*np.log10(isolist.tflux_e[i]))
        SBprof_data['totmag_direct_e'].append(np.abs(2.5*tflux_e_err/(isolist.tflux_e[i] * np.log(10))))
        SBprof_data['SB_fix'].append(sbglob[i])
        SBprof_data['SB_fix_e'].append(sbglobE[i])
        SBprof_data['totmag_fix'].append(cogglob[i])
        SBprof_data['totmag_fix_e'].append(cogglobE[i])

    if 'doplot' in kwargs and kwargs['doplot']:
        CHOOSE = np.logical_and(np.array(SBprof_data['SB']) < 99, np.array(SBprof_data['SB_e']) < 1)
        plt.errorbar(np.array(SBprof_data['R'])[CHOOSE], np.array(SBprof_data['SB'])[CHOOSE], yerr = np.array(SBprof_data['SB_e'])[CHOOSE],
                     elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = 'purple', label = 'SB')
        plt.errorbar(np.array(SBprof_data['R'])[CHOOSE], np.array(SBprof_data['totmag'])[CHOOSE], yerr = np.array(SBprof_data['totmag_e'])[CHOOSE],
                     elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = 'orange', label = 'COG')
        plt.xlabel('Radius [arcsec]')
        plt.ylabel('Brightness [mag, mag/arcsec^2]')
        plt.axhline(-2.5*np.log10(background_noise/2) + zeropoint + 2.5*np.log10(pixscale**2), color = 'purple', linewidth = 0.5, linestyle = '--', label = 'Sky noise')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.savefig('%sphotometry_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
        plt.clf()                
        
    return {'prof header': params, 'prof units': SBprof_units, 'prof data': SBprof_data, 'prof format': SBprof_format}

def Isophote_Extract_Forced(IMG, pixscale, name, results, **kwargs):
    """
    Run isophote data extraction given exact specification for pa and ellip profiles.
    
    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """
    if 'fit ellip_err' in results and (not results['fit ellip_err'] is None) and 'fit pa_err' in results and (not results['fit pa_err'] is None):
        Ee = results['fit ellip_err']
        PAe = results['fit pa_err']
    else:
        Ee = np.zeros(len(results['fit R']))
        PAe = np.zeros(len(results['fit R']))
    return Generate_Profile(IMG, pixscale, np.logical_or(results['overflow mask'],results['mask']),
                            results['background'], results['background noise'], results['center'], results['fit R'],
                            results['fit ellip'], Ee, results['fit pa'], PAe, {'ellip': results['init ellip'], 'pa': results['init pa']}, name, **kwargs)

def Isophote_Extract(IMG, pixscale, name, results, **kwargs):
    """
    Extract isophotes given output profile from Isophotes_Simultaneous_Fit which
    parametrizes pa and ellipticity via functions which map all the reals to the
    appropriate parameter range. This function also extrapolates the profile to
    large and small radii (simply by taking the parameters at the edge, no
    functional extrapolation). By default uses a linear radius growth, however
    for large images, it uses a geometric radius growth of 10% per isophote.

    IMG: 2d ndarray with flux values for the image
    pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
    name: string name of galaxy in image, used for log files to make searching easier
    results: dictionary contianing results from past steps in the pipeline
    kwargs: user specified arguments
    """

    use_center = results['center']
        
    # Radius values to evaluate isophotes
    R = [kwargs['sampleinitR'] if 'sampleinitR' in kwargs else min(1.,results['psf fwhm']/2)]
    while ((R[-1] < kwargs['sampleendR'] if 'sampleendR' in kwargs else True) and R[-1] < 2*results['fit R'][-1] and R[-1] < min(IMG.shape)/2) or (kwargs['extractfull'] if 'extractfull' in kwargs else False):
        if 'samplestyle' in kwargs and kwargs['samplestyle'] == 'geometric-linear':
            if len(R) > 1 and abs(R[-1] - R[-2]) >= (kwargs['samplelinearscale'] if 'samplelinearscale' in kwargs else 3*results['psf fwhm']):
                R.append(R[-1] + (kwargs['samplelinearscale'] if 'samplelinearscale' in kwargs else results['psf fwhm']))
            else:
                R.append(R[-1]*(1. + (kwargs['samplegeometricscale'] if 'samplegeometricscale' in kwargs else 0.1)))
        elif 'samplestyle' in kwargs and kwargs['samplestyle'] == 'linear':
            R.append(R[-1] + (kwargs['samplelinearscale'] if 'samplelinearscale' in kwargs else 0.5*results['psf fwhm']))
        else:
            R.append(R[-1]*(1. + (kwargs['samplegeometricscale'] if 'samplegeometricscale' in kwargs else 0.1)))
    R = np.array(R)
    logging.info('%s: R complete in range [%.1f,%.1f]' % (name,R[0],R[-1]))
    
    # Interpolate profile values, when extrapolating just take last point
    E = _x_to_eps(np.interp(R, results['fit R'], _inv_x_to_eps(results['fit ellip'])))
    E[R < results['fit R'][0]] = results['fit ellip'][0] #R[R < results['fit R'][0]] * results['fit ellip'][0] / results['fit R'][0]
    E[R < results['psf fwhm']] = R[R < results['psf fwhm']] * results['fit ellip'][0] / results['psf fwhm']
    E[R > results['fit R'][-1]] = results['fit ellip'][-1]
    tmp_pa_s = np.interp(R, results['fit R'], np.sin(2*results['fit pa']))
    tmp_pa_c = np.interp(R, results['fit R'], np.cos(2*results['fit pa']))
    PA = _x_to_pa(((np.arctan(tmp_pa_s/tmp_pa_c) + (np.pi*(tmp_pa_c < 0))) % (2*np.pi))/2)#_x_to_pa(np.interp(R, results['fit R'], results['fit pa']))
    PA[R < results['fit R'][0]] = _x_to_pa(results['fit pa'][0])
    PA[R > results['fit R'][-1]] = _x_to_pa(results['fit pa'][-1])

    # Get errors for pa and ellip
    if 'fit ellip_err' in results and (not results['fit ellip_err'] is None) and 'fit pa_err' in results and (not results['fit pa_err'] is None):
        Ee = np.clip(np.interp(R, results['fit R'], results['fit ellip_err']), a_min = 1e-3, a_max = None)
        Ee[R < results['fit R'][0]] = results['fit ellip_err'][0]
        Ee[R > results['fit R'][-1]] = results['fit ellip_err'][-1]
        PAe = np.clip(np.interp(R, results['fit R'], results['fit pa_err']), a_min = 1e-3, a_max = None)
        PAe[R < results['fit R'][0]] = results['fit pa_err'][0]
        PAe[R > results['fit R'][-1]] = results['fit pa_err'][-1]
    else:
        Ee = np.zeros(len(results['fit R']))
        PAe = np.zeros(len(results['fit R']))
    
    # Stop from masking anything at the center
    results['mask'][int(use_center['x'] - 10*results['psf fwhm']):int(use_center['x'] + 10*results['psf fwhm']),
                            int(use_center['y'] - 10*results['psf fwhm']):int(use_center['y'] + 10*results['psf fwhm'])] = False
    compund_Mask = np.logical_or(results['overflow mask'],results['mask'])

    # Extract SB profile
    return Generate_Profile(IMG,pixscale,compund_Mask,
                            results['background'],
                            results['background noise'],
                            use_center, R, E, Ee, PA, PAe,
                            {'ellip': results['init ellip'], 'pa': results['init pa']},
                            name, **kwargs)
