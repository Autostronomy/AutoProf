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
from autoprofutils.SharedFunctions import _x_to_pa, _x_to_eps, _inv_x_to_eps, _inv_x_to_pa, SBprof_to_COG_errorprop, _iso_extract, _iso_within, _iso_between

def _Generate_Profile(IMG, pixscale, name, results, R, E, Ee, PA, PAe, **kwargs):
    
    # Create image array with background and mask applied
    try:
        mask = results['mask']
    except:
        mask = None
    dat = IMG - results['background']
    zeropoint = kwargs['zeropoint'] if 'zeropoint' in kwargs else 22.5

    sb = []
    sbE = []
    cogdirect = []
    sbfix = []
    sbfixE = []

    for i in range(len(R)):
        if R[i] < (kwargs['isoband_start'] if 'isoband_start' in kwargs else 100):
            isovals = _iso_extract(dat, R[i], E[i], PA[i], results['center'], mask = mask)
            isovalsfix = _iso_extract(dat, R[i], results['init ellip'], results['init pa'], results['center'], mask = mask)
        else:
            isobandwidth = R[i]*(kwargs['isoband_width'] if 'isoband_width' in kwargs else 0.025)
            isovals = _iso_between(dat, R[i] - isobandwidth, R[i] + isobandwidth, E[i], PA[i], results['center'], mask = mask)
            isovalsfix = _iso_between(dat, R[i] - isobandwidth, R[i] + isobandwidth, results['init ellip'], results['init pa'], results['center'], mask = mask)
        isotot = _iso_within(dat, R[i], E[i], PA[i], results['center'], mask = mask)
        medflux = np.median(isovals)
        medfluxfix = np.median(isovalsfix)
        sb.append((-2.5*np.log10(medflux) + zeropoint + 5*np.log10(pixscale)) if medflux > 0 else 99.999)
        sbE.append((2.5*iqr(isovals, rng = (31.731/2, 100 - 31.731/2)) / (2*np.sqrt(len(isovals))*medflux*np.log(10))) if medflux > 0 else 99.999)
        sbfix.append((-2.5*np.log10(medfluxfix) + zeropoint + 5*np.log10(pixscale)) if medfluxfix > 0 else 99.999)
        sbfixE.append((2.5*iqr(isovalsfix, rng = (31.731/2, 100 - 31.731/2)) / (2*np.sqrt(len(isovalsfix))*np.median(isovalsfix)*np.log(10))) if medfluxfix > 0 else 99.999)
        cogdirect.append(-2.5*np.log10(isotot) + zeropoint)
        
    # Compute Curve of Growth from SB profile
    cog, cogE = SBprof_to_COG_errorprop(R * pixscale, np.array(sb), np.array(sbE), 1. - E,
                                        Ee, N = 100, method = 0, symmetric_error = True)
    cogE[cog > 99] = 99.999
    cogfix, cogfixE = SBprof_to_COG_errorprop(R * pixscale, np.array(sbfix), np.array(sbfixE), 1. - E,
                                              Ee, N = 100, method = 0, symmetric_error = True)
    cogfixE[cogfix > 99] = 99.999
    
    # For each radius evaluation, write the profile parameters
    params = ['R', 'SB', 'SB_e', 'totmag', 'totmag_e', 'ellip', 'ellip_e', 'pa', 'pa_e', 'totmag_direct', 'SB_fix', 'SB_fix_e', 'totmag_fix', 'totmag_fix_e']
        
    SBprof_data = dict((h,None) for h in params)
    SBprof_units = {'R': 'arcsec', 'SB': 'mag*arcsec^-2', 'SB_e': 'mag*arcsec^-2', 'totmag': 'mag', 'totmag_e': 'mag',
                    'ellip': 'unitless', 'ellip_e': 'unitless', 'pa': 'deg', 'pa_e': 'deg', 'totmag_direct': 'mag',
                    'SB_fix': 'mag*arcsec^-2', 'SB_fix_e': 'mag*arcsec^-2', 'totmag_fix': 'mag', 'totmag_fix_e': 'mag'}
    SBprof_format = {'R': '%.4f', 'SB': '%.4f', 'SB_e': '%.4f', 'totmag': '%.4f', 'totmag_e': '%.4f',
                    'ellip': '%.3f', 'ellip_e': '%.3f', 'pa': '%.2f', 'pa_e': '%.2f', 'totmag_direct': '%.4f',
                     'SB_fix': '%.4f', 'SB_fix_e': '%.4f', 'totmag_fix': '%.4f', 'totmag_fix_e': '%.4f'}
    
    SBprof_data['R'] = list(R * pixscale)
    SBprof_data['SB'] = list(sb)
    SBprof_data['SB_e'] = list(sbE)
    SBprof_data['totmag'] = list(cog)
    SBprof_data['totmag_e'] = list(cogE)
    SBprof_data['ellip'] = list(E)
    SBprof_data['ellip_e'] = list(Ee)
    SBprof_data['pa'] = list(PA*180/np.pi)
    SBprof_data['pa_e'] = list(PAe*180/np.pi)
    SBprof_data['totmag_direct'] = list(cogdirect)
    SBprof_data['SB_fix'] = list(sbfix)
    SBprof_data['SB_fix_e'] = list(sbfixE)
    SBprof_data['totmag_fix'] = list(cogfix)
    SBprof_data['totmag_fix_e'] = list(cogfixE)

    if 'doplot' in kwargs and kwargs['doplot']:
        CHOOSE = np.logical_and(np.array(SBprof_data['SB']) < 99, np.array(SBprof_data['SB_e']) < 1)
        errscale = 1.
        if np.all(np.array(SBprof_data['SB_e'])[CHOOSE] < 0.8):
            errscale = 1/np.max(np.array(SBprof_data['SB_e'])[CHOOSE])
        plt.errorbar(np.array(SBprof_data['R'])[CHOOSE], np.array(SBprof_data['SB'])[CHOOSE], yerr = errscale*np.array(SBprof_data['SB_e'])[CHOOSE],
                     elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = 'r', label = 'Surface Brightness (err$\\cdot$%.1f)' % errscale)
        plt.errorbar(np.array(SBprof_data['R'])[np.logical_and(CHOOSE,np.arange(len(CHOOSE)) % 4 == 0)],
                     np.array(SBprof_data['SB'])[np.logical_and(CHOOSE,np.arange(len(CHOOSE)) % 4 == 0)],
                     yerr = np.array(SBprof_data['SB_e'])[np.logical_and(CHOOSE,np.arange(len(CHOOSE)) % 4 == 0)],
                     elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = 'limegreen')
        plt.errorbar(np.array(SBprof_data['R'])[CHOOSE], np.array(SBprof_data['totmag'])[CHOOSE], yerr = np.array(SBprof_data['totmag_e'])[CHOOSE],
                     elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = 'orange', label = 'Curve of Growth')
        plt.xlabel('Radius [arcsec]')
        plt.ylabel('Brightness [mag, mag/arcsec^2]')
        bkgrdnoise = -2.5*np.log10(results['background noise']) + zeropoint + 2.5*np.log10(pixscale**2)
        plt.axhline(bkgrdnoise, color = 'purple', linewidth = 0.5, linestyle = '--', label = '1$\\sigma$ noise/pixel: %.1f mag arcsec$^{-2}$' % bkgrdnoise)
        plt.gca().invert_yaxis()
        plt.legend(fontsize = 10)
        plt.savefig('%sphotometry_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
        plt.close()                

        useR = np.array(SBprof_data['R'])[CHOOSE]/pixscale
        useE = np.array(SBprof_data['ellip'])[CHOOSE]
        usePA = np.array(SBprof_data['pa'])[CHOOSE]
        ranges = [[max(0,int(results['center']['x']-useR[-1]*1.2)), min(dat.shape[1],int(results['center']['x']+useR[-1]*1.2))],
                  [max(0,int(results['center']['y']-useR[-1]*1.2)), min(dat.shape[0],int(results['center']['y']+useR[-1]*1.2))]]
        plt.imshow(np.clip(dat[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]],
                           a_min = 0,a_max = None), origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch())) 
        for i in range(len(useR)):
            plt.gca().add_patch(Ellipse((results['center']['x'] - ranges[0][0],results['center']['y'] - ranges[1][0]), 2*useR[i], 2*useR[i]*(1. - useE[i]),
                                        usePA[i], fill = False, linewidth = 0.3, color = 'limegreen' if (i % 4 == 0) else 'r', linestyle = '-' if useR[i] < results['fit R'][-1] else '--'))
        plt.savefig('%sphotometry_ellipse_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), dpi = 400)
        plt.close()

        
    return {'prof header': params, 'prof units': SBprof_units, 'prof data': SBprof_data, 'prof format': SBprof_format}
    

def Isophote_Extract_Forced(IMG, pixscale, name, results, **kwargs):
    """
    Run isophote data extraction given exact specification for pa and ellip profiles.
    """
    if 'fit ellip_err' in results and (not results['fit ellip_err'] is None) and 'fit pa_err' in results and (not results['fit pa_err'] is None):
        Ee = np.array(results['fit ellip_err'])
        PAe = np.array(results['fit pa_err'])
    else:
        Ee = np.zeros(len(results['fit R']))
        PAe = np.zeros(len(results['fit R']))
        
    return _Generate_Profile(IMG, pixscale, name, results, np.array(results['fit R']), np.array(results['fit ellip']), Ee, np.array(results['fit pa']), PAe, **kwargs)
    
    
def Isophote_Extract(IMG, pixscale, name, results, **kwargs):
    """
    Extract isophotes given output profile from Isophotes_Simultaneous_Fit which
    parametrizes pa and ellipticity via functions which map all the reals to the
    appropriate parameter range. This function also extrapolates the profile to
    large and small radii (simply by taking the parameters at the edge, no
    functional extrapolation). By default uses a linear radius growth, however
    for large images, it uses a geometric radius growth of 10% per isophote.
    """

    use_center = results['center']
        
    # Radius values to evaluate isophotes
    R = [kwargs['sampleinitR'] if 'sampleinitR' in kwargs else min(1.,results['psf fwhm']/2)]
    while (((R[-1] < kwargs['sampleendR'] if 'sampleendR' in kwargs else True) and R[-1] < 3*results['fit R'][-1]) or (kwargs['extractfull'] if 'extractfull' in kwargs else False)) and R[-1] < max(IMG.shape)/np.sqrt(2):
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
    
    return _Generate_Profile(IMG, pixscale, name, results, R, E, Ee, PA, PAe, **kwargs)
