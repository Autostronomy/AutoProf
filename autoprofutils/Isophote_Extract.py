import numpy as np
from photutils.isophote import EllipseSample, EllipseGeometry, Isophote, IsophoteList
from photutils.isophote import Ellipse as Photutils_Ellipse
from scipy.optimize import minimize
from scipy.stats import iqr
from scipy.fftpack import fft, ifft
from time import time
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
from copy import copy
import logging
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _x_to_pa, _x_to_eps, _inv_x_to_eps, _inv_x_to_pa, SBprof_to_COG_errorprop, _iso_extract, _iso_between, LSBImage, AddLogo, _average, _scatter, flux_to_sb, flux_to_mag, PA_shift_convention, autocolours

def _Generate_Profile(IMG, results, R, E, Ee, PA, PAe, options):
    
    # Create image array with background and mask applied
    try:
        if np.any(results['mask']):
            mask = results['mask']
        else:
            mask = None
    except:
        mask = None
    dat = IMG - results['background']
    zeropoint = options['ap_zeropoint'] if 'ap_zeropoint' in options else 22.5

    sb = []
    sbE = []
    pixels = []
    cogdirect = []
    sbfix = []
    sbfixE = []
    Fmodes = []

    count_neg = 0
    medflux = np.inf
    end_prof = None
    compare_interp = []
    for i in range(len(R)):
        if 'ap_isoband_fixed' in options:
            isobandwidth = options['ap_isoband_width'] if 'ap_isoband_width' in options else 0.5
        else:
            isobandwidth = R[i]*(options['ap_isoband_width'] if 'ap_isoband_width' in options else 0.025)
        isisophoteband = False
        if medflux > (results['background noise']*(options['ap_isoband_start'] if 'ap_isoband_start' in options else 2)) or isobandwidth < 0.5:
            isovals = _iso_extract(dat, R[i], E[i], PA[i], results['center'], mask = mask, more = True,
                                   rad_interp = (options['ap_iso_interpolate_start'] if 'ap_iso_interpolate_start' in options else 5)*results['psf fwhm'],
                                   interp_method = (options['ap_iso_interpolate_method'] if 'ap_iso_interpolate_method' in options else 'lanczos'),
                                   sigmaclip = options['ap_isoclip'] if 'ap_isoclip' in options else False,
                                   sclip_iterations = options['ap_isoclip_iterations'] if 'ap_isoclip_iterations' in options else 10,
                                   sclip_nsigma = options['ap_isoclip_nsigma'] if 'ap_isoclip_nsigma' in options else 5)
            isovalsfix = _iso_extract(dat, R[i], results['init ellip'], results['init pa'], results['center'], mask = mask,
                                      rad_interp = (options['ap_iso_interpolate_start'] if 'ap_iso_interpolate_start' in options else 5)*results['psf fwhm'],
                                      sigmaclip = options['ap_isoclip'] if 'ap_isoclip' in options else False,
                                      sclip_iterations = options['ap_isoclip_iterations'] if 'ap_isoclip_iterations' in options else 10,
                                      sclip_nsigma = options['ap_isoclip_nsigma'] if 'ap_isoclip_nsigma' in options else 5)
        else:
            isisophoteband = True
            isovals = _iso_between(dat, R[i] - isobandwidth, R[i] + isobandwidth, E[i], PA[i], results['center'], mask = mask, more = True,
                                   sigmaclip = options['ap_isoclip'] if 'ap_isoclip' in options else False,
                                   sclip_iterations = options['ap_isoclip_iterations'] if 'ap_isoclip_iterations' in options else 10,
                                   sclip_nsigma = options['ap_isoclip_nsigma'] if 'ap_isoclip_nsigma' in options else 5)
            isovalsfix = _iso_between(dat, R[i] - isobandwidth, R[i] + isobandwidth, results['init ellip'], results['init pa'], results['center'], mask = mask,
                                      sigmaclip = options['ap_isoclip'] if 'ap_isoclip' in options else False,
                                      sclip_iterations = options['ap_isoclip_iterations'] if 'ap_isoclip_iterations' in options else 10,
                                      sclip_nsigma = options['ap_isoclip_nsigma'] if 'ap_isoclip_nsigma' in options else 5)
        isotot = np.sum(_iso_between(dat, 0, R[i], E[i], PA[i], results['center'], mask = mask))
        medflux = _average(isovals[0], options['ap_isoaverage_method'] if 'ap_isoaverage_method' in options else 'median')
        scatflux = _scatter(isovals[0], options['ap_isoaverage_method'] if 'ap_isoaverage_method' in options else 'median')
        medfluxfix = _average(isovalsfix, options['ap_isoaverage_method'] if 'ap_isoaverage_method' in options else 'median')
        scatfluxfix = _scatter(isovalsfix, options['ap_isoaverage_method'] if 'ap_isoaverage_method' in options else 'median')
        if 'ap_fouriermodes' in options and options['ap_fouriermodes'] > 0:
            if mask is None and (not 'ap_isoclip' in options or not options['ap_isoclip']) and not isisophoteband:
                coefs = fft(isovals[0])
            else:
                N = int(max(100, np.sqrt(len(isovals[0]))))
                theta = np.linspace(0,2*np.pi*(1.-1./N), N)
                coefs = fft(np.interp(theta, isovals[1], isovals[0], period = 2*np.pi))
            Fmodes.append({'a': [np.abs(coefs[0])/len(coefs)] + list(np.imag(coefs[1:int(max(options['ap_fouriermodes']+1,2))])/(np.abs(coefs[0]) + np.sqrt(len(coefs))*results['background noise'])),
                           'b': [np.abs(coefs[0])/len(coefs)] + list(np.real(coefs[1:int(max(options['ap_fouriermodes']+1,2))])/(np.abs(coefs[0]) + np.sqrt(len(coefs))*results['background noise']))})
                
        sb.append(flux_to_sb(medflux, options['ap_pixscale'], zeropoint) if medflux > 0 else 99.999)
        sbE.append((2.5*scatflux / (np.sqrt(len(isovals[0]))*medflux*np.log(10))) if medflux > 0 else 99.999)
        pixels.append(len(isovals[0]))
        sbfix.append(flux_to_sb(medfluxfix, options['ap_pixscale'], zeropoint) if medfluxfix > 0 else 99.999)
        sbfixE.append((2.5*scatfluxfix / (np.sqrt(len(isovalsfix))*medfluxfix*np.log(10))) if medfluxfix > 0 else 99.999)
        cogdirect.append(flux_to_mag(isotot, zeropoint) if isotot > 0 else 99.999)
        if medflux <= 0:
            count_neg += 1
        if 'ap_truncate_evaluation' in options and options['ap_truncate_evaluation'] and count_neg >= 2:
            end_prof = i+1
            break
        
    # Compute Curve of Growth from SB profile
    cog, cogE = SBprof_to_COG_errorprop(R[:end_prof]* options['ap_pixscale'], np.array(sb), np.array(sbE), 1. - E[:end_prof],
                                        Ee[:end_prof], N = 100, method = 0, symmetric_error = True)
    if cog is None:
        cog = 99.999*np.ones(len(R))
        cogE = 99.999*np.ones(len(R))
    else:
        cog[np.logical_not(np.isfinite(cog))] == 99.999
        cogE[cog > 99] = 99.999
    cogfix, cogfixE = SBprof_to_COG_errorprop(R[:end_prof] * options['ap_pixscale'], np.array(sbfix), np.array(sbfixE), 1. - E[:end_prof],
                                              Ee[:end_prof], N = 100, method = 0, symmetric_error = True)
    if cogfix is None:
        cogfix = 99.999*np.ones(len(R))
        cogfixE = 99.999*np.ones(len(R))
    else:
        cogfix[np.logical_not(np.isfinite(cogfix))] == 99.999
        cogfixE[cogfix > 99] = 99.999
    
    # For each radius evaluation, write the profile parameters
    params = ['R', 'SB', 'SB_e', 'totmag', 'totmag_e', 'ellip', 'ellip_e', 'pa', 'pa_e', 'pixels', 'totmag_direct', 'SB_fix', 'SB_fix_e', 'totmag_fix', 'totmag_fix_e']
        
    SBprof_data = dict((h,None) for h in params)
    SBprof_units = {'R': 'arcsec', 'SB': 'mag*arcsec^-2', 'SB_e': 'mag*arcsec^-2', 'totmag': 'mag', 'totmag_e': 'mag',
                    'ellip': 'unitless', 'ellip_e': 'unitless', 'pa': 'deg', 'pa_e': 'deg', 'pixels': 'count', 'totmag_direct': 'mag',
                    'SB_fix': 'mag*arcsec^-2', 'SB_fix_e': 'mag*arcsec^-2', 'totmag_fix': 'mag', 'totmag_fix_e': 'mag'}
    SBprof_format = {'R': '%.4f', 'SB': '%.4f', 'SB_e': '%.4f', 'totmag': '%.4f', 'totmag_e': '%.4f',
                     'ellip': '%.3f', 'ellip_e': '%.3f', 'pa': '%.2f', 'pa_e': '%.2f', 'pixels': '%i', 'totmag_direct': '%.4f',
                     'SB_fix': '%.4f', 'SB_fix_e': '%.4f', 'totmag_fix': '%.4f', 'totmag_fix_e': '%.4f'}
    
    SBprof_data['R'] = list(R[:end_prof] * options['ap_pixscale'])
    SBprof_data['SB'] = list(sb)
    SBprof_data['SB_e'] = list(sbE)
    SBprof_data['totmag'] = list(cog)
    SBprof_data['totmag_e'] = list(cogE)
    SBprof_data['ellip'] = list(E[:end_prof])
    SBprof_data['ellip_e'] = list(Ee[:end_prof])
    SBprof_data['pa'] = list(PA[:end_prof]*180/np.pi)
    SBprof_data['pa_e'] = list(PAe[:end_prof]*180/np.pi)
    SBprof_data['pixels'] = list(pixels)
    SBprof_data['totmag_direct'] = list(cogdirect)
    SBprof_data['SB_fix'] = list(sbfix)
    SBprof_data['SB_fix_e'] = list(sbfixE)
    SBprof_data['totmag_fix'] = list(cogfix)
    SBprof_data['totmag_fix_e'] = list(cogfixE)

    if 'ap_fouriermodes' in options:
        for i in range(int(options['ap_fouriermodes']+1)):
            aa, bb = 'a%i' % i, 'b%i' % i
            params += [aa, bb]
            SBprof_units.update({aa: 'flux' if i == 0 else 'a%i/F0' % i, bb: 'flux' if i == 0 else 'b%i/F0' % i})
            SBprof_format.update({aa: '%.4f', bb: '%.4f'})
            SBprof_data[aa] = list(F['a'][i] for F in Fmodes)
            SBprof_data[bb] = list(F['b'][i] for F in Fmodes)

    if 'ap_doplot' in options and options['ap_doplot']:
        CHOOSE = np.logical_and(np.array(SBprof_data['SB']) < 99, np.array(SBprof_data['SB_e']) < 1)
        if np.sum(CHOOSE) < 5:
            CHOOSE = np.ones(len(CHOOSE), dtype = bool)
        errscale = 1.
        if np.all(np.array(SBprof_data['SB_e'])[CHOOSE] < 0.5):
            errscale = 1/np.max(np.array(SBprof_data['SB_e'])[CHOOSE])
        lnlist = []
        lnlist.append(plt.errorbar(np.array(SBprof_data['R'])[CHOOSE], np.array(SBprof_data['SB'])[CHOOSE], yerr = errscale*np.array(SBprof_data['SB_e'])[CHOOSE],
                                   elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = autocolours['red1'], label = 'Surface Brightness (err$\\cdot$%.1f)' % errscale))
        plt.errorbar(np.array(SBprof_data['R'])[np.logical_and(CHOOSE,np.arange(len(CHOOSE)) % 4 == 0)],
                     np.array(SBprof_data['SB'])[np.logical_and(CHOOSE,np.arange(len(CHOOSE)) % 4 == 0)],
                     yerr = np.array(SBprof_data['SB_e'])[np.logical_and(CHOOSE,np.arange(len(CHOOSE)) % 4 == 0)],
                     elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = autocolours['blue1'])
        # plt.errorbar(np.array(SBprof_data['R'])[CHOOSE], np.array(SBprof_data['totmag'])[CHOOSE], yerr = np.array(SBprof_data['totmag_e'])[CHOOSE],
        #              elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = 'orange', label = 'Curve of Growth')
        plt.xlabel('Semi-Major-Axis [arcsec]', fontsize = 16)
        plt.ylabel('Surface Brightness [mag arcsec$^{-2}$]', fontsize = 16)
        plt.xlim([0,None])
        bkgrdnoise = -2.5*np.log10(results['background noise']) + zeropoint + 2.5*np.log10(options['ap_pixscale']**2)
        lnlist.append(plt.axhline(bkgrdnoise, color = 'purple', linewidth = 0.5, linestyle = '--', label = '1$\\sigma$ noise/pixel: %.1f mag arcsec$^{-2}$' % bkgrdnoise))
        plt.gca().invert_yaxis()
        plt.tick_params(labelsize = 14)
        # ax2 = plt.gca().twinx()
        # lnlist += ax2.plot(np.array(SBprof_data['R'])[CHOOSE], np.array(SBprof_data['pa'])[CHOOSE]/180, color = 'b', label = 'PA/180')
        # lnlist += ax2.plot(np.array(SBprof_data['R'])[CHOOSE], np.array(SBprof_data['ellip'])[CHOOSE], color = 'orange', linestyle = '--', label = 'ellipticity')
        labs = [l.get_label() for l in lnlist]
        plt.legend(lnlist, labs, fontsize = 11)
        # ax2.set_ylabel('Position Angle, Ellipticity', fontsize = 16)
        # ax2.tick_params(labelsize = 14)
        plt.tight_layout()
        if not ('ap_nologo' in options and options['ap_nologo']):
            AddLogo(plt.gcf())
        plt.savefig('%sphotometry_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
        plt.close()                

        useR = np.array(SBprof_data['R'])[CHOOSE]/options['ap_pixscale']
        useE = np.array(SBprof_data['ellip'])[CHOOSE]
        usePA = np.array(SBprof_data['pa'])[CHOOSE]
        ranges = [[max(0,int(results['center']['x']-useR[-1]*1.2)), min(dat.shape[1],int(results['center']['x']+useR[-1]*1.2))],
                  [max(0,int(results['center']['y']-useR[-1]*1.2)), min(dat.shape[0],int(results['center']['y']+useR[-1]*1.2))]]
        LSBImage(dat[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]], results['background noise'])
        fitlim = results['fit R'][-1] if 'fit R' in results else np.inf
        for i in range(len(useR)):
            plt.gca().add_patch(Ellipse((results['center']['x'] - ranges[0][0],results['center']['y'] - ranges[1][0]), 2*useR[i], 2*useR[i]*(1. - useE[i]),
                                        usePA[i], fill = False, linewidth = 1.2*((i+1)/len(useR))**2, color = autocolours['blue1'] if (i % 4 == 0) else autocolours['red1'], linestyle = '-' if useR[i] < fitlim else '--'))
        if not ('ap_nologo' in options and options['ap_nologo']):
            AddLogo(plt.gcf())
        plt.savefig('%sphotometry_ellipse_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
        plt.close()
        
    return {'prof header': params, 'prof units': SBprof_units, 'prof data': SBprof_data, 'prof format': SBprof_format}

def Isophote_Extract_Forced(IMG, results, options):
    """
    Run isophote data extraction given exact specification for pa and ellip profiles.
    """

    with open(options['ap_forcing_profile'], 'r') as f:
        raw = f.readlines()
        for i,l in enumerate(raw):
            if l[0] != '#':
                readfrom = i
                break
        header = list(h.strip() for h in raw[readfrom].split(','))
        force = dict((h,[]) for h in header)
        for l in raw[readfrom+2:]:
            for d, h in zip(l.split(','), header):
                force[h].append(float(d.strip()))

    force['pa'] = PA_shift_convention(np.array(force['pa']), deg = True) * np.pi/180
    
    if 'ellip_e' in force and 'pa_e' in force:
        Ee = np.array(force['ellip_e'])
        PAe = np.array(force['pa_e'])*np.pi/180
    else:
        Ee = np.zeros(len(force['R']))
        PAe = np.zeros(len(force['R']))

    return IMG, _Generate_Profile(IMG, results, np.array(force['R'])/options['ap_pixscale'],
                                  np.array(force['ellip']), Ee,
                                  (np.array(force['pa']) + (options['ap_forced_pa_shift'] if 'ap_forced_pa_shift' in options else 0.)) % np.pi, PAe, options)
    
    
def Isophote_Extract(IMG, results, options):
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
    R = [options['ap_sampleinitR'] if 'ap_sampleinitR' in options else min(1.,results['psf fwhm']/2)]
    while (((R[-1] < options['ap_sampleendR'] if 'ap_sampleendR' in options else True) and R[-1] < 3*results['fit R'][-1]) or (options['ap_extractfull'] if 'ap_extractfull' in options else False)) and R[-1] < max(IMG.shape)/np.sqrt(2):
        if 'ap_samplestyle' in options and options['ap_samplestyle'] == 'geometric-linear':
            if len(R) > 1 and abs(R[-1] - R[-2]) >= (options['ap_samplelinearscale'] if 'ap_samplelinearscale' in options else 3*results['psf fwhm']):
                R.append(R[-1] + (options['ap_samplelinearscale'] if 'ap_samplelinearscale' in options else results['psf fwhm']/2))
            else:
                R.append(R[-1]*(1. + (options['ap_samplegeometricscale'] if 'ap_samplegeometricscale' in options else 0.1)))
        elif 'ap_samplestyle' in options and options['ap_samplestyle'] == 'linear':
            R.append(R[-1] + (options['ap_samplelinearscale'] if 'ap_samplelinearscale' in options else 0.5*results['psf fwhm']))
        else:
            R.append(R[-1]*(1. + (options['ap_samplegeometricscale'] if 'ap_samplegeometricscale' in options else 0.1)))
    R = np.array(R)
    logging.info('%s: R complete in range [%.1f,%.1f]' % (options['ap_name'],R[0],R[-1]))
    
    # Interpolate profile values, when extrapolating just take last point
    E = _x_to_eps(np.interp(R, results['fit R'], _inv_x_to_eps(results['fit ellip'])))
    E[R < results['fit R'][0]] = results['fit ellip'][0]
    E[R > results['fit R'][-1]] = results['fit ellip'][-1]
    tmp_pa_s = np.interp(R, results['fit R'], np.sin(2*results['fit pa']))
    tmp_pa_c = np.interp(R, results['fit R'], np.cos(2*results['fit pa']))
    PA = _x_to_pa(((np.arctan(tmp_pa_s/tmp_pa_c) + (np.pi*(tmp_pa_c < 0))) % (2*np.pi))/2)
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
        Ee = np.zeros(len(R))
        PAe = np.zeros(len(R))
    
    return IMG, _Generate_Profile(IMG, results, R, E, Ee, PA, PAe, options)

def Isophote_Extract_Photutils(IMG, results, options):

    params = ['R', 'SB', 'SB_e', 'totmag', 'totmag_e', 'ellip', 'ellip_e', 'pa', 'pa_e', 'a3', 'a3_e', 'b3', 'b3_e', 'a4', 'a4_e', 'b4', 'b4_e']
        
    SBprof_data = dict((h,[]) for h in params)
    SBprof_units = {'R': 'arcsec', 'SB': 'mag*arcsec^-2', 'SB_e': 'mag*arcsec^-2', 'totmag': 'mag', 'totmag_e': 'mag',
                    'ellip': 'unitless', 'ellip_e': 'unitless', 'pa': 'deg', 'pa_e': 'deg', 'a3': 'unitless', 'a3_e': 'unitless',
                    'b3': 'unitless', 'b3_e': 'unitless', 'a4': 'unitless', 'a4_e': 'unitless', 'b4': 'unitless', 'b4_e': 'unitless'}
    SBprof_format = {'R': '%.4f', 'SB': '%.4f', 'SB_e': '%.4f', 'totmag': '%.4f', 'totmag_e': '%.4f',
                     'ellip': '%.3f', 'ellip_e': '%.3f', 'pa': '%.2f', 'pa_e': '%.2f', 'a3': '%.3f', 'a3_e': '%.3f',
                     'b3': '%.3f', 'b3_e': '%.3f', 'a4': '%.3f', 'a4_e': '%.3f', 'b4': '%.3f', 'b4_e': '%.3f'}
    zeropoint = options['ap_zeropoint'] if 'ap_zeropoint' in options else 22.5
    res = {}
    dat = IMG - results['background']
    if not 'fit R' in results and not 'fit photutils isolist' in results:
        logging.info('%s: photutils fitting and extracting image data' % options['ap_name'])
        geo = EllipseGeometry(x0 = results['center']['x'],
                              y0 = results['center']['y'],
                              sma = results['init R']/2,
                              eps = results['init ellip'],
                              pa = results['init pa'])
        ellipse = Photutils_Ellipse(dat, geometry = geo)

        isolist = ellipse.fit_image(fix_center = True, linear = False)
        res.update({'fit photutils isolist': isolist,
                    'auxfile fitlimit': 'fit limit semi-major axis: %.2f pix' % isolist.sma[-1]})
    elif not 'fit photutils isolist' in results:
        logging.info('%s: photutils extracting image data' % options['ap_name'])
        list_iso = []
        for i in range(len(results['fit R'])):
            if results['fit R'][i] <= 0:
                continue
            # Container for ellipse geometry
            geo = EllipseGeometry(sma = results['fit R'][i],
                                  x0 = results['center']['x'], y0 = results['center']['y'],
                                  eps = results['fit ellip'][i], pa = results['fit pa'][i])
            # Extract the isophote information
            ES = EllipseSample(dat, sma = results['fit R'][i], geometry = geo)
            ES.update(fixed_parameters = None)
            list_iso.append(Isophote(ES, niter = 30, valid = True, stop_code = 0))
        
        isolist = IsophoteList(list_iso)
        res.update({'fit photutils isolist': isolist,
                    'auxfile fitlimit': 'fit limit semi-major axis: %.2f pix' % isolist.sma[-1]})
    else:
        isolist = results['fit photutils isolist']
    
    for i in range(len(isolist.sma)):
        SBprof_data['R'].append(isolist.sma[i]*options['ap_pixscale'])
        SBprof_data['SB'].append(flux_to_sb(np.median(isolist.sample[i].values[2]), options['ap_pixscale'], zeropoint)) 
        SBprof_data['SB_e'].append(2.5*isolist.int_err[i]/(isolist.intens[i] * np.log(10))) 
        SBprof_data['totmag'].append(flux_to_mag(isolist.tflux_e[i], zeropoint)) 
        SBprof_data['totmag_e'].append(2.5*isolist.rms[i]/(np.sqrt(isolist.npix_e[i])*isolist.tflux_e[i] * np.log(10))) 
        SBprof_data['ellip'].append(isolist.eps[i]) 
        SBprof_data['ellip_e'].append(isolist.ellip_err[i]) 
        SBprof_data['pa'].append(isolist.pa[i]*180/np.pi) 
        SBprof_data['pa_e'].append(isolist.pa_err[i]*180/np.pi) 
        SBprof_data['a3'].append(isolist.a3[i])
        SBprof_data['a3_e'].append(isolist.a3_err[i]) 
        SBprof_data['b3'].append(isolist.b3[i])
        SBprof_data['b3_e'].append(isolist.b3_err[i]) 
        SBprof_data['a4'].append(isolist.a4[i])
        SBprof_data['a4_e'].append(isolist.a4_err[i]) 
        SBprof_data['b4'].append(isolist.b4[i])
        SBprof_data['b4_e'].append(isolist.b4_err[i])
        for k in SBprof_data.keys():
            if not np.isfinite(SBprof_data[k][-1]):
                SBprof_data[k][-1] = 99.999
    res.update({'prof header': params, 'prof units': SBprof_units, 'prof data': SBprof_data, 'prof format': SBprof_format})
    
    if 'ap_doplot' in options and options['ap_doplot']:
        CHOOSE = np.logical_and(np.array(SBprof_data['SB']) < 99, np.array(SBprof_data['SB_e']) < 1)
        if np.sum(CHOOSE) < 5:
            CHOOSE = np.ones(len(CHOOSE), dtype = bool)
        errscale = 1.
        if np.all(np.array(SBprof_data['SB_e'])[CHOOSE] < 0.5):
            errscale = 1/np.max(np.array(SBprof_data['SB_e'])[CHOOSE])
        lnlist = []
        lnlist.append(plt.errorbar(np.array(SBprof_data['R'])[CHOOSE], np.array(SBprof_data['SB'])[CHOOSE], yerr = errscale*np.array(SBprof_data['SB_e'])[CHOOSE],
                                   elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = 'r', label = 'Surface Brightness (err$\\cdot$%.1f)' % errscale))
        plt.errorbar(np.array(SBprof_data['R'])[np.logical_and(CHOOSE,np.arange(len(CHOOSE)) % 4 == 0)],
                     np.array(SBprof_data['SB'])[np.logical_and(CHOOSE,np.arange(len(CHOOSE)) % 4 == 0)],
                     yerr = np.array(SBprof_data['SB_e'])[np.logical_and(CHOOSE,np.arange(len(CHOOSE)) % 4 == 0)],
                     elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = 'limegreen')
        plt.xlabel('Semi-Major-Axis [arcsec]', fontsize = 16)
        plt.ylabel('Surface Brightness [mag arcsec$^{-2}$]', fontsize = 16)
        plt.xlim([0,None])
        bkgrdnoise = -2.5*np.log10(results['background noise']) + zeropoint + 2.5*np.log10(options['ap_pixscale']**2)
        lnlist.append(plt.axhline(bkgrdnoise, color = 'purple', linewidth = 0.5, linestyle = '--', label = '1$\\sigma$ noise/pixel: %.1f mag arcsec$^{-2}$' % bkgrdnoise))
        plt.gca().invert_yaxis()
        plt.tick_params(labelsize = 14)
        labs = [l.get_label() for l in lnlist]
        plt.legend(lnlist, labs, fontsize = 11)
        plt.tight_layout()
        if not ('ap_nologo' in options and options['ap_nologo']):
            AddLogo(plt.gcf())
        plt.savefig('%sphotometry_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
        plt.close()                

        useR = np.array(SBprof_data['R'])[CHOOSE]/options['ap_pixscale']
        useE = np.array(SBprof_data['ellip'])[CHOOSE]
        usePA = np.array(SBprof_data['pa'])[CHOOSE]
        ranges = [[max(0,int(results['center']['x']-useR[-1]*1.2)), min(IMG.shape[1],int(results['center']['x']+useR[-1]*1.2))],
                  [max(0,int(results['center']['y']-useR[-1]*1.2)), min(IMG.shape[0],int(results['center']['y']+useR[-1]*1.2))]]
        LSBImage(dat[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]], results['background noise'])
        fitlim = results['fit R'][-1] if 'fit R' in results else np.inf
        for i in range(len(useR)):
            plt.gca().add_patch(Ellipse((results['center']['x'] - ranges[0][0],results['center']['y'] - ranges[1][0]), 2*useR[i], 2*useR[i]*(1. - useE[i]),
                                        usePA[i], fill = False, linewidth = ((i+1)/len(useR))**2, color = 'limegreen' if (i % 4 == 0) else 'r', linestyle = '-' if useR[i] < fitlim else '--'))
        if not ('ap_nologo' in options and options['ap_nologo']):
            AddLogo(plt.gcf())
        plt.savefig('%sphotometry_ellipse_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
        plt.close()
            
    return IMG, res
