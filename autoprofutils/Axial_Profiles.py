import numpy as np
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _iso_extract, _iso_between, Angle_TwoAngles, LSBImage, _iso_line, AddLogo, autocmap, _average, _scatter, flux_to_sb
from autoprofutils.Diagnostic_Plots import Plot_Axial_Profiles
from scipy.stats import iqr
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
import matplotlib
import logging

def Axial_Profiles(IMG, results, options):

    mask = results['mask'] if 'mask' in results else None
    pa = results['init pa'] + ((options['ap_axialprof_pa']*np.pi/180) if 'ap_axialprof_pa' in options else 0.) 
    dat = IMG - results['background']
    zeropoint = options['ap_zeropoint'] if 'ap_zeropoint' in options else 22.5

    if 'prof data' in results:
        Rproflim = results['prof data']['R'][-1]/options['ap_pixscale']
    else:
        Rproflim = min(IMG.shape)/2
    
    R = [0]
    while R[-1] < Rproflim:
        if 'ap_samplestyle' in options and options['ap_samplestyle'] == 'linear':
            step = options['ap_samplelinearscale'] if 'ap_samplelinearscale' in options else 0.5*results['psf fwhm']
        else:
            step = R[-1]*(options['ap_samplegeometricscale'] if 'ap_samplegeometricscale' in options else 0.1)
        R.append(R[-1] + max(1,step))

    sb = {}
    sbE = {}
    for rd in [1, -1]:
        for ang in [1, -1]:
            key = (rd,ang)
            sb[key] = []
            sbE[key] = []
            branch_pa = (pa + ang*np.pi/2) % (2*np.pi)
            for pi, pR in enumerate(R):
                sb[key].append([])
                sbE[key].append([])
                width = (R[pi] - R[pi-1]) if pi > 0 else 1.
                flux, XX = _iso_line(dat, R[-1], width, branch_pa,
                                     {'x': results['center']['x'] + ang*rd*pR*np.cos(pa + (0 if ang > 0 else np.pi)),
                                      'y': results['center']['y'] + ang*rd*pR*np.sin(pa + (0 if ang > 0 else np.pi))})
                for oi, oR in enumerate(R):
                    length = (R[oi] - R[oi-1]) if oi > 0 else 1.
                    CHOOSE = np.logical_and(XX > (oR - length/2), XX < (oR + length/2))
                    if np.sum(CHOOSE) == 0:
                        sb[key][-1].append(99.999)
                        sbE[key][-1].append(99.999)
                        continue
                    medflux = _average(flux[CHOOSE], options['ap_isoaverage_method'] if 'ap_isoaverage_method' in options else 'median')
                    scatflux = _scatter(flux[CHOOSE], options['ap_isoaverage_method'] if 'ap_isoaverage_method' in options else 'median')
                    sb[key][-1].append(flux_to_sb(medflux, options['ap_pixscale'], zeropoint) if medflux > 0 else 99.999)
                    sbE[key][-1].append((2.5*scatflux / (np.sqrt(np.sum(CHOOSE))*medflux*np.log(10))) if medflux > 0 else 99.999)
                    

    with open('%s%s_axial_profile.prof' % ((options['ap_saveto'] if 'ap_saveto' in options else ''), options['ap_name']), 'w') as f:
        f.write('R')
        for rd in [1,-1]:
            for ang in [1, -1]:
                for pR in R:
                    f.write(',sb[%.3f:%s90],sbE[%.3f:%s90]' % (rd*pR*options['ap_pixscale'], '+' if ang > 0 else '-', rd*pR*options['ap_pixscale'], '+' if ang > 0 else '-'))
        f.write('\n')
        f.write('arcsec')
        for rd in [1,-1]:
            for ang in [1, -1]:
                for pR in R:
                    f.write(',mag*arcsec^-2,mag*arcsec^-2')
        f.write('\n')
        for oi, oR in enumerate(R):
            f.write('%.4f' % (oR*options['ap_pixscale']))
            for rd in [1,-1]:
                for ang in [1, -1]:
                    key = (rd,ang)
                    for pi, pR in enumerate(R):
                        f.write(',%.4f,%.4f' % (sb[key][pi][oi], sbE[key][pi][oi]))
            f.write('\n')

    if 'ap_doplot' in options and options['ap_doplot']:
        Plot_Axial_Profiles(dat, R, sb, sbE, pa, results, options)
            
    return IMG, {}
