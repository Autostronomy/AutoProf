import numpy as np
from scipy.fftpack import fft, ifft, dct, idct
from scipy.optimize import minimize
from scipy.stats import iqr
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _iso_extract, _x_to_eps, _x_to_pa, _inv_x_to_pa, _inv_x_to_eps
import logging
from copy import copy
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from time import time

def _fitEllip_loss(e, dat, r, p, c, n):
    isovals = _iso_extract(dat,r,e,p,c)
    coefs = fft(np.clip(isovals, a_max = np.quantile(isovals,0.85), a_min = None))
    return np.abs(coefs[2]) / (len(isovals)*(np.abs(np.median(isovals))+n))

def Isophote_Initialize(IMG, pixscale, name, results, **kwargs):
    """
    Determine the global pa and ellipticity for a galaxy. First grow circular isophotes
    until reaching near the noise floor, then evaluate the phase of the second FFT
    coefficients and determine the average direction. Then fit an ellipticity for one
    of the outer isophotes.
    """

    ######################################################################
    # Initial attempt to find size of galaxy in image
    # based on when isophotes SB values start to get
    # close to the background noise level
    circ_ellipse_radii = [results['psf fwhm']]
    phasekeep = []
    allphase = []
    dat = IMG - results['background']

    while circ_ellipse_radii[-1] < (len(IMG)/2):
        circ_ellipse_radii.append(circ_ellipse_radii[-1]*(1+0.2))
        isovals = _iso_extract(dat,circ_ellipse_radii[-1],0.,0.,results['center'], more = True)
        coefs = fft(np.clip(isovals[0], a_max = np.quantile(isovals[0],0.85), a_min = None))
        allphase.append(coefs[2])
        if np.abs(coefs[2]) > np.abs(coefs[1]) and np.abs(coefs[2]) > np.abs(coefs[3]):
            phasekeep.append(coefs[2])
        # Stop when at 3 time background noise
        if np.quantile(isovals[0], 0.8) < (3*results['background noise']) and len(circ_ellipse_radii) > 4: # _iso_extract(IMG - results['background'],circ_ellipse_radii[-1],0.,0.,results['center'])
            break
    logging.info('%s: init scale: %f pix' % (name, circ_ellipse_radii[-1]))
    # Find global position angle.
    if len(phasekeep) >= 5:
        phase = (-np.angle(np.mean(phasekeep[-5:]))/2) % np.pi
    else:
        phase = (-np.angle(np.mean(allphase[int(len(allphase)/2):]))/2) % np.pi

    # Find global ellipticity
    test_ellip = np.linspace(0.05,0.95,15)
    test_f2 = []
    for e in test_ellip:
        test_f2.append(sum(list(_fitEllip_loss(e, dat, circ_ellipse_radii[-2]*m, phase, results['center'], results['background noise']) for m in np.linspace(0.8,1.2,5))))
    ellip = test_ellip[np.argmin(test_f2)]
    res = minimize(lambda e,d,r,p,c,n: sum(list(_fitEllip_loss(_x_to_eps(e[0]),d,r*m,p,c,n) for m in np.linspace(0.8,1.2,5))),
                   x0 = _inv_x_to_eps(ellip), args = (dat, circ_ellipse_radii[-2],
                                                      phase, results['center'],results['background noise']),
                   method = 'Nelder-Mead',options = {'initial_simplex': [[_inv_x_to_eps(ellip)-1/15], [_inv_x_to_eps(ellip)+1/15]]})
    if res.success:
        logging.debug('%s: using optimal ellipticity %.3f over grid ellipticity %.3f' % (name, _x_to_eps(res.x[0]), ellip))
        ellip = _x_to_eps(res.x[0])

    # Compute the error on the parameters
    ######################################################################
    RR = np.linspace(circ_ellipse_radii[-2] - results['psf fwhm'], circ_ellipse_radii[-2] + results['psf fwhm'], 10)
    errallphase = []
    for rr in RR:
        isovals = _iso_extract(dat,rr,0.,0.,results['center'], more = True)
        coefs = fft(np.clip(isovals[0], a_max = np.quantile(isovals[0],0.85), a_min = None))
        errallphase.append(coefs[2])
    sample_pas = (-np.angle(1j*np.array(errallphase)/np.mean(errallphase))/2) % np.pi
    pa_err = iqr(sample_pas, rng = [16,84])/2
    res_multi = map(lambda rrp: minimize(lambda e,d,r,p,c,n: _fitEllip_loss(_x_to_eps(e[0]),d,r,p,c,n),
                                        x0 = _inv_x_to_eps(ellip), args = (dat, rrp[0], rrp[1], results['center'],results['background noise']),
                                         method = 'Nelder-Mead',options = {'initial_simplex': [[_inv_x_to_eps(ellip)-1/15], [_inv_x_to_eps(ellip)+1/15]]}), zip(RR,sample_pas))
    ellip_err = iqr(list(_x_to_eps(rm.x[0]) for rm in res_multi),rng = [16,84])/2
        
    circ_ellipse_radii = np.array(circ_ellipse_radii)
    
    if name != '' and 'doplot' in kwargs and kwargs['doplot']:

        ranges = [[max(0,int(results['center']['x']-circ_ellipse_radii[-1]*1.5)), min(dat.shape[1],int(results['center']['x']+circ_ellipse_radii[-1]*1.5))],
                  [max(0,int(results['center']['y']-circ_ellipse_radii[-1]*1.5)), min(dat.shape[0],int(results['center']['y']+circ_ellipse_radii[-1]*1.5))]]
        
        plt.imshow(np.clip(dat[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]],a_min = 0, a_max = None),
                   origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch())) 
        plt.gca().add_patch(Ellipse((results['center']['x'] - ranges[0][0],results['center']['y'] - ranges[1][0]), 2*circ_ellipse_radii[-1], 2*circ_ellipse_radii[-1]*(1. - ellip),
                                    phase*180/np.pi, fill = False, linewidth = 1, color = 'y'))
        plt.plot([results['center']['x'] - ranges[0][0]],[results['center']['y'] - ranges[1][0]], marker = 'x', markersize = 3, color = 'r')
        plt.savefig('%sinitialize_ellipse_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
        plt.close()

        # paper plot
        # fig, ax = plt.subplots(2,1)
        # ax[0].plot(circ_ellipse_radii[:-1], ((-np.angle(allphase)/2) % np.pi)*180/np.pi, color = 'k')
        # ax[0].axhline(phase*180/np.pi, color = 'r')
        # ax[0].axhline((phase+pa_err)*180/np.pi, color = 'r', linestyle = '--')
        # ax[0].axhline((phase-pa_err)*180/np.pi, color = 'r', linestyle = '--')
        # ax[0].set_xlabel('Radius [pix]')
        # ax[0].set_ylabel('FFT$_{1}$ phase [deg]')
        # ax[1].plot(test_ellip, test_f2, color = 'k')
        # ax[1].axvline(ellip, color = 'r')
        # ax[1].axvline(ellip + ellip_err, color = 'r', linestyle = '--')
        # ax[1].axvline(ellip - ellip_err, color = 'r', linestyle = '--')
        # ax[1].set_xlabel('Ellipticity [1 - b/a]')
        # ax[1].set_ylabel('Loss [FFT$_{2}$/med(flux)]')
        # plt.tight_layout()
        # plt.savefig('%sinitialize_ellipse_optimize_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
        # plt.close()
        
    return {'init ellip': ellip, 'init ellip_err': ellip_err, 'init pa': phase, 'init pa_err': pa_err, 'init R': circ_ellipse_radii[-2]}
