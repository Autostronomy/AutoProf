import numpy as np
from scipy.fftpack import fft, ifft, dct, idct
from scipy.optimize import minimize
from scipy.stats import iqr
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.Elliptical_Isophotes import _Fit_Isophotes_Loss
from autoprofutils.SharedFunctions import _iso_extract, _x_to_eps, _x_to_pa, _inv_x_to_pa, _inv_x_to_eps
import logging
from copy import copy
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from time import time

def Isophote_Initialize_GridSearch(IMG, pixscale, name, results, **kwargs):
    """
    Determine the global pa and ellipticity for a galaxy
    """
    ######################################################################
    # Find global ellipticity and position angle.
    # Initial attempt to find size of galaxy in image
    # based on when isophotes SB values start to get
    # close to the background noise level
    circ_ellipse_radii = [results['psf']['median']]
    while circ_ellipse_radii[-1] < (len(IMG)/2):
        circ_ellipse_radii.append(circ_ellipse_radii[-1]*(1+0.3))
        # Stop when at 10 time background noise
        if np.quantile(_iso_extract(IMG - results['background']['median'],circ_ellipse_radii[-1],0.,0.,results['center']), 0.6) < (3*results['background']['iqr']) and len(circ_ellipse_radii) > 4:
            break
    logging.info('%s: init scale: %f' % (name, circ_ellipse_radii[-1]))
    circ_ellipse_radii = np.array(circ_ellipse_radii)

    ######################################################################
    # Large scale fit with constant pa and ellipticity via grid search.
    # simultaneously fits at all scales as rough galaxy radius is not
    # very accurate yet.

    # Make list of pa and ellipticity values to use for grid search
    initializations = []
    N_e, N_pa = 10, 20
    for e in np.linspace(_inv_x_to_eps(0.2),_inv_x_to_eps(0.8),N_e): 
        for p in np.linspace(0., (N_pa-1)*np.pi/N_pa,N_pa): 
            initializations.append((e,p))

    # Cycle through pa and ellipticity values and compute loss
    best = [np.inf, -1, []]
    loss_results = []
    logging.info('%s: Initializing' % name)
    for e,p in initializations:
        logging.debug('%s: %.2f, %.2f' % (name,e,p))
        loss_results.append(_Fit_Isophotes_Loss([[e,p]]*len(circ_ellipse_radii), IMG,
                                                circ_ellipse_radii, results['center'],
                                                range(len(circ_ellipse_radii)),
                                                iso_norms = [], fftlim = 7))
        # Track best pa/ellipticity combination so far
        if loss_results[-1] < best[0]:
            best[0] = loss_results[-1]
            best[1] = int(len(loss_results)-1)
            best[2] = copy([e,p])
            logging.debug('%s: best: %f %s' % (name,best[0], str(best[2][0])))
    if name != '' and 'doplot' in kwargs and kwargs['doplot']:
        plt.imshow(np.clip(IMG,a_min = 0, a_max = None), origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch())) 
        plt.gca().add_patch(Ellipse((results['center']['x'],results['center']['y']), 2*circ_ellipse_radii[-1], 2*circ_ellipse_radii[-1]*(1. - _x_to_eps(best[2][0])),
                                    _x_to_pa(best[2][1])*180/np.pi, fill = False, linewidth = 1, color = 'y'))
        plt.plot([results['center']['x']],[results['center']['y']], marker = 'x', markersize = 10, color = 'y')
        plt.savefig('%sinitialize_ellipse_%s.png' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
        plt.clf()

    logging.info('%s: best initialization: %s' % (name, str(best[2][0])))
    return {'ellip': best[2][0], 'pa': best[2][1]}

def _CircfitEllip_loss(e, dat, r, p, c):
    isovals = _iso_extract(dat,r,e,p,c)
    coefs = fft(np.clip(isovals, a_max = np.quantile(isovals,0.85), a_min = None))
    return np.abs(coefs[2]) / (len(isovals)*np.abs(np.median(isovals)))


def Isophote_Initialize_CircFit(IMG, pixscale, name, results, **kwargs):
    """
    Determine the global pa and ellipticity for a galaxy
    """

    ######################################################################
    # Find global ellipticity and position angle.
    # Initial attempt to find size of galaxy in image
    # based on when isophotes SB values start to get
    # close to the background noise level
    circ_ellipse_radii = [results['psf']['median']]
    phasekeep = []
    allphase = []
    dat = IMG - results['background']['median']

    while circ_ellipse_radii[-1] < (len(IMG)/2):
        circ_ellipse_radii.append(circ_ellipse_radii[-1]*(1+0.2))
        isovals = _iso_extract(dat,circ_ellipse_radii[-1],0.,0.,results['center'], more = True)
        coefs = fft(np.clip(isovals[0], a_max = np.quantile(isovals[0],0.85), a_min = None))
        allphase.append(coefs[2])
        if np.abs(coefs[2]) > np.abs(coefs[1]) and np.abs(coefs[2]) > np.abs(coefs[3]):
            phasekeep.append(coefs[2])
        # plt.plot(isovals[1], isovals[0], label = '%.2f' % circ_ellipse_radii[-1])
        # print(len(isovals[0]), len(ifft(coefs)), len(ifft(coefs[:5], n = len(isovals[0]))))
        # print(coefs[:5])
        # print('start angle: ', isovals[1][0], np.angle(coefs[2]), np.angle(coefs[2]) % np.pi, np.angle(coefs[2]) % (2*np.pi), np.abs(np.angle(coefs[2]))/2)
        # plt.plot(isovals[1], ifft(coefs[:5], n = len(isovals[0])), label = 'FFT 5')
        # test_point = np.zeros(len(coefs), dtype = np.csingle)
        # test_point[0] = coefs[0]
        # test_angle = np.angle(coefs[2])
        # test_point[2] = np.max(np.abs(coefs[1:]))*(np.cos(test_angle) + np.sin(test_angle)*1j)
        # print('test ', test_point[:5], np.angle(test_point[:5]))
        # plt.plot(isovals[1], ifft(test_point), label = 'testpoint')
        # plt.axvline(( - np.angle(coefs[2])/2) % np.pi)
        # # plt.plot(isovals[1], ifft(coefs[:4], n = len(isovals[0])), label = 'FFT 4')
        # # plt.plot(isovals[1], ifft(coefs[:3], n = len(isovals[0])), label = 'FFT 3')
        # plt.title('Pass test: %s' % str(np.abs(coefs[2]) > np.abs(coefs[1]) and np.abs(coefs[2]) > np.abs(coefs[3])))
        # plt.xlabel('angle')
        # plt.ylabel('flux')
        # plt.legend()
        # plt.savefig('%sinitialize_test_%s_%.2i.png' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name, len(circ_ellipse_radii)))
        # plt.clf()
        # plt.plot(range(1,len(coefs)), np.abs(coefs[1:]))
        # plt.savefig('%sinitializefft_coefs_%s_%.2i.png' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name, len(circ_ellipse_radii)))
        # plt.clf()
        # Stop when at 3 time background noise
        if np.quantile(isovals[0], 0.8) < (3*results['background']['iqr']) and len(circ_ellipse_radii) > 4: # _iso_extract(IMG - results['background']['median'],circ_ellipse_radii[-1],0.,0.,results['center'])
            break
    logging.info('%s: init scale: %f' % (name, circ_ellipse_radii[-1]))
    if len(phasekeep) >= 5:
        phase = np.median((-np.angle(phasekeep[-5:])/2) % np.pi)
    else:
        phase = np.median((-np.angle(allphase[int(len(allphase)/2):])/2) % np.pi)

    start = time()
    test_ellip = np.linspace(0.05,0.95,15)
    test_f2 = []
    for e in test_ellip:
        test_f2.append(_CircfitEllip_loss(e, dat, circ_ellipse_radii[-2], phase, results['center']))
    ellip = test_ellip[np.argmin(test_f2)]
    res = minimize(lambda e,d,r,p,c: _CircfitEllip_loss(_x_to_eps(e[0]),d,r,p,c),
                   x0 = _inv_x_to_eps(ellip), args = (dat, circ_ellipse_radii[-2],
                                                      phase, results['center']),
                   method = 'Nelder-Mead',options = {'initial_simplex': [[_inv_x_to_eps(ellip)-1/15], [_inv_x_to_eps(ellip)+1/15]]})
    if res.success:
        logging.debug('%s: using optimal ellipticity %.3f over grid ellipticity %.3f' % (name, _x_to_eps(res.x[0]), ellip))
        ellip = _x_to_eps(res.x[0])
    # logging.info('%s: ellipticity time: %f' % (name, time() - start))
    # plt.plot(test_ellip, np.array(test_iqr) - np.mean(test_iqr) , label = 'iqr', color = 'r')
    # plt.plot(test_ellip, np.array(test_f2) - np.mean(test_f2), label = 'f2', color = 'b')
    # # te = np.linspace(0.15,0.9,20)
    # # plt.plot(te, np.polyval(p, te), label = 'p: %s' % str(p))
    # plt.axvline(ellip, color = 'r')
    # plt.axvline(ellip2, color = 'b')
    # plt.xlabel('ellipticity')
    # plt.ylabel('isovals')
    # plt.legend()
    # plt.savefig('%sinitialize_ellip_%s.png' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
    # plt.clf()
        
    circ_ellipse_radii = np.array(circ_ellipse_radii)
    if name != '' and 'doplot' in kwargs and kwargs['doplot']:
        plt.imshow(np.clip(dat,a_min = 0, a_max = None), origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch())) 
        plt.gca().add_patch(Ellipse((results['center']['x'],results['center']['y']), 2*circ_ellipse_radii[-1], 2*circ_ellipse_radii[-1]*(1. - ellip),
                                    phase*180/np.pi, fill = False, linewidth = 1, color = 'y'))
        plt.plot([results['center']['x']],[results['center']['y']], marker = 'x', markersize = 3, color = 'y')
        plt.savefig('%sinitialize_ellipse_%s.png' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
        plt.clf()
    return {'ellip': ellip, 'pa': phase}
