import numpy as np
from scipy.stats import iqr
from scipy.fftpack import fft, ifft
from scipy.optimize import minimize
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
from autoprofutils.SharedFunctions import _iso_extract, _x_to_pa, _x_to_eps, _inv_x_to_eps, _inv_x_to_pa
from autoprofutils.Isophote_Initialize import Isophote_Initialize_CircFit
from autoprofutils.Check_Fit import Check_Fit_IQR

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RANSACRegressor, HuberRegressor

def _ellip_smooth(R, E, deg):
    model = make_pipeline(PolynomialFeatures(deg), HuberRegressor(epsilon=2.))
    model.fit(np.log10(R).reshape(-1,1), _inv_x_to_eps(E))
    return _x_to_eps(model.predict(np.log10(R).reshape(-1,1)))
    
def _pa_smooth(R, PA, deg):

    model_s = make_pipeline(PolynomialFeatures(deg), HuberRegressor())
    model_c = make_pipeline(PolynomialFeatures(deg), HuberRegressor())
    model_c.fit(np.log10(R).reshape(-1,1), np.cos(2*PA))
    model_s.fit(np.log10(R).reshape(-1,1), np.sin(2*PA))
    pred_pa_s = np.clip(model_s.predict(np.log10(R).reshape(-1,1)), a_min = -1, a_max = 1)
    pred_pa_c = np.clip(model_c.predict(np.log10(R).reshape(-1,1)), a_min = -1, a_max = 1)

    return ((np.arctan(pred_pa_s/pred_pa_c) + (np.pi*(pred_pa_c < 0))) % (2*np.pi))/2
    

def _FFT_center_loss(dat, R, E, PA, C, noise, name = ''):
    fft_loss = 0
    for i in range(len(R)):
        isovals = _iso_extract(dat,R[i],E[i],PA[i],C)
        coefs = fft(np.clip(isovals, a_max = np.quantile(isovals,0.85), a_min = None))
        fft_loss += np.abs(coefs[1])/(len(isovals)*(abs(np.median(isovals)) + noise))
    return fft_loss

def _FFT_Robust_loss(dat, R, E, PA, i, C, noise, reg_scale = 1., break_index = 0, name = ''):

    isovals = _iso_extract(dat,R[i],E[i],PA[i],C)
    
    if not np.all(np.isfinite(isovals)):
        logging.warning('Failed to evaluate isophotal flux values, skipping this ellip/pa combination')
        return np.inf
    
    coefs = fft(np.clip(isovals, a_max = np.quantile(isovals,0.85), a_min = None))
    
    f2_loss = np.abs(coefs[2]) / (len(isovals)*(abs(np.median(isovals))))

    if i == 0:
        reg_loss = abs((E[0] - E[1])/0.1)
        reg_loss += abs(np.arccos(np.sin(2*PA[0])*np.sin(2*PA[1]) + np.cos(2*PA[0])*np.cos(2*PA[1]))/(2*0.3))
        reg_loss *= 2
    elif i == (len(R)-1):
        reg_loss = abs((E[-1] - E[-2])/0.1)
        reg_loss += abs(np.arccos(np.sin(2*PA[-1])*np.sin(2*PA[-2]) + np.cos(2*PA[-1])*np.cos(2*PA[-2]))/(2*0.3))
        reg_loss *= 2
    else:
        reg_loss = 0
        # if break_index != i:
        #     reg_loss += abs((E[i] - E[i+1])/0.1)
        #     reg_loss += abs(np.arccos(np.sin(2*PA[i])*np.sin(2*PA[i+1]) + np.cos(2*PA[i])*np.cos(2*PA[i+1]))/(2*0.3))
        if break_index != i-1:
            reg_loss += abs((E[i] - E[i-1])/0.1)
            reg_loss += abs(np.arccos(np.sin(2*PA[i])*np.sin(2*PA[i-1]) + np.cos(2*PA[i])*np.cos(2*PA[i-1]))/(2*0.3))

    return f2_loss + (np.abs(coefs[2])/(len(isovals)*(abs(np.median(isovals)))))*reg_loss*reg_scale

def Isophote_Fit_FFT_Robust(IMG, pixscale, name, results, **kwargs):
    """
    """

    if 'scale' in kwargs:
        scale = kwargs['scale']
    else:
        scale = 0.2

    # subtract background from image during processing
    dat = IMG - results['background']['background']

    # Determine sampling radii
    ######################################################################
    shrink = 0
    while shrink < 5:
        sample_radii = [results['psf']['fwhm']/(1.5+shrink)]
        while sample_radii[-1] < (max(IMG.shape)/2):
            isovals = _iso_extract(dat,sample_radii[-1],results['isophoteinit']['ellip'],
                                   results['isophoteinit']['pa'],results['center'], more = True)
            if np.median(isovals[0]) < 2*results['background']['noise']:
                break
            sample_radii.append(sample_radii[-1]*(1.+scale/(1.+shrink)))
        if len(sample_radii) < 15:
            shrink += 1
        else:
            break
    if shrink >= 5:
        raise Exception('Unable to initialize ellipse fit, check diagnostic plots. Possible missed center.')
    ellip = np.ones(len(sample_radii))*results['isophoteinit']['ellip']
    ellip[0] = 0.05
    pa = np.ones(len(sample_radii))*results['isophoteinit']['pa']
    logging.debug('%s: sample radii: %s' % (name, str(sample_radii)))
    
    # Fit isophotes
    ######################################################################
    perturb_scale = np.array([0.03, 0.06])

    N_perturb = 4

    count = 0

    count_nochange = 0
    use_center = copy(results['center'])
    break_index = len(sample_radii)
    while count < 300 and count_nochange < 3*len(sample_radii):
        # Periodically include logging message
        if count % 10 == 0:
            logging.debug('%s: count: %i' % (name,count))
        count += 1

        if count % 5 == 0:
            smooth_ellip = np.array(list(np.mean(ellip[i:i+3]) for i in range(len(sample_radii)-3)))
            smooth_ellip[0] = smooth_ellip[1]
            smooth_pa = np.array(list(np.angle(np.mean(np.exp(2j*pa[i:i+3]))) for i in range(len(sample_radii)-3)))
            phase_dist = np.sqrt((np.arccos(np.sin(smooth_pa[:-3])*np.sin(smooth_pa[3:]) + np.cos(smooth_pa[:-3])*np.cos(smooth_pa[3:]))/(10*np.pi/180))**2 + ((smooth_ellip[:-3] - smooth_ellip[3:])/0.07)**2)
            break_index = np.argmax(phase_dist)+2 #np.clip(break_index + np.random.randint(-1,2), a_min = 1, a_max = len(sample_radii)-2)
            if break_index < 6 or break_index > len(sample_radii)-5 or max(phase_dist) < 1:
                break_index = len(sample_radii)
        I = np.array(range(1,len(sample_radii)))
        np.random.shuffle(I)
        for i in I:
            perturbations = []
            perturbations.append({'ellip': copy(ellip), 'pa': copy(pa)})
            perturbations[-1]['loss'] = _FFT_Robust_loss(dat, sample_radii, perturbations[-1]['ellip'], perturbations[-1]['pa'], i,
                                                         use_center, results['background']['noise'], count/150, break_index = break_index, name = name)
            for n in range(N_perturb):
                perturbations.append({'ellip': copy(ellip), 'pa': copy(pa)})
                if count % 3 in [0,1]:
                    perturbations[-1]['ellip'][i] = np.clip(perturbations[-1]['ellip'][i] + np.random.normal(loc = 0, scale = perturb_scale[0]), a_min = 0.04, a_max = 0.96)
                if count % 3 in [1,2]:
                    perturbations[-1]['pa'][i] = (perturbations[-1]['pa'][i] + np.random.normal(loc = 0, scale = perturb_scale[1])) % np.pi
                perturbations[-1]['loss'] = _FFT_Robust_loss(dat, sample_radii, perturbations[-1]['ellip'], perturbations[-1]['pa'], i,
                                                             use_center, results['background']['noise'], count/150, break_index = break_index, name = name)
            
            best = np.argmin(list(p['loss'] for p in perturbations))
            if best > 0:
                ellip = copy(perturbations[best]['ellip'])
                pa = copy(perturbations[best]['pa'])
                count_nochange = 0
            else:
                count_nochange += 1

        if count % 10 == 0:
            plt.scatter(sample_radii, _inv_x_to_eps(ellip), color = 'r', label = 'ellip')
            plt.scatter(sample_radii, pa/np.pi, color = 'b', label = 'pa')
            if break_index < len(sample_radii):
                show_ellip = np.zeros(len(sample_radii))
                show_pa = np.zeros(len(sample_radii))
                show_ellip[:break_index+1] = _ellip_smooth(sample_radii[:break_index+1], ellip[:break_index+1], deg = 3)
                show_ellip[break_index+1:] = _ellip_smooth(sample_radii[break_index+1:], ellip[break_index+1:], deg = 3)
                show_pa[:break_index+1] = _pa_smooth(sample_radii[:break_index+1], pa[:break_index+1], deg = 3)
                show_pa[break_index+1:] = _pa_smooth(sample_radii[break_index+1:], pa[break_index+1:], deg = 3)
            else:
                show_ellip = _ellip_smooth(sample_radii, ellip, deg = 4)
                show_pa = _pa_smooth(sample_radii, pa, deg = 4)
            plt.plot(sample_radii, _inv_x_to_eps(show_ellip), color = 'orange', linewidth = 2, linestyle='--', label = 'huber ellip')
            plt.plot(sample_radii, show_pa/np.pi, color = 'purple', linewidth = 2, linestyle='--', label = 'huber pa')
            #plt.xscale('log')
            plt.legend()
            plt.savefig('plots/isoprof_%s_%i.jpg' % (name, count))
            plt.clf()
                
    logging.info('%s: Completed isohpote fit in %i itterations' % (name, count))
    # Smooth ellip and pa profile
    ######################################################################
    old_ellip = copy(ellip)
    old_pa = copy(pa)
    ellip[:3] = min(ellip[:3])
    if break_index < len(sample_radii):
        ellip[:break_index+1] = _ellip_smooth(sample_radii[:break_index+1], ellip[:break_index+1], deg = 3)
        ellip[break_index+1:] = _ellip_smooth(sample_radii[break_index+1:], ellip[break_index+1:], deg = 3)
        pa[:break_index+1] = _pa_smooth(sample_radii[:break_index+1], pa[:break_index+1], deg = 3)
        pa[break_index+1:] = _pa_smooth(sample_radii[break_index+1:], pa[break_index+1:], deg = 3)
    else:
        ellip = _ellip_smooth(sample_radii, ellip, 4)
        pa = _pa_smooth(sample_radii, pa, 4)
    
    if 'doplot' in kwargs and kwargs['doplot']:    
        plt.imshow(np.clip(dat[max(0,int(use_center['y']-sample_radii[-1]*1.2)): min(dat.shape[0],int(use_center['y']+sample_radii[-1]*1.2)),
                               max(0,int(use_center['x']-sample_radii[-1]*1.2)): min(dat.shape[1],int(use_center['x']+sample_radii[-1]*1.2))],
                           a_min = 0,a_max = None), origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch())) 
        for i in range(len(sample_radii)):
            plt.gca().add_patch(Ellipse((int(sample_radii[-1]*1.2),int(sample_radii[-1]*1.2)), 2*sample_radii[i], 2*sample_radii[i]*(1. - ellip[i]),
                                        pa[i]*180/np.pi, fill = False, linewidth = 0.5, color = 'y' if i == break_index else 'r'))
        plt.savefig('%sloss_ellipse_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), dpi = 300)
        plt.clf()                

    # Compute errors
    ######################################################################
    ellip_err = np.zeros(len(ellip))
    ellip_err[:2] = np.sqrt(np.sum((ellip[:4] - old_ellip[:4])**2)/4)
    ellip_err[-1] = np.sqrt(np.sum((ellip[-4:] - old_ellip[-4:])**2)/4)
    pa_err = np.zeros(len(pa))
    pa_err[:2] = np.sqrt(np.sum((pa[:4] - old_pa[:4])**2)/4)
    pa_err[-1] = np.sqrt(np.sum((pa[-4:] - old_pa[-4:])**2)/4)
    for i in range(2,len(pa)-1):
        ellip_err[i] = np.sqrt(np.sum((ellip[i-2:i+2] - old_ellip[i-2:i+2])**2)/4)
        pa_err[i] = np.sqrt(np.sum((pa[i-2:i+2] - old_pa[i-2:i+2])**2)/4)

    res = {'ellip': ellip, 'pa': pa, 'R': sample_radii, 'ellip_err': ellip_err, 'pa_err': pa_err}
    if break_index < len(sample_radii):
        res['break radius'] = sample_radii[break_index]
    return res


def _FFT_loss(x, dat, R, C, noise, name = ''):

    isovals = _iso_extract(dat,R,x[0],x[1],C)
    
    if not np.all(np.isfinite(isovals)):
        logging.warning('Failed to evaluate isophotal flux values, skipping this ellip/pa combination')
        return np.inf
    
    coefs = fft(np.clip(isovals, a_max = np.quantile(isovals,0.85), a_min = None))
    
    f2_loss = np.abs(coefs[2]) / np.abs(coefs[0])
    return f2_loss

def Isophote_Fit_FFT(IMG, pixscale, name, results, **kwargs):

    if 'scale' in kwargs:
        scale = kwargs['scale']
    else:
        scale = 0.3

    # subtract background from image during processing
    dat = IMG - results['background']['background']

    # Determine sampling radii
    ######################################################################
    shrink = 0
    while shrink < 5:
        sample_radii = [2*results['psf']['fwhm']/(1.+shrink)]
        floor_count = 0
        while sample_radii[-1] < (max(IMG.shape)/2) and floor_count < 2:
            sample_radii.append(sample_radii[-1]*(1.+scale/(1.+shrink)))
            isovals = _iso_extract(dat,sample_radii[-1],results['isophoteinit']['ellip'],
                                   results['isophoteinit']['pa'],results['center'], more = True)
            if np.median(isovals[0]) <= results['background']['noise']:
                floor_count += 1
        if len(sample_radii) < 10:
            shrink += 1
        else:
            break
    if shrink >= 5:
        raise Exception('Unable to initialize ellipse fit, check diagnostic plots. Possible missed center.')
    logging.info('%s: sample radii: %s' % (name, str(sample_radii)))
    
    # Fit isophotes
    ######################################################################
    ellip = []
    pa = []
    for i in range(len(sample_radii)):
        res = minimize(lambda x,d,r,c,n: _FFT_loss([_x_to_eps(x[0]),_x_to_pa(x[1])],d,r,c,n), x0 = [_inv_x_to_eps(results['isophoteinit']['ellip']), _x_to_pa(results['isophoteinit']['pa'])], args = (dat, sample_radii[i], results['center'], results['background']['noise']), method = 'Nelder-Mead')
        ellip.append(_x_to_eps(res.x[0]))
        pa.append(_x_to_pa(res.x[1]))
    
    plt.imshow(np.clip(dat, a_min = 0,a_max = None), origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch())) 
    for i in range(len(sample_radii)):
        plt.gca().add_patch(Ellipse((results['center']['x'],results['center']['y']), 2*sample_radii[i], 2*sample_radii[i]*(1. - ellip[i]),
                                    pa[i]*180/np.pi, fill = False, linewidth = 0.2, color = 'y'))
    plt.savefig('%sloss_ellipse_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), dpi = 300)
    plt.clf()                
    return {'ellip': np.array(ellip), 'pa': np.array(pa), 'R': np.array(sample_radii)}

def Isophote_Fit_Forced(IMG, pixscale, name, results, **kwargs):

    with open(kwargs['forcing_profile'], 'r') as f:
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
                
    if 'doplot' in kwargs and kwargs['doplot']:
        dat = IMG - results['background']['background']
        plt.imshow(np.clip(dat[max(0,int(results['center']['y']-(np.array(force['R'])[-1]/pixscale)*1.2)): min(dat.shape[0],int(results['center']['y']+(np.array(force['R'])[-1]/pixscale)*1.2)),
                               max(0,int(results['center']['x']-(np.array(force['R'])[-1]/pixscale)*1.2)): min(dat.shape[1],int(results['center']['x']+(np.array(force['R'])[-1]/pixscale)*1.2))],
                           a_min = 0,a_max = None), origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch())) 
        for i in range(0,len(np.array(force['R'])),2):
            plt.gca().add_patch(Ellipse((int((np.array(force['R'])[-1]/pixscale)*1.2),int((np.array(force['R'])[-1]/pixscale)*1.2)), 2*(np.array(force['R'])[i]/pixscale), 2*(np.array(force['R'])[i]/pixscale)*(1. - force['ellip'][i]),
                                        force['pa'][i], fill = False, linewidth = 0.5, color = 'r'))
        plt.savefig('%sloss_ellipse_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), dpi = 300)
        plt.clf()                
    return {'ellip': np.array(force['ellip']),
            'pa': np.array(force['pa'])*np.pi/180,
            'R': list(np.array(force['R'])/pixscale)}
