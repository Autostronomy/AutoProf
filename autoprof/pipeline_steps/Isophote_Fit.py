import numpy as np
from scipy.stats import iqr
from scipy.fftpack import fft, ifft
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from time import time
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.isophote import EllipseSample, EllipseGeometry, Isophote, IsophoteList
from photutils.isophote import Ellipse as Photutils_Ellipse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from copy import copy
import logging
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _iso_extract, _x_to_pa, _x_to_eps, _inv_x_to_eps, _inv_x_to_pa, Angle_TwoAngles, Angle_Scatter, LSBImage, AddLogo, PA_shift_convention, autocolours
from autoprofutils.Diagnostic_Plots import Plot_Isophote_Fit

def Photutils_Fit(IMG, results, options):
    """Photutils elliptical isophote wrapper.
    
    This simply gives users access to the photutils isophote
    fitting method. See: `photutils
    <https://photutils.readthedocs.io/en/stable/isophote.html>`_ for
    more information.

    References
    ----------
    - 'background'
    - 'center'
    - 'init R'
    - 'init ellip'
    - 'init pa'
        
    Returns
    -------
    IMG: ndarray
      Unaltered galaxy image
    
    results: dict
      .. code-block:: python
     
        {'fit ellip': , # array of ellipticity values (ndarray)
         'fit pa': , # array of PA values (ndarray)
         'fit R': , # array of semi-major axis values (ndarray)
         'fit ellip_err': , # optional, array of ellipticity error values (ndarray)
         'fit pa_err': , # optional, array of PA error values (ndarray)
         'auxfile fitlimit': # optional, auxfile message (string)
    
        }    
    """

    dat = IMG - results['background']
    geo = EllipseGeometry(x0 = results['center']['x'],
                          y0 = results['center']['y'],
                          sma = results['init R']/2,
                          eps = results['init ellip'],
                          pa = results['init pa'])
    ellipse = Photutils_Ellipse(dat, geometry = geo)

    isolist = ellipse.fit_image(fix_center = True, linear = False)
    res = {'fit R': isolist.sma[1:], 'fit ellip': isolist.eps[1:], 'fit ellip_err': isolist.ellip_err[1:],
           'fit pa': isolist.pa[1:], 'fit pa_err': isolist.pa_err[1:], 'fit photutils isolist': isolist, 'auxfile fitlimit': 'fit limit semi-major axis: %.2f pix' % isolist.sma[-1]}
    
    if 'ap_doplot' in options and options['ap_doplot']:
        Plot_Isophote_Fit(dat, res['fit R'], res['fit ellip'], res['fit pa'], res['fit ellip_err'], res['fit pa_err'], results, options)

    return IMG, res


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
    

def _FFT_Robust_loss(dat, R, E, PA, i, C, noise, mask = None, reg_scale = 1., name = ''):

    isovals = _iso_extract(dat,R[i],E[i],PA[i],C, mask = mask, interp_mask = False if mask is None else True, interp_method = 'bicubic')
    
    if mask is None:
        coefs = fft(np.clip(isovals, a_max = np.quantile(isovals,0.85), a_min = None))
    else:
        coefs = fft(np.clip(isovals, a_max = np.quantile(isovals,0.9), a_min = None))

    f2_loss = np.abs(coefs[2]) / (len(isovals)*(max(0,np.median(isovals)) + noise/np.sqrt(len(isovals))))

    reg_loss = 0
    if i < (len(R)-1):
        reg_loss += abs((E[i] - E[i+1])/(1 - E[i+1])) 
        reg_loss += abs(Angle_TwoAngles(2*PA[i], 2*PA[i+1])/(2*0.2))
    if i > 0:
        reg_loss += abs((E[i] - E[i-1])/(1 - E[i-1])) 
        reg_loss += abs(Angle_TwoAngles(2*PA[i], 2*PA[i-1])/(2*0.2))

    return f2_loss*(1 + reg_loss*reg_scale)

def _FFT_Robust_Errors(dat, R, E, PA, C, noise, mask = None, reg_scale = 1., name = ''):

    PA_err = np.zeros(len(R))
    E_err = np.zeros(len(R))
    for ri in range(len(R)):
        temp_fits = []
        for i in range(10):
            low_ri = max(0, ri - 1)
            high_ri = min(len(R) - 1, ri + 1)
            temp_fits.append(minimize(lambda x: _FFT_Robust_loss(dat, [R[low_ri], R[ri]*(1 - 0.05 + i*0.1/9), R[high_ri]],
                                                                 [E[low_ri], np.clip(x[0], 0,1), E[high_ri]],
                                                                 [PA[low_ri], x[1] % np.pi, PA[high_ri]],
                                                                 1, C, noise, mask = mask,
                                                                 reg_scale = reg_scale, name = name),
                                      x0 = [E[ri], PA[ri]], method = 'SLSQP', options = {'ftol': 0.001}).x)
        temp_fits = np.array(temp_fits)
        E_err[ri] = iqr(np.clip(temp_fits[:,0], 0, 1), rng = [16,84])/2
        PA_err[ri] = Angle_Scatter(2*(temp_fits[:,1] % np.pi))/4. # multiply by 2 to get [0, 2pi] range
    return E_err, PA_err
        
def Isophote_Fit_FFT_Robust(IMG, results, options):
    """Fit elliptical isophotes to a galaxy image using FFT coefficients and regularization.
    
    The isophotal fitting routine simultaneously optimizes a
    collection of elliptical isophotes by minimizing the 2nd FFT
    coefficient power, regularized for robustness. A series of
    isophotes are constructed which grow geometrically until they
    begin to reach the background level.  Then the algorithm
    iteratively updates the position angle and ellipticity of each
    isophote individually for many rounds.  Each round updates every
    isophote in a random order.  Each round cycles between three
    options: optimizing position angle, ellipticity, or both.  To
    optimize the parameters, 5 values (pa, ellip, or both) are
    randomly sampled and the "loss" is computed.  The loss is a
    combination of the relative amplitude of the second FFT
    coefficient (compared to the median flux), and a regularization
    term.  The regularization term penalizes adjacent isophotes for
    having different position angle or ellipticity (using the l1
    norm).  Thus, all the isophotes are coupled and tend to fit
    smoothly varying isophotes.  When the optimization has completed
    three rounds without any isophotes updating, the profile is
    assumed to have converged.

    An uncertainty for each ellipticity and position angle value is
    determined by taking the RMS between the fitted values and a
    smoothed polynomial fit values for 4 points.  This is a very rough
    estimate of the uncertainty, but works sufficiently well in the
    outskirts.

    Arguments
    -----------------
    ap_scale: float
      growth scale when fitting isophotes, not the same as *ap_sample---scale*.

      :default:
        0.2

    ap_fit_limit: float
      noise level out to which to extend the fit in units of pixel background noise level. Default is 2, smaller values will end fitting further out in the galaxy image.

      :default:
        2

    ap_regularize_scale: float
      scale factor to apply to regularization coupling factor between isophotes.
      Default of 1, larger values make smoother fits, smaller values give more chaotic fits.

      :default:
        1
    
    References
    ----------
    - 'background'
    - 'background noise'
    - 'psf fwhm'
    - 'center'
    - 'mask' (optional)
    - 'init ellip'
    - 'init pa'
        
    Returns
    -------
    IMG: ndarray
      Unaltered galaxy image
    
    results: dict
      .. code-block:: python
     
        {'fit ellip': , # array of ellipticity values (ndarray)
         'fit pa': , # array of PA values (ndarray)
         'fit R': , # array of semi-major axis values (ndarray)
         'fit ellip_err': , # optional, array of ellipticity error values (ndarray)
         'fit pa_err': , # optional, array of PA error values (ndarray)
         'auxfile fitlimit': # optional, auxfile message (string)
    
        }

    """

    if 'ap_scale' in options:
        scale = options['ap_scale']
    else:
        scale = 0.2

    # subtract background from image during processing
    dat = IMG - results['background']
    mask = results['mask'] if 'mask' in results else None
    if not np.any(mask):
        mask = None
    
    # Determine sampling radii
    ######################################################################
    shrink = 0
    while shrink < 5:
        sample_radii = [max(1.,results['psf fwhm']/2)]
        while sample_radii[-1] < (max(IMG.shape)/2):
            isovals = _iso_extract(dat,sample_radii[-1],results['init ellip'],
                                   results['init pa'],results['center'], more = False, mask = mask)
            if np.median(isovals) < (options['ap_fit_limit'] if 'ap_fit_limit' in options else 2)*results['background noise']:
                break
            sample_radii.append(sample_radii[-1]*(1.+scale/(1.+shrink)))
        if len(sample_radii) < 15:
            shrink += 1
        else:
            break
    if shrink >= 5:
        raise Exception('Unable to initialize ellipse fit, check diagnostic plots. Possible missed center.')
    ellip = np.ones(len(sample_radii))*results['init ellip']
    pa = np.ones(len(sample_radii))*results['init pa']
    logging.debug('%s: sample radii: %s' % (options['ap_name'], str(sample_radii)))
    # Fit isophotes
    ######################################################################
    perturb_scale = np.array([0.03, 0.06])
    regularize_scale = options['ap_regularize_scale'] if 'ap_regularize_scale' in options else 1.
    N_perturb = 5

    count = 0

    count_nochange = 0
    use_center = copy(results['center'])
    I = np.array(range(len(sample_radii)))
    while count < 300 and count_nochange < (3*(len(sample_radii)-1)):
        # Periodically include logging message
        if count % 10 == 0:
            logging.debug('%s: count: %i' % (options['ap_name'],count))
        count += 1
        
        np.random.shuffle(I)
        for i in I:
            perturbations = []
            perturbations.append({'ellip': copy(ellip), 'pa': copy(pa)})
            perturbations[-1]['loss'] = _FFT_Robust_loss(dat, sample_radii, perturbations[-1]['ellip'], perturbations[-1]['pa'], i,
                                                         use_center, results['background noise'], mask = mask, reg_scale = regularize_scale if count > 4 else 0, name = options['ap_name'])
            for n in range(N_perturb):
                perturbations.append({'ellip': copy(ellip), 'pa': copy(pa)})
                if count % 3 in [0,1]:
                    perturbations[-1]['ellip'][i] = _x_to_eps(_inv_x_to_eps(perturbations[-1]['ellip'][i]) + np.random.normal(loc = 0, scale = perturb_scale[0]))
                if count % 3 in [1,2]:
                    perturbations[-1]['pa'][i] = (perturbations[-1]['pa'][i] + np.random.normal(loc = 0, scale = perturb_scale[1])) % np.pi
                perturbations[-1]['loss'] = _FFT_Robust_loss(dat, sample_radii, perturbations[-1]['ellip'], perturbations[-1]['pa'], i,
                                                             use_center, results['background noise'], mask = mask, reg_scale = regularize_scale if count > 4 else 0, name = options['ap_name'])
            
            best = np.argmin(list(p['loss'] for p in perturbations))
            if best > 0:
                ellip = copy(perturbations[best]['ellip'])
                pa = copy(perturbations[best]['pa'])
                count_nochange = 0
            else:
                count_nochange += 1

    logging.info('%s: Completed isohpote fit in %i itterations' % (options['ap_name'], count))
    # detect collapsed center
    ######################################################################
    for i in range(5):
        if (_inv_x_to_eps(ellip[i]) - _inv_x_to_eps(ellip[i+1])) > 0.5:
            ellip[:i+1] = ellip[i+1]
            pa[:i+1] = pa[i+1]
            
    # Compute errors
    ######################################################################
    ellip_err, pa_err = _FFT_Robust_Errors(dat, sample_radii, ellip, pa, use_center, results['background noise'],
                                           mask = mask, reg_scale = regularize_scale, name = options['ap_name'])
    # Plot fitting results
    ######################################################################    
    if 'ap_doplot' in options and options['ap_doplot']:
        Plot_Isophote_Fit(dat, sample_radii, ellip, pa, ellip_err, pa_err, results, options)

    res = {'fit ellip': ellip, 'fit pa': pa, 'fit R': sample_radii,
           'fit ellip_err': ellip_err, 'fit pa_err': pa_err,
           'auxfile fitlimit': 'fit limit semi-major axis: %.2f pix' % sample_radii[-1]}
    return IMG, res

def Isophote_Fit_Forced(IMG, results, options):
    """Read previously fit PA/ellipticity profile.

    Reads a .prof file and extracts the corresponding PA/ellipticity profile. The profile is extracted generically, so any csv file with columns for 'R', 'pa', 'ellip', and optionally 'pa_e' and 'ellip_e' will be able to create a forced fit. This can be used for testing purposes, such as selecting a specific isophote to extract or comparing AutoProf SB extraction methods with other softwares.

    Arguments
    -----------------
    ap_forcing_profile: string
        File path to .prof file providing forced photometry PA and
        ellip values to apply to *ap_image_file* (required for forced
        photometry)

      :default:
        None
    
    References
    ----------
    - 'background'
    - 'background noise'
    - 'center'
        
    Returns
    -------
    IMG: ndarray
      Unaltered galaxy image
    
    results: dict
      .. code-block:: python
     
        {'fit ellip': , # array of ellipticity values (ndarray)
         'fit pa': , # array of PA values (ndarray)
         'fit R': , # array of semi-major axis values (ndarray)
         'fit ellip_err': , # optional, array of ellipticity error values (ndarray)
         'fit pa_err': , # optional, array of PA error values (ndarray)
    
        }

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

    force['pa'] = PA_shift_convention(np.array(force['pa']), deg = True)
                
    if 'ap_doplot' in options and options['ap_doplot']:
        Plot_Isophote_Fit(IMG - results['background'], np.array(force['R']), np.array(force['ellip']), np.array(force['pa']),
                          np.array(force['ellip_e']) if 'ellip_e' in force else np.zeros(len(force['R'])),
                          np.array(force['pa_e']) if 'pa_e' in force else np.zeros(len(force['R'])), results, options)
        
    res = {'fit ellip': np.array(force['ellip']),
           'fit pa': np.array(force['pa'])*np.pi/180,
           'fit R': list(np.array(force['R'])/options['ap_pixscale'])}
    if 'ellip_e' in force and 'pa_e' in force:
        res['fit ellip_err'] = np.array(force['ellip_e'])
        res['fit pa_err'] = np.array(force['pa_e'])*np.pi/180
    return IMG, res


######################################################################
def _FFT_mean_loss(dat, R, E, PA, i, C, noise, mask = None, reg_scale = 1., name = ''):

    isovals = _iso_extract(dat,R[i],E[i],PA[i],C, mask = mask, interp_mask = False if mask is None else True)
    
    if not np.all(np.isfinite(isovals)):
        logging.warning('Failed to evaluate isophotal flux values, skipping this ellip/pa combination')
        return np.inf

    coefs = fft(isovals)
    
    f2_loss = np.abs(coefs[2]) / (len(isovals)*(max(0,np.mean(isovals)) + noise))

    reg_loss = 0
    if i < (len(R)-1):
        reg_loss += abs((E[i] - E[i+1])/(1 - E[i+1])) #abs((_inv_x_to_eps(E[i]) - _inv_x_to_eps(E[i+1]))/0.1)
        reg_loss += abs(Angle_TwoAngles(2*PA[i], 2*PA[i+1])/(2*0.3))
    if i > 0:
        reg_loss += abs((E[i] - E[i-1])/(1 - E[i-1])) #abs((_inv_x_to_eps(E[i]) - _inv_x_to_eps(E[i-1]))/0.1)
        reg_loss += abs(Angle_TwoAngles(2*PA[i], 2*PA[i-1])/(2*0.3))

    return f2_loss*(1 + reg_loss*reg_scale) #(np.abs(coefs[2])/(len(isovals)*(abs(np.median(isovals)))))*reg_loss*reg_scale

def Isophote_Fit_FFT_mean(IMG, results, options):
    """Fit elliptical isophotes to a galaxy image using FFT coefficients and regularization.

    Same as the standard isophote fitting routine, except uses less
    robust mean/std measures. This is only intended for low S/N data
    where pixels have low integer counts.

    Arguments
    -----------------
    ap_scale: float
      growth scale when fitting isophotes, not the same as
      *ap_sample---scale*.

      :default:
        0.2

    ap_fit_limit: float
      noise level out to which to extend the fit in units of pixel
      background noise level. Default is 2, smaller values will end
      fitting further out in the galaxy image.

      :default:
        2

    ap_regularize_scale: float
      scale factor to apply to regularization coupling factor between
      isophotes.  Default of 1, larger values make smoother fits,
      smaller values give more chaotic fits.

      :default:
        1
    
    References
    ----------
    - 'background'
    - 'background noise'
    - 'center'
    - 'psf fwhm'
    - 'init ellip'
    - 'init pa'
        
    Returns
    -------
    IMG: ndarray
      Unaltered galaxy image
    
    results: dict
      .. code-block:: python
     
        {'fit ellip': , # array of ellipticity values (ndarray)
         'fit pa': , # array of PA values (ndarray)
         'fit R': , # array of semi-major axis values (ndarray)
         'fit ellip_err': , # optional, array of ellipticity error values (ndarray)
         'fit pa_err': , # optional, array of PA error values (ndarray)
         'auxfile fitlimit': # optional, auxfile message (string)
    
        }

    """

    if 'ap_scale' in options:
        scale = options['ap_scale']
    else:
        scale = 0.2

    # subtract background from image during processing
    dat = IMG - results['background']
    mask = results['mask'] if 'mask' in results else None
    if not np.any(mask):
        mask = None
    
    # Determine sampling radii
    ######################################################################
    shrink = 0
    while shrink < 5:
        sample_radii = [3*results['psf fwhm']/2]
        while sample_radii[-1] < (max(IMG.shape)/2):
            isovals = _iso_extract(dat,sample_radii[-1],results['init ellip'],
                                   results['init pa'],results['center'], more = False, mask = mask)
            if np.mean(isovals) < (options['ap_fit_limit'] if 'ap_fit_limit' in options else 1)*results['background noise']:
                break
            sample_radii.append(sample_radii[-1]*(1.+scale/(1.+shrink)))
        if len(sample_radii) < 15:
            shrink += 1
        else:
            break
    if shrink >= 5:
        raise Exception('Unable to initialize ellipse fit, check diagnostic plots. Possible missed center.')
    ellip = np.ones(len(sample_radii))*results['init ellip']
    pa = np.ones(len(sample_radii))*results['init pa']
    logging.debug('%s: sample radii: %s' % (options['ap_name'], str(sample_radii)))
    
    # Fit isophotes
    ######################################################################
    perturb_scale = np.array([0.03, 0.06])
    regularize_scale = options['ap_regularize_scale'] if 'ap_regularize_scale' in options else 1.
    N_perturb = 5

    count = 0

    count_nochange = 0
    use_center = copy(results['center'])
    I = np.array(range(len(sample_radii)))
    while count < 300 and count_nochange < (3*len(sample_radii)):
        # Periodically include logging message
        if count % 10 == 0:
            logging.debug('%s: count: %i' % (options['ap_name'],count))
        count += 1
        
        np.random.shuffle(I)
        for i in I:
            perturbations = []
            perturbations.append({'ellip': copy(ellip), 'pa': copy(pa)})
            perturbations[-1]['loss'] = _FFT_mean_loss(dat, sample_radii, perturbations[-1]['ellip'], perturbations[-1]['pa'], i,
                                                       use_center, results['background noise'], mask = mask, reg_scale = regularize_scale if count > 4 else 0, name = options['ap_name'])
            for n in range(N_perturb):
                perturbations.append({'ellip': copy(ellip), 'pa': copy(pa)})
                if count % 3 in [0,1]:
                    perturbations[-1]['ellip'][i] = _x_to_eps(_inv_x_to_eps(perturbations[-1]['ellip'][i]) + np.random.normal(loc = 0, scale = perturb_scale[0]))
                if count % 3 in [1,2]:
                    perturbations[-1]['pa'][i] = (perturbations[-1]['pa'][i] + np.random.normal(loc = 0, scale = perturb_scale[1])) % np.pi
                perturbations[-1]['loss'] = _FFT_mean_loss(dat, sample_radii, perturbations[-1]['ellip'], perturbations[-1]['pa'], i,
                                                           use_center, results['background noise'], mask = mask, reg_scale = regularize_scale if count > 4 else 0, name = options['ap_name'])
            
            best = np.argmin(list(p['loss'] for p in perturbations))
            if best > 0:
                ellip = copy(perturbations[best]['ellip'])
                pa = copy(perturbations[best]['pa'])
                count_nochange = 0
            else:
                count_nochange += 1
                
    logging.info('%s: Completed isohpote fit in %i itterations' % (options['ap_name'], count))
    # detect collapsed center
    ######################################################################
    for i in range(5):
        if (_inv_x_to_eps(ellip[i]) - _inv_x_to_eps(ellip[i+1])) > 0.5:
            ellip[:i+1] = ellip[i+1]
            pa[:i+1] = pa[i+1]

    # Smooth ellip and pa profile
    ######################################################################
    smooth_ellip = copy(ellip)
    smooth_pa = copy(pa)
    ellip[:3] = min(ellip[:3])
    smooth_ellip = _ellip_smooth(sample_radii, smooth_ellip, 5)
    smooth_pa = _pa_smooth(sample_radii, smooth_pa, 5)
    
    if 'ap_doplot' in options and options['ap_doplot']:
        ranges = [[max(0,int(use_center['x']-sample_radii[-1]*1.2)), min(dat.shape[1],int(use_center['x']+sample_radii[-1]*1.2))],
                  [max(0,int(use_center['y']-sample_radii[-1]*1.2)), min(dat.shape[0],int(use_center['y']+sample_radii[-1]*1.2))]]
        LSBImage(dat[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]], results['background noise'])
        # plt.imshow(np.clip(dat[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]],
        #                    a_min = 0,a_max = None), origin = 'lower', cmap = 'Greys', norm = ImageNormalize(stretch=LogStretch())) 
        for i in range(len(sample_radii)):
            plt.gca().add_patch(Ellipse((use_center['x'] - ranges[0][0],use_center['y'] - ranges[1][0]), 2*sample_radii[i], 2*sample_radii[i]*(1. - ellip[i]),
                                        pa[i]*180/np.pi, fill = False, linewidth = ((i+1)/len(sample_radii))**2, color = 'r'))
        if not ('ap_nologo' in options and options['ap_nologo']):
            AddLogo(plt.gcf())
        plt.savefig('%sfit_ellipse_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
        plt.close()
        
        plt.scatter(sample_radii, ellip, color = 'r', label = 'ellip')
        plt.scatter(sample_radii, pa/np.pi, color = 'b', label = 'pa/$np.pi$')
        show_ellip = _ellip_smooth(sample_radii, ellip, deg = 5)
        show_pa = _pa_smooth(sample_radii, pa, deg = 5)
        plt.plot(sample_radii, show_ellip, color = 'orange', linewidth = 2, linestyle='--', label = 'smooth ellip')
        plt.plot(sample_radii, show_pa/np.pi, color = 'purple', linewidth = 2, linestyle='--', label = 'smooth pa/$np.pi$')
        #plt.xscale('log')
        plt.legend()
        if not ('ap_nologo' in options and options['ap_nologo']):
            AddLogo(plt.gcf())
        plt.savefig('%sphaseprofile_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
        plt.close()

    # Compute errors
    ######################################################################
    ellip_err = np.zeros(len(ellip))
    ellip_err[:2] = np.sqrt(np.sum((ellip[:4] - smooth_ellip[:4])**2)/4)
    ellip_err[-1] = np.sqrt(np.sum((ellip[-4:] - smooth_ellip[-4:])**2)/4)
    pa_err = np.zeros(len(pa))
    pa_err[:2] = np.sqrt(np.sum((pa[:4] - smooth_pa[:4])**2)/4)
    pa_err[-1] = np.sqrt(np.sum((pa[-4:] - smooth_pa[-4:])**2)/4)
    for i in range(2,len(pa)-1):
        ellip_err[i] = np.sqrt(np.sum((ellip[i-2:i+2] - smooth_ellip[i-2:i+2])**2)/4)
        pa_err[i] = np.sqrt(np.sum((pa[i-2:i+2] - smooth_pa[i-2:i+2])**2)/4)

    res = {'fit ellip': ellip, 'fit pa': pa, 'fit R': sample_radii,
           'fit ellip_err': ellip_err, 'fit pa_err': pa_err,
           'auxfile fitlimit': 'fit limit semi-major axis: %.2f pix' % sample_radii[-1]}
    return IMG, res
