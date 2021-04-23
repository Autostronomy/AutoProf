import numpy as np
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _iso_extract, AddLogo
from photutils.centroids import centroid_2dg, centroid_com, centroid_1dg
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy.fftpack import fft, ifft
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import logging
from copy import copy, deepcopy

def Center_Forced(IMG, results, options):
    """
    Takes the center from an aux file, or given value.
    """
    current_center = {'x': IMG.shape[1]/2, 'y': IMG.shape[0]/2}
    if 'ap_guess_center' in options:
        current_center = deepcopy(options['ap_guess_center'])
        logging.info('%s: Center initialized by user: %s' % (options['ap_name'], str(current_center)))
    if 'ap_set_center' in options:
        logging.info('%s: Center set by user: %s' % (options['ap_name'], str(options['ap_set_center'])))
        return IMG, {'center': deepcopy(options['ap_set_center'])}

    try:
        with open(options['ap_forcing_profile'][:-4] + 'aux', 'r') as f:
            for line in f.readlines():
                if line[:6] == 'center':
                    x_loc = line.find('x:')
                    y_loc = line.find('y:')
                    current_center = {'x': float(line[x_loc+3:line.find('pix')]),
                                      'y': float(line[y_loc+3:line.rfind('pix')])}
                    break
            else:
                logging.warning('%s: Forced center failed! Using image center (or guess).' % options['ap_name'])
    except:
        logging.warning('%s: Forced center failed! Using image center (or guess).' % options['ap_name'])        
    return IMG, {'center': current_center, 'auxfile center': 'center x: %.2f pix, y: %.2f pix' % (current_center['x'], current_center['y'])}
    
def Center_2DGaussian(IMG, results, options):
    """
    Compute the pixel location of the galaxy center by fitting
    a 2d Gaussian as implimented by the photutils package.
    """
    
    current_center = {'x': IMG.shape[1]/2, 'y': IMG.shape[0]/2}
    if 'ap_guess_center' in options:
        current_center = deepcopy(options['ap_guess_center'])
        logging.info('%s: Center initialized by user: %s' % (options['ap_name'], str(current_center)))
    if 'ap_set_center' in options:
        logging.info('%s: Center set by user: %s' % (options['ap_name'], str(options['ap_set_center'])))
        return IMG, {'center': deepcopy(options['ap_set_center'])}
    
    # Create mask to focus centering algorithm on the center of the image
    ranges = [[max(0,int(current_center['x'] - 50*results['psf fwhm'])), min(IMG.shape[1],int(current_center['x'] + 50*results['psf fwhm']))],
              [max(0,int(current_center['y'] - 50*results['psf fwhm'])), min(IMG.shape[0],int(current_center['y'] + 50*results['psf fwhm']))]]
    centralize_mask = np.ones(IMG.shape, dtype = bool)
    centralize_mask[ranges[1][0]:ranges[1][1],
                    ranges[0][0]:ranges[0][1]] = False

    try:
        x, y = centroid_2dg(IMG - results['background'], mask = centralize_mask)
        current_center = {'x': x, 'y': y}
    except:
        logging.warning('%s: 2D Gaussian center finding failed! using image center (or guess).' % options['ap_name'])
        
    # Plot center value for diagnostic purposes
    if 'ap_doplot' in options and options['ap_doplot']:    
        plt.imshow(np.clip(IMG - results['background'],a_min = 0, a_max = None),
                   origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        plt.plot([current_center['x']],[current_center['y']], marker = 'x', markersize = 10, color = 'y')
        plt.savefig('%scenter_vis_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']))
        plt.close()
    logging.info('%s Center found: x %.1f, y %.1f' % (options['ap_name'], x, y))    
    return IMG, {'center': current_center, 'auxfile center': 'center x: %.2f pix, y: %.2f pix' % (current_center['x'], current_center['y'])}

def Center_1DGaussian(IMG, results, options):
    """
    Compute the pixel location of the galaxy center using a photutils method.
    Looking at 100 seeing lengths around the center of the image (images
    should already be mostly centered), finds the galaxy center by fitting
    several 1d Gaussians.
    """
    
    current_center = {'x': IMG.shape[1]/2, 'y': IMG.shape[0]/2}
    if 'ap_guess_center' in options:
        current_center = deepcopy(options['ap_guess_center'])
        logging.info('%s: Center initialized by user: %s' % (options['ap_name'], str(current_center)))
    if 'ap_set_center' in options:
        logging.info('%s: Center set by user: %s' % (options['ap_name'], str(options['ap_set_center'])))
        return IMG, {'center': deepcopy(options['ap_set_center'])}
    
    # Create mask to focus centering algorithm on the center of the image
    ranges = [[max(0,int(current_center['x'] - 50*results['psf fwhm'])), min(IMG.shape[1],int(current_center['x'] + 50*results['psf fwhm']))],
              [max(0,int(current_center['y'] - 50*results['psf fwhm'])), min(IMG.shape[0],int(current_center['y'] + 50*results['psf fwhm']))]]
    centralize_mask = np.ones(IMG.shape, dtype = bool)
    centralize_mask[ranges[1][0]:ranges[1][1],
                    ranges[0][0]:ranges[0][1]] = False
    
    try:
        x, y = centroid_1dg(IMG - results['background'],
                            mask = centralize_mask) 
        current_center = {'x': x, 'y': y}
    except:
        logging.warning('%s: 2D Gaussian center finding failed! using image center (or guess).' % options['ap_name'])
    
    # Plot center value for diagnostic purposes
    if 'ap_doplot' in options and options['ap_doplot']:    
        plt.imshow(np.clip(IMG - results['background'],a_min = 0, a_max = None),
                   origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        plt.plot([y],[x], marker = 'x', markersize = 10, color = 'y')
        plt.savefig('%scenter_vis_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']))
        plt.close()
    logging.info('%s Center found: x %.1f, y %.1f' % (options['ap_name'], x, y))    
    return IMG, {'center': current_center, 'auxfile center': 'center x: %.2f pix, y: %.2f pix' % (current_center['x'], current_center['y'])}

def Center_OfMass(IMG, results, options):
    """
    Compute the pixel location of the galaxy center using a light weighted
    center of mass. Looking at 50 seeing lengths around the center of the
    image (images should already be mostly centered), finds the average
    light weighted center of the image.
    """
    
    current_center = {'x': IMG.shape[1]/2, 'y': IMG.shape[0]/2}
    if 'ap_guess_center' in options:
        current_center = deepcopy(options['ap_guess_center'])
        logging.info('%s: Center initialized by user: %s' % (options['ap_name'], str(current_center)))
    if 'ap_set_center' in options:
        logging.info('%s: Center set by user: %s' % (options['ap_name'], str(options['ap_set_center'])))
        return IMG, {'center': deepcopy(options['ap_set_center'])}
    
    # Create mask to focus centering algorithm on the center of the image
    ranges = [[max(0,int(current_center['x'] - 50*results['psf fwhm'])), min(IMG.shape[1],int(current_center['x'] + 50*results['psf fwhm']))],
              [max(0,int(current_center['y'] - 50*results['psf fwhm'])), min(IMG.shape[0],int(current_center['y'] + 50*results['psf fwhm']))]]
    centralize_mask = np.ones(IMG.shape, dtype = bool)
    centralize_mask[ranges[1][0]:ranges[1][1],
                    ranges[0][0]:ranges[0][1]] = False
    
    try:
        x, y = centroid_com(IMG - results['background'],
                            mask = centralize_mask)
        current_center = {'x': x, 'y': y}
    except:
        logging.warning('%s: 2D Gaussian center finding failed! using image center (or guess).' % options['ap_name'])
    
    # Plot center value for diagnostic purposes
    if 'ap_doplot' in options and options['ap_doplot']:    
        plt.imshow(np.clip(IMG - results['background'],a_min = 0, a_max = None),
                   origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        plt.plot([y],[x], marker = 'x', markersize = 10, color = 'y')
        plt.savefig('%scenter_vis_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']))
        plt.close()
    logging.info('%s Center found: x %.1f, y %.1f' % (options['ap_name'], x, y))    
    return IMG, {'center': current_center, 'auxfile center': 'center x: %.2f pix, y: %.2f pix' % (current_center['x'], current_center['y'])}

def _hillclimb_loss(x, IMG, PSF, noise):
    center_loss = 0
    for rr in range(3):
        isovals = _iso_extract(IMG,(rr+0.5)*PSF,0.,
                               0.,{'x': x[0], 'y': x[1]}, more = False, rad_interp = 10*PSF)
        coefs = fft(isovals)
        center_loss += np.abs(coefs[1])/(len(isovals)*(max(0,np.median(isovals)) + noise))
    return center_loss

def Center_HillClimb(IMG, results, options):
    """
    Using 10 circular isophotes out to 10 times the PSF length, the first FFT coefficient
    phases are averaged to find the direction of increasing flux. Flux values are sampled
    along this direction and a quadratic fit gives the maximum. This is iteratively
    repeated until the step size becomes very small.
    """
    
    current_center = {'x': IMG.shape[1]/2, 'y': IMG.shape[0]/2}
    if 'ap_guess_center' in options:
        current_center = deepcopy(options['ap_guess_center'])
        logging.info('%s: Center initialized by user: %s' % (options['ap_name'], str(current_center)))
    if 'ap_set_center' in options:
        logging.info('%s: Center set by user: %s' % (options['ap_name'], str(options['ap_set_center'])))
        return IMG, {'center': deepcopy(options['ap_set_center'])}

    dat = IMG - results['background']

    sampleradii = np.linspace(1,10,10) * results['psf fwhm']

    track_centers = []
    small_update_count = 0
    total_count = 0
    while small_update_count <= 5 and total_count <= 100:
        total_count += 1
        phases = []
        isovals = []
        coefs = []
        for r in sampleradii:
            isovals.append(_iso_extract(dat,r,0.,0.,current_center, more = True))
            coefs.append(fft(np.clip(isovals[-1][0], a_max = np.quantile(isovals[-1][0],0.85), a_min = None)))
            phases.append((-np.angle(coefs[-1][1])) % (2*np.pi))
        complexphase = np.array(np.cos(phases) + np.sin(phases)*1j,dtype = np.complex_)
        direction = np.angle(np.mean(complexphase)) % (2*np.pi) 
        levels = []
        level_locs = []
        for i, r in enumerate(sampleradii):
            floc = np.argmin(np.abs(isovals[i][1] - direction))
            rloc = np.argmin(np.abs(isovals[i][1] - ((direction+np.pi) % (2*np.pi))))
            smooth = np.abs(ifft(coefs[i][:min(10,len(coefs[i]))],n = len(coefs[i])))
            if smooth[floc] > (3*results['background noise']):
                levels.append(smooth[floc])
                level_locs.append(r)
            if smooth[rloc] > (3*results['background noise']):
                levels.insert(0,smooth[rloc])
                level_locs.insert(0,-r)
        try:
            p = np.polyfit(level_locs, levels, deg = 2)
            if p[0] < 0 and len(levels) > 3:
                dist = np.clip(-p[1]/(2*p[0]), a_min = min(level_locs), a_max = max(level_locs))
            else:
                dist = level_locs[np.argmax(levels)]
        except:
            dist = 1.
        current_center['x'] += dist*np.cos(direction)
        current_center['y'] += dist*np.sin(direction)
        if abs(dist) < (0.25*results['psf fwhm']):
            small_update_count += 1
        else:
            small_update_count = 0
        track_centers.append([current_center['x'], current_center['y']])

    # refine center
    res = minimize(_hillclimb_loss, x0 =  [current_center['x'], current_center['y']], args = (dat, results['psf fwhm'], results['background noise']), method = 'Nelder-Mead')
    if res.success:
        current_center['x'] = res.x[0]
        current_center['y'] = res.x[1]

    # paper plot
    if 'ap_paperplots' in options and options['ap_paperplots']:    
        plt.imshow(np.clip(dat,a_min = 0, a_max = None), origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        plt.plot([current_center['x']],[current_center['y']], marker = 'x', markersize = 3, color = 'r')
        for i in range(3):
            plt.gca().add_patch(Ellipse((current_center['x'],current_center['y']), 2*((i+0.5)*results['psf fwhm']),
                                        2*((i+0.5)*results['psf fwhm']),
                                        0, fill = False, linewidth = 0.5, color = 'y'))
        plt.tight_layout()
        if not ('ap_nologo' in options and options['ap_nologo']):
            AddLogo(plt.gcf())
        plt.savefig('%stest_center_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
        plt.close()

        xwidth = 2*max(np.abs(track_centers[:,0] - current_center['x']))
        ywidth = 2*max(np.abs(track_centers[:,1] - current_center['y']))
        width = max(xwidth, ywidth)
        ranges = [[int(current_center['x'] - width), int(current_center['x'] + width)],
                  [int(current_center['y'] - width), int(current_center['y'] + width)]]
        fig, axarr = plt.subplots(2,1, figsize = (3,6))
        plt.subplots_adjust(hspace = 0.01, wspace = 0.01, left = 0.05, right = 0.95, top = 0.95, bottom = 0.05)
        axarr[0].imshow(np.clip(dat[ranges[1][0]:ranges[1][1],ranges[0][0]:ranges[0][1]],a_min = 0, a_max = None),
                        origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()),
                        extent = [ranges[0][0],ranges[0][1],ranges[1][0]-1,ranges[1][1]-1])
        axarr[0].plot(track_centers[:,0], track_centers[:,1], color = 'y')
        axarr[0].scatter(track_centers[:,0], track_centers[:,1], c = range(len(track_centers)), cmap = 'Reds')
        axarr[0].set_xticks([])
        axarr[0].set_yticks([])        
        width = 10.
        ranges = [[int(current_center['x'] - width), int(current_center['x'] + width)],
                  [int(current_center['y'] - width), int(current_center['y'] + width)]]
        axarr[1].imshow(np.clip(dat[ranges[1][0]:ranges[1][1],ranges[0][0]:ranges[0][1]],a_min = 0, a_max = None),
                        origin = 'lower', cmap = 'Greys_r',
                        extent = [ranges[0][0],ranges[0][1],ranges[1][0]-1,ranges[1][1]-1])
        axarr[1].plot(track_centers[:,0], track_centers[:,1], color = 'y')
        axarr[1].scatter(track_centers[:,0], track_centers[:,1], c = range(len(track_centers)), cmap = 'Reds')
        axarr[1].set_xlim(ranges[0])
        axarr[1].set_ylim(np.array(ranges[1])-1)        
        axarr[1].set_xticks([])
        axarr[1].set_yticks([])
        plt.savefig('%sCenter_path_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi' in options else 300)
        plt.close()
        
    return IMG, {'center': current_center, 'auxfile center': 'center x: %.2f pix, y: %.2f pix' % (current_center['x'], current_center['y'])}


def _hillclimb_mean_loss(x, IMG, PSF, noise):
    center_loss = 0
    for rr in range(3):
        isovals = _iso_extract(IMG,(rr+0.5)*PSF,0.,
                               0.,{'x': x[0], 'y': x[1]}, more = False, rad_interp = 10*PSF)
        coefs = fft(isovals)
        center_loss += np.abs(coefs[1])/(len(isovals)*(max(0,np.mean(isovals))+noise)) 
    return center_loss

def Center_HillClimb_mean(IMG, results, options):
    """
    Using 10 circular isophotes out to 10 times the PSF length, the first FFT coefficient
    phases are averaged to find the direction of increasing flux. Flux values are sampled
    along this direction and a quadratic fit gives the maximum. This is iteratively
    repeated until the step size becomes very small.
    """
    current_center = {'x': IMG.shape[0]/2, 'y': IMG.shape[1]/2}

    current_center = {'x': IMG.shape[1]/2, 'y': IMG.shape[0]/2}
    if 'ap_guess_center' in options:
        current_center = deepcopy(options['ap_guess_center'])
        logging.info('%s: Center initialized by user: %s' % (options['ap_name'], str(current_center)))
    if 'ap_set_center' in options:
        logging.info('%s: Center set by user: %s' % (options['ap_name'], str(options['ap_set_center'])))
        return IMG, {'center': deepcopy(options['ap_set_center'])}

    dat = IMG - results['background']

    sampleradii = np.linspace(1,10,10) * results['psf fwhm']

    track_centers = []
    small_update_count = 0
    total_count = 0
    while small_update_count <= 5 and total_count <= 100:
        total_count += 1
        phases = []
        isovals = []
        coefs = []
        for r in sampleradii:
            isovals.append(_iso_extract(dat,r,0.,0.,current_center, more = True))
            coefs.append(fft(isovals[-1][0]))
            phases.append((-np.angle(coefs[-1][1])) % (2*np.pi))
        complexphase = np.array(np.cos(phases) + np.sin(phases)*1j,dtype = np.complex_)
        direction = np.angle(np.mean(complexphase)) % (2*np.pi)  # fixme angle average
        levels = []
        level_locs = []
        for i, r in enumerate(sampleradii):
            floc = np.argmin(np.abs(isovals[i][1] - direction))
            rloc = np.argmin(np.abs(isovals[i][1] - ((direction+np.pi) % (2*np.pi))))
            smooth = np.abs(ifft(coefs[i][:min(10,len(coefs[i]))],n = len(coefs[i])))
            if smooth[floc] > (3*results['background noise']):
                levels.append(smooth[floc])
                level_locs.append(r)
            if smooth[rloc] > (3*results['background noise']):
                levels.insert(0,smooth[rloc])
                level_locs.insert(0,-r)
        try:
            p = np.polyfit(level_locs, levels, deg = 2)
            if p[0] < 0 and len(levels) > 3:
                dist = np.clip(-p[1]/(2*p[0]), a_min = min(level_locs), a_max = max(level_locs))
            else:
                dist = level_locs[np.argmax(levels)]
        except:
            dist = 1.
        current_center['x'] += dist*np.cos(direction)
        current_center['y'] += dist*np.sin(direction)
        if abs(dist) < (0.25*results['psf fwhm']):
            small_update_count += 1
        else:
            small_update_count = 0
        track_centers.append([current_center['x'], current_center['y']])

    # refine center
    res = minimize(_hillclimb_loss, x0 =  [current_center['x'], current_center['y']], args = (dat, results['psf fwhm'], results['background noise']), method = 'Nelder-Mead')
    if res.success:
        current_center['x'] = res.x[0]
        current_center['y'] = res.x[1]
        
    return IMG, {'center': current_center, 'auxfile center': 'center x: %.2f pix, y: %.2f pix' % (current_center['x'], current_center['y'])}
