import numpy as np
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _x_to_pa, _x_to_eps, _inv_x_to_eps, _inv_x_to_pa, SBprof_to_COG_errorprop, _iso_extract, _iso_between, LSBImage, AddLogo, _average, _scatter, flux_to_sb, flux_to_mag, PA_shift_convention, autocolours, fluxdens_to_fluxsum_errorprop, mag_to_flux

def Plot_Background(values, bkgrnd, noise, results, options):

    hist, bins = np.histogram(values[np.logical_and((values-bkgrnd) < 20*noise, (values-bkgrnd) > -5*noise)], bins = max(10,int(np.sqrt(len(values))/2)))
    plt.figure(figsize = (5,5))
    plt.bar(bins[:-1], np.log10(hist), width = bins[1] - bins[0], color = 'k', label = 'pixel values')
    plt.axvline(bkgrnd, color = '#84DCCF', label = 'sky level: %.5e' % bkgrnd)
    plt.axvline(bkgrnd - noise, color = '#84DCCF', linewidth = 0.7, linestyle = '--', label = '1$\\sigma$ noise/pix: %.5e' % noise)
    plt.axvline(bkgrnd + noise, color = '#84DCCF', linewidth = 0.7, linestyle = '--')
    plt.xlim([bkgrnd-5*noise, bkgrnd+20*noise])
    plt.legend(fontsize = 12)
    plt.tick_params(labelsize = 12)
    plt.xlabel('Pixel Flux', fontsize = 16)
    plt.ylabel('log$_{10}$(count)', fontsize = 16)
    plt.tight_layout()
    if not ('ap_nologo' in options and options['ap_nologo']):
        AddLogo(plt.gcf())
    plt.savefig('%sBackground_hist_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
    plt.close()        

def Plot_PSF_Stars(IMG, stars_x, stars_y, stars_fwhm, psf, results, options, flagstars = None):
    LSBImage(IMG - results['background'], results['background noise'])
    for i in range(len(stars_fwhm)):
        plt.gca().add_patch(Ellipse((stars_x[i],stars_y[i]), 20*psf, 20*psf,
                                    0, fill = False, linewidth = 1.5, color = autocolours['red1'] if not flagstars is None and flagstars[i] else autocolours['blue1']))
    plt.tight_layout()
    if not ('ap_nologo' in options and options['ap_nologo']):
        AddLogo(plt.gcf())
    plt.savefig('%sPSF_Stars_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
    plt.close()
    

def Plot_SB_Profile(dat, R, SB, SB_e, ellip, pa, results, options):

    zeropoint = options['ap_zeropoint'] if 'ap_zeropoint' in options else 22.5
    
    CHOOSE = np.logical_and(SB < 99, SB_e < 1)
    if np.sum(CHOOSE) < 5:
        CHOOSE = np.ones(len(CHOOSE), dtype = bool)
    errscale = 1.
    if np.all(SB_e[CHOOSE] < 0.5):
        errscale = 1/np.max(SB_e[CHOOSE])
    lnlist = []
    lnlist.append(plt.errorbar(R[CHOOSE], SB[CHOOSE], yerr = errscale*SB_e[CHOOSE],
                               elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = autocolours['red1'], label = 'Surface Brightness (err$\\cdot$%.1f)' % errscale))
    plt.errorbar(R[np.logical_and(CHOOSE,np.arange(len(CHOOSE)) % 4 == 0)],
                 SB[np.logical_and(CHOOSE,np.arange(len(CHOOSE)) % 4 == 0)],
                 yerr = SB_e[np.logical_and(CHOOSE,np.arange(len(CHOOSE)) % 4 == 0)],
                 elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = autocolours['blue1'])
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

    useR = R[CHOOSE]/options['ap_pixscale']
    useE = np.array(ellip)[CHOOSE]
    usePA = np.array(pa)[CHOOSE]
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


def Plot_I_Profile(dat, R, I, I_e, ellip, pa, results, options):

    CHOOSE = np.isfinite(I)
    if np.sum(CHOOSE) < 5:
        CHOOSE = np.ones(len(CHOOSE), dtype = bool)
    errscale = 1.
    lnlist = []
    lnlist.append(plt.errorbar(R[CHOOSE], I[CHOOSE], yerr = errscale*I_e[CHOOSE],
                               elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = autocolours['red1'], label = 'Intensity (err$\\cdot$%.1f)' % errscale))
    plt.errorbar(R[np.logical_and(CHOOSE,np.arange(len(CHOOSE)) % 4 == 0)],
                 I[np.logical_and(CHOOSE,np.arange(len(CHOOSE)) % 4 == 0)],
                 yerr = I_e[np.logical_and(CHOOSE,np.arange(len(CHOOSE)) % 4 == 0)],
                 elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = autocolours['blue1'])
    plt.xlabel('Semi-Major-Axis [arcsec]', fontsize = 16)
    plt.ylabel('Intensity [flux arcsec$^{-2}$]', fontsize = 16)
    plt.yscale('log')
    plt.xlim([0,None])
    bkgrdnoise = results['background noise'] / (options['ap_pixscale']**2)
    lnlist.append(plt.axhline(bkgrdnoise, color = 'purple', linewidth = 0.5, linestyle = '--', label = '1$\\sigma$ noise/pixel: %.1f flux arcsec$^{-2}$' % bkgrdnoise))
    plt.tick_params(labelsize = 14)
    labs = [l.get_label() for l in lnlist]
    plt.legend(lnlist, labs, fontsize = 11)
    plt.tight_layout()
    if not ('ap_nologo' in options and options['ap_nologo']):
        AddLogo(plt.gcf())
    plt.savefig('%sphotometry_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
    plt.close()                

    useR = R[CHOOSE]/options['ap_pixscale']
    useE = np.array(ellip)[CHOOSE]
    usePA = np.array(pa)[CHOOSE]
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

