import numpy as np
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _iso_between, LSBImage, _iso_line, AddLogo, autocmap, Sigma_Clip_Upper
from scipy.stats import iqr
import matplotlib.pyplot as plt
import logging

def Slice_Profile(IMG, results, options):

    dat = IMG - (results['background'] if 'background' in results else 0.)
    zeropoint = options['ap_zeropoint'] if 'ap_zeropoint' in options else 22.5
    
    use_anchor = results['center'] if 'center' in results else {'x': IMG.shape[1]/2, 'y': IMG.shape[0]/2}
    if 'ap_slice_anchor' in options:
        use_anchor = options['ap_slice_anchor']
    else:
        logging.warning('%s: ap_slice_anchor not specified by user, using: %s' % (options['ap_name'], str(use_anchor)))

    use_pa = results['init pa'] if 'init pa' in results else 0.
    if 'ap_slice_pa' in options:
        use_pa = options['ap_slice_pa']*np.pi/180
    else:
        logging.warning('%s: ap_slice_pa not specified by user, using: %.2f' % (options['ap_name'], use_pa))

    use_length = results['init R'] if 'init R' in results else min(IMG.shape)
    if 'ap_slice_length' in options:
        use_length = options['ap_slice_length']
    else:
        logging.warning('%s: ap_slice_length not specified by user, using: %.2f' % (options['ap_name'], use_length))

    use_width = 10.
    if 'ap_slice_width' in options:
        use_width = options['ap_slice_width']
    else:
        logging.warning('%s: ap_slice_width not specified by user, using: %.2f' % (options['ap_name'], use_width))

    use_step = results['psf fwhm'] if 'psf fwhm' in results else max(2., use_length/100)
    if 'ap_slice_step' in options:
        use_step = options['ap_slice_step']
    else:
        logging.warning('%s: ap_slice_step not specified by user, using: %.2f' % (options['ap_name'], use_step))
    
    F, X = _iso_line(dat, use_length, use_width, use_pa, use_anchor, more = False)

    windows = np.arange(0, use_length, use_step)

    R = (windows[1:] + windows[:-1])/2
    sb = []
    sb_e = []
    sb_sclip = []
    sb_sclip_e = []
    for i in range(len(windows)-1):
        isovals = F[np.logical_and(X >= windows[i], X < windows[i+1])]
        isovals_sclip = Sigma_Clip_Upper(isovals, iterations = 10, nsigma = 5)

        medflux = np.median(isovals)
        medflux_sclip = np.median(isovals_sclip)

        sb.append((-2.5*np.log10(medflux) + zeropoint + 5*np.log10(options['ap_pixscale'])) if medflux > 0 else 99.999)
        sb_e.append((2.5*iqr(isovals, rng = (31.731/2, 100 - 31.731/2)) / (2*np.sqrt(len(isovals))*medflux*np.log(10))) if medflux > 0 else 99.999)
        sb_sclip.append((-2.5*np.log10(medflux_sclip) + zeropoint + 5*np.log10(options['ap_pixscale'])) if medflux_sclip > 0 else 99.999)
        sb_sclip_e.append((2.5*iqr(isovals_sclip, rng = (31.731/2, 100 - 31.731/2)) / (2*np.sqrt(len(isovals))*medflux_sclip*np.log(10))) if medflux_sclip > 0 else 99.999)

    
    with open('%s%s_slice_profile_AP.prof' % ((options['ap_saveto'] if 'ap_saveto' in options else ''), options['ap_name']), 'w') as f:
        f.write('# flux sum: %f\n' % (np.sum(F[np.logical_and(X >= 0, X <= use_length)])))
        f.write('# flux mean: %f\n' % (np.mean(F[np.logical_and(X >= 0, X <= use_length)])))
        f.write('# flux std: %f\n' % (np.std(F[np.logical_and(X >= 0, X <= use_length)])))
        f.write('# flux median: %f\n' % (np.median(F[np.logical_and(X >= 0, X <= use_length)])))
        f.write('# flux 16-84%% range: %f\n' % (iqr(F[np.logical_and(X >= 0, X <= use_length)], rng = [16,84])))
        f.write('R,sb,sb_e,sb_sclip,sb_sclip_e\n')
        f.write('arcsec,mag*arcsec^-2,mag*arcsec^-2,mag*arcsec^-2,mag*arcsec^-2\n')
        for i in range(len(R)):
            f.write('%.4f,%.4f,%.4f,%.4f,%.4f\n' % (R[i]*options['ap_pixscale'], sb[i], sb_e[i], sb_sclip[i], sb_sclip_e[i]))

    if 'ap_doplot' in options and options['ap_doplot']:
        CHOOSE = np.array(sb_e) < 0.5
        plt.errorbar(np.array(R)[CHOOSE]*options['ap_pixscale'], np.array(sb)[CHOOSE], yerr = np.array(sb_e)[CHOOSE],
                     elinewidth = 1, linewidth = 0, marker = '.', markersize = 3, color = 'r')
        plt.xlabel('Position on line [arcsec]', fontsize = 16)
        plt.ylabel('Surface Brightness [mag arcsec$^{-2}$]', fontsize = 16)
        if 'background noise' in results:
            bkgrdnoise = -2.5*np.log10(results['background noise']) + zeropoint + 2.5*np.log10(options['ap_pixscale']**2)
            plt.axhline(bkgrdnoise, color = 'purple', linewidth = 0.5, linestyle = '--', label = '1$\\sigma$ noise/pixel: %.1f mag arcsec$^{-2}$' % bkgrdnoise)
        plt.gca().invert_yaxis()
        plt.legend(fontsize = 15)
        plt.tick_params(labelsize = 14)
        plt.tight_layout()
        if not ('ap_nologo' in options and options['ap_nologo']):
            AddLogo(plt.gcf())
        plt.savefig('%sslice_profile_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
        plt.close()
        
        ranges = [[max(0,int(use_anchor['x']+0.5*use_length*np.cos(use_pa)-use_length*0.7)), min(IMG.shape[1],int(use_anchor['x']+0.5*use_length*np.cos(use_pa)+use_length*0.7))],
                  [max(0,int(use_anchor['y']+0.5*use_length*np.sin(use_pa)-use_length*0.7)), min(IMG.shape[0],int(use_anchor['y']+0.5*use_length*np.sin(use_pa)+use_length*0.7))]]
        LSBImage(dat[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]], results['background noise'] if 'background noise' in results else iqr(dat, rng = (31.731/2, 100 - 31.731/2))/2)
        
        XX, YY = np.meshgrid(np.arange(ranges[0][1] - ranges[0][0], dtype = float), np.arange(ranges[1][1] - ranges[1][0], dtype = float))
        XX -= use_anchor['x'] - float(ranges[0][0])
        YY -= use_anchor['y'] - float(ranges[1][0])
        XX, YY = (XX*np.cos(-use_pa) - YY*np.sin(-use_pa), XX*np.sin(-use_pa) + YY*np.cos(-use_pa))
        ZZ = np.ones(XX.shape)
        ZZ[np.logical_not(np.logical_and(np.logical_and(YY <= use_width/2, YY >= -use_width/2),
                                         np.logical_and(XX >= 0, XX <= use_length)))] = np.nan
        plt.imshow(ZZ, origin = 'lower', cmap = 'Reds_r', alpha = 0.6)
        plt.tight_layout()
        if not ('ap_nologo' in options and options['ap_nologo']):
            AddLogo(plt.gcf())
        plt.savefig('%sslice_profile_window_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
        plt.close()
        
    return IMG, {}
