import numpy as np
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _iso_extract, _iso_between, Angle_TwoAngles, LSBImage, _iso_line, AddLogo, autocmap
from scipy.stats import iqr
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
import matplotlib
import logging

def Orthogonal_Sample(IMG, results, **kwargs):

    mask = results['mask'] if 'mask' in results else None
    pa = (kwargs['orthsample_pa']*np.pi/180) if 'orthsample_pa' in kwargs else results['init pa']
    if 'orthsample_parallel' in kwargs and kwargs['orthsample_parallel']:
        pa += np.pi/2
    dat = IMG - results['background']
    zeropoint = kwargs['zeropoint'] if 'zeropoint' in kwargs else 22.5

    if 'prof data' in results:
        Rproflim = results['prof data']['R'][-1]/kwargs['pixscale']
    else:
        Rproflim = min(IMG.shape)/2
    
    R = [0]
    while R[-1] < Rproflim:
        if 'samplestyle' in kwargs and kwargs['samplestyle'] == 'linear':
            step = kwargs['samplelinearscale'] if 'samplelinearscale' in kwargs else 0.5*results['psf fwhm']
        else:
            step = R[-1]*(kwargs['samplegeometricscale'] if 'samplegeometricscale' in kwargs else 0.1)
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
                    medflux = np.median(flux[CHOOSE])
                    sb[key][-1].append((-2.5*np.log10(medflux) + zeropoint + 5*np.log10(kwargs['pixscale'])) if medflux > 0 else 99.999)
                    sbE[key][-1].append((2.5*iqr(flux[CHOOSE], rng = (31.731/2, 100 - 31.731/2)) / (2*np.sqrt(np.sum(CHOOSE))*medflux*np.log(10))) if medflux > 0 else 99.999)
                    

    with open('%s%s_orthogonal_sample.prof' % ((kwargs['saveto'] if 'saveto' in kwargs else ''), kwargs['name']), 'w') as f:
        f.write('R')
        for rd in [1,-1]:
            for ang in [1, -1]:
                for pR in R:
                    f.write(',sb[%.3f:%s90],sbE[%.3f:%s90]' % (rd*pR*kwargs['pixscale'], '+' if ang > 0 else '-', rd*pR*kwargs['pixscale'], '+' if ang > 0 else '-'))
        f.write('\n')
        f.write('arcsec')
        for rd in [1,-1]:
            for ang in [1, -1]:
                for pR in R:
                    f.write(',mag*arcsec^-2,mag*arcsec^-2')
        f.write('\n')
        for oi, oR in enumerate(R):
            f.write('%.4f' % (oR*kwargs['pixscale']))
            for rd in [1,-1]:
                for ang in [1, -1]:
                    key = (rd,ang)
                    for pi, pR in enumerate(R):
                        f.write(',%.4f,%.4f' % (sb[key][pi][oi], sbE[key][pi][oi]))
            f.write('\n')

    if 'doplot' in kwargs and kwargs['doplot']:

        #colours = {(1,1): 'cool', (1,-1): 'summer', (-1,1): 'autumn', (-1,-1): 'winter'}
        count = 0
        for rd in [1,-1]:
            for ang in [1, -1]:
                key = (rd,ang)
                #cmap = matplotlib.cm.get_cmap('viridis_r')
                norm = matplotlib.colors.Normalize(vmin=0, vmax=R[-1]*kwargs['pixscale'])
                for pi, pR in enumerate(R):
                    if pi % 3 != 0:
                        continue
                    CHOOSE = np.logical_and(np.array(sb[key][pi]) < 99, np.array(sbE[key][pi]) < 1)
                    plt.errorbar(np.array(R)[CHOOSE]*kwargs['pixscale'], np.array(sb[key][pi])[CHOOSE], yerr = np.array(sbE[key][pi])[CHOOSE],
                                 elinewidth = 1, linewidth = 0, marker = '.', markersize = 3, color = autocmap.reversed()(norm(pR*kwargs['pixscale'])))
                plt.xlabel('%s-axis position on line [arcsec]' % ('Major' if 'orthsample_parallel' in kwargs and kwargs['orthsample_parallel'] else 'Minor'))
                plt.ylabel('Surface Brightness [mag/arcsec^2]')
                # cb1 = matplotlib.colorbar.ColorbarBase(plt.gca(), cmap=cmap,
                #                                        norm=norm)
                cb1 = plt.colorbar(matplotlib.cm.ScalarMappable(norm = norm, cmap = autocmap.reversed()))
                cb1.set_label('%s-axis position of line [arcsec]'  % ('Minor' if 'orthsample_parallel' in kwargs and kwargs['orthsample_parallel'] else 'Major'))
                # plt.colorbar()
                bkgrdnoise = -2.5*np.log10(results['background noise']) + zeropoint + 2.5*np.log10(kwargs['pixscale']**2)
                plt.axhline(bkgrdnoise, color = 'purple', linewidth = 0.5, linestyle = '--', label = '1$\\sigma$ noise/pixel: %.1f mag arcsec$^{-2}$' % bkgrdnoise)
                plt.gca().invert_yaxis()
                plt.legend(fontsize = 10)
                plt.title('%sR : pa%s90' % ('+' if rd > 0 else '-', '+' if ang > 0 else '-'))
                plt.tight_layout()
                if not ('nologo' in kwargs and kwargs['nologo']):
                    AddLogo(plt.gcf())
                plt.savefig('%sorthogonal_sample_q%i_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', count, kwargs['name']), dpi = kwargs['plotdpi'] if 'plotdpi'in kwargs else 300)
                plt.close()
                count += 1


        CHOOSE = np.array(results['prof data']['SB_e']) < 0.2
        firstbad = np.argmax(np.logical_not(CHOOSE))
        if firstbad > 3:
            CHOOSE[firstbad:] = False
        outto = np.array(results['prof data']['R'])[CHOOSE][-1]*1.5/kwargs['pixscale']
        ranges = [[max(0,int(results['center']['x']-outto-2)), min(IMG.shape[1],int(results['center']['x']+outto+2))],
                  [max(0,int(results['center']['y']-outto-2)), min(IMG.shape[0],int(results['center']['y']+outto+2))]]
        LSBImage(dat[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]], results['background noise'])
        count = 0
        cmap = matplotlib.cm.get_cmap('hsv')
        colorind = (np.linspace(0,1 - 1/4,4) + 0.1) % 1
        colours = list(cmap(c) for c in colorind) #['b', 'r', 'orange', 'limegreen']
        for rd in [1,-1]:
            for ang in [1, -1]:
                key = (rd,ang)
                branch_pa = (pa + ang*np.pi/2) % (2*np.pi)
                for pi, pR in enumerate(R):
                    if pi % 3 != 0:
                        continue
                    start = np.array([results['center']['x'] + ang*rd*pR*np.cos(pa + (0 if ang > 0 else np.pi)),
                                      results['center']['y'] + ang*rd*pR*np.sin(pa + (0 if ang > 0 else np.pi))])
                    end = start + R[-1]*np.array([np.cos(branch_pa), np.sin(branch_pa)])
                    start -= np.array([ranges[0][0], ranges[1][0]])
                    end -= np.array([ranges[0][0], ranges[1][0]])
                    plt.plot([start[0],end[0]], [start[1],end[1]], linewidth = 0.5, color = colours[count], label = ('%sR : pa%s90' % ('+' if rd > 0 else '-', '+' if ang > 0 else '-')) if pi == 0 else None)
                count += 1
        plt.legend()
        plt.xlim([0,ranges[0][1] - ranges[0][0]])
        plt.ylim([0,ranges[1][1] - ranges[1][0]])
        plt.tight_layout()
        if not ('nologo' in kwargs and kwargs['nologo']):
            AddLogo(plt.gcf())
        plt.savefig('%sorthogonal_sample_lines_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', kwargs['name']), dpi = kwargs['plotdpi'] if 'plotdpi'in kwargs else 300)
        plt.close()        
            
    return IMG, {}
