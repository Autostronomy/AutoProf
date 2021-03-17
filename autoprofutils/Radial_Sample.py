import numpy as np
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _iso_extract, Angle_TwoAngles
from scipy.stats import iqr
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
import matplotlib
import logging

def Radial_Sample(IMG, pixscale, name, results, **kwargs):

    nspokes = kwargs['radsample_spokes'] if 'radsample_spokes' in kwargs else 4
    spokeangles = np.linspace(0, 2*np.pi*(1 - 1./nspokes), nspokes)

    spokewidth = kwargs['radsample_width'] if 'radsample_width' in kwargs else 10.
    spokewidth *= np.pi/180

    zeropoint = kwargs['zeropoint'] if 'zeropoint' in kwargs else 22.5
    
    if spokewidth*nspokes > 2*np.pi:
        logging.warning('%s: Radial sampling spokes are overlapping! %i spokes with a width of %.3f rad' % (nspokes, spokewidth))

    pa = results['init pa']
    eps = results['init ellip']
    R = np.array(results['prof data']['R'])/pixscale
    dat = IMG - results['background']
    ranges = [[max(0,int(results['center']['x']-R[-1]-2)), min(IMG.shape[1],int(results['center']['x']+R[-1]+2))],
              [max(0,int(results['center']['y']-R[-1]-2)), min(IMG.shape[0],int(results['center']['y']+R[-1]+2))]]
    XX, YY = np.meshgrid(np.arange(ranges[0][1] - ranges[0][0], dtype = float), np.arange(ranges[1][1] - ranges[1][0], dtype = float))
    XX -= results['center']['x'] - float(ranges[0][0])
    YY -= results['center']['y'] - float(ranges[1][0])
    XX, YY = (XX*np.cos(-pa) - YY*np.sin(-pa), XX*np.sin(-pa) + YY*np.cos(-pa))
    theta = (np.arctan(YY/XX) + np.pi*(XX < 0))
    YY /= 1 - eps
    RR = np.sqrt(XX**2 + YY**2)

    sb = list([] for i in spokeangles)
    sbE = list([] for i in spokeangles)

    for i in range(len(R)):
        if R[i] < 40:
            isovals = list(_iso_extract(dat, R[i], eps, pa, results['center'], more = True, forceN = max(int(3*2*np.pi/spokewidth), int(5*R[i]))))
            isovals[1] = (isovals[1] - pa)
        else:
            isobandwidth = R[i]*(kwargs['isoband_width'] if 'isoband_width' in kwargs else 0.025)            
            rselect = np.logical_and(RR > R[i] - isobandwidth, RR < R[i] + isobandwidth)
            isovals = (dat[ranges[1][0]:ranges[1][1],ranges[0][0]:ranges[0][1]][rselect], theta[rselect])

        for sa_i in range(len(spokeangles)):
            aselect = np.abs(Angle_TwoAngles(spokeangles[sa_i], isovals[1])) < (spokewidth/2)
            if np.sum(aselect) == 0:
                sb[sa_i].append(99.999)
                sbE[sa_i].append(99.999)
                continue
            medflux = np.median(isovals[0][aselect])
            sb[sa_i].append((-2.5*np.log10(medflux) + zeropoint + 5*np.log10(pixscale)) if medflux > 0 else 99.999)
            sbE[sa_i].append((2.5*iqr(isovals[0][aselect], rng = (31.731/2, 100 - 31.731/2)) / (2*np.sqrt(np.sum(aselect))*medflux*np.log(10))) if medflux > 0 else 99.999)

    newprofheader = results['prof header']
    newprofunits = results['prof units']
    newprofformat = results['prof format']
    newprofdata = results['prof data']
    for sa_i in range(len(spokeangles)):
        p1, p2 = ('SB[%.1f]' % (spokeangles[sa_i]*180/np.pi), 'SB_e[%.1f]' % (spokeangles[sa_i]*180/np.pi))
        newprofheader.append(p1)
        newprofheader.append(p2)
        newprofunits[p1] = 'mag*arcsec^-2'
        newprofunits[p2] = 'mag*arcsec^-2'
        newprofformat[p1] = '%.4f'
        newprofformat[p2] = '%.4f'
        newprofdata[p1] = sb[sa_i]
        newprofdata[p2] = sbE[sa_i]
        
    if 'doplot' in kwargs and kwargs['doplot']:
        cmap = matplotlib.cm.get_cmap('tab20' if nspokes <= 20 else 'viridis')
        colorind = np.linspace(0,1,nspokes)
        for sa_i in range(len(spokeangles)):
            CHOOSE = np.logical_and(np.array(sb[sa_i]) < 99, np.array(sbE[sa_i]) < 1)
            plt.errorbar(np.array(R)[CHOOSE]*pixscale, np.array(sb[sa_i])[CHOOSE], yerr = np.array(sbE[sa_i])[CHOOSE],
                         elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = cmap(colorind[sa_i]), label = 'Spoke %.2f' % (spokeangles[sa_i]*180/np.pi))
        plt.xlabel('Radius [arcsec]')
        plt.ylabel('Surface Brightness [mag/arcsec^2]')
        bkgrdnoise = -2.5*np.log10(results['background noise']) + zeropoint + 2.5*np.log10(pixscale**2)
        plt.axhline(bkgrdnoise, color = 'purple', linewidth = 0.5, linestyle = '--', label = '1$\\sigma$ noise/pixel: %.1f mag arcsec$^{-2}$' % bkgrdnoise)
        plt.gca().invert_yaxis()
        plt.legend(fontsize = 10)
        plt.savefig('%sradial_sample_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
        plt.close()

        plt.imshow(np.clip(dat[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]],
                           a_min = 0,a_max = None), origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        cx, cy = (results['center']['x'] - ranges[0][0], results['center']['y'] - ranges[1][0])
        for sa_i in range(len(spokeangles)):
            endx, endy = (R[-1]*np.cos(spokeangles[sa_i]+pa), R[-1]*np.sin(spokeangles[sa_i]+pa))
            plt.plot([cx, endx + cx], [cy, endy + cy], color = cmap(colorind[sa_i]), linewidth = 2)
            endx, endy = (R[-1]*np.cos(spokeangles[sa_i]+pa + spokewidth/2), R[-1]*np.sin(spokeangles[sa_i]+pa + spokewidth/2))
            plt.plot([cx, endx + cx], [cy, endy + cy], color = cmap(colorind[sa_i]), linestyle = '--')
            endx, endy = (R[-1]*np.cos(spokeangles[sa_i]+pa - spokewidth/2), R[-1]*np.sin(spokeangles[sa_i]+pa - spokewidth/2))
            plt.plot([cx, endx + cx], [cy, endy + cy], color = cmap(colorind[sa_i]), linestyle = '--')
            
        plt.xlim([0,ranges[0][1] - ranges[0][0]])
        plt.ylim([0,ranges[1][1] - ranges[1][0]])
        plt.savefig('%sradial_sample_spokes_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
        plt.close()
        
    return {'prof header': newprofheader, 'prof units': newprofunits, 'prof data': newprofdata, 'prof format': newprofformat}
