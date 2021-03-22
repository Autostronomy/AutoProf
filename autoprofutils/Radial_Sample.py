import numpy as np
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _iso_extract, _iso_between, Angle_TwoAngles
from scipy.stats import iqr
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
import matplotlib
import logging

def Radial_Sample(IMG, pixscale, name, results, **kwargs):

    mask = results['mask'] if 'mask' in results else None
    nwedges = kwargs['radsample_nwedges'] if 'radsample_nwedges' in kwargs else 4
    wedgeangles = np.linspace(0, 2*np.pi*(1 - 1./nwedges), nwedges)

    zeropoint = kwargs['zeropoint'] if 'zeropoint' in kwargs else 22.5

    R = np.array(results['prof data']['R'])/pixscale
    if 'radsample_variable_pa' in kwargs and kwargs['radsample_variable_pa']:
        pa = np.array(results['prof data']['pa'])*np.pi/180
    else:
        pa = np.ones(len(R))*((kwargs['radsample_pa']*np.pi/180) if 'radsample_pa' in kwargs else results['init pa'])
    dat = IMG - results['background']

    maxwedgewidth = kwargs['radsample_width'] if 'radsample_width' in kwargs else 15.
    maxwedgewidth *= np.pi/180
    if 'radsample_expwidth' in kwargs and kwargs['radsample_expwidth']:
        wedgewidth = maxwedgewidth*np.exp(R/R[-1] - 1)
    else:
        wedgewidth = np.ones(len(R)) * maxwedgewidth

    if wedgewidth[-1]*nwedges > 2*np.pi:
        logging.warning('%s: Radial sampling wedges are overlapping! %i wedges with a maximum width of %.3f rad' % (nwedges, wedgewidth[-1]))
        
    sb = list([] for i in wedgeangles)
    sbE = list([] for i in wedgeangles)

    for i in range(len(R)):
        if R[i] < 100:
            isovals = list(_iso_extract(dat, R[i], 0, 0, results['center'], more = True, minN = int(5*2*np.pi/wedgewidth[i]), mask = mask))
        else:
            isobandwidth = R[i]*(kwargs['isoband_width'] if 'isoband_width' in kwargs else 0.025)
            isovals = list(_iso_between(dat, R[i] - isobandwidth, R[i] + isobandwidth, 0, 0, results['center'], more = True, mask = mask))
        isovals[1] -= pa[i]
        
        for sa_i in range(len(wedgeangles)):
            aselect = np.abs(Angle_TwoAngles(wedgeangles[sa_i], isovals[1])) < (wedgewidth[i]/2)
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
    for sa_i in range(len(wedgeangles)):
        p1, p2 = ('SB[%.1f]' % (wedgeangles[sa_i]*180/np.pi), 'SB_e[%.1f]' % (wedgeangles[sa_i]*180/np.pi))
        newprofheader.append(p1)
        newprofheader.append(p2)
        newprofunits[p1] = 'mag*arcsec^-2'
        newprofunits[p2] = 'mag*arcsec^-2'
        newprofformat[p1] = '%.4f'
        newprofformat[p2] = '%.4f'
        newprofdata[p1] = sb[sa_i]
        newprofdata[p2] = sbE[sa_i]
        
    if 'doplot' in kwargs and kwargs['doplot']:
        ranges = [[max(0,int(results['center']['x']-R[-1]-2)), min(IMG.shape[1],int(results['center']['x']+R[-1]+2))],
                  [max(0,int(results['center']['y']-R[-1]-2)), min(IMG.shape[0],int(results['center']['y']+R[-1]+2))]]
        cmap = matplotlib.cm.get_cmap('tab10' if nwedges <= 10 else 'viridis')
        colorind = np.arange(nwedges)/10
        for sa_i in range(len(wedgeangles)):
            CHOOSE = np.logical_and(np.array(sb[sa_i]) < 99, np.array(sbE[sa_i]) < 1)
            plt.errorbar(np.array(R)[CHOOSE]*pixscale, np.array(sb[sa_i])[CHOOSE], yerr = np.array(sbE[sa_i])[CHOOSE],
                         elinewidth = 1, linewidth = 0, marker = '.', markersize = 5, color = cmap(colorind[sa_i]), label = 'Wedge %.2f' % (wedgeangles[sa_i]*180/np.pi))
        plt.xlabel('Radius [arcsec]')
        plt.ylabel('Surface Brightness [mag/arcsec^2]')
        bkgrdnoise = -2.5*np.log10(results['background noise']) + zeropoint + 2.5*np.log10(pixscale**2)
        plt.axhline(bkgrdnoise, color = 'purple', linewidth = 0.5, linestyle = '--', label = '1$\\sigma$ noise/pixel: %.1f mag arcsec$^{-2}$' % bkgrdnoise)
        plt.gca().invert_yaxis()
        plt.legend(fontsize = 10)
        plt.savefig('%sradial_sample_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), dpi = 500)
        plt.close()

        plt.imshow(np.clip(dat[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]],
                           a_min = 0,a_max = None), origin = 'lower', cmap = 'Greys_r', norm = ImageNormalize(stretch=LogStretch()))
        cx, cy = (results['center']['x'] - ranges[0][0], results['center']['y'] - ranges[1][0])
        for sa_i in range(len(wedgeangles)):
            endx, endy = (R*np.cos(wedgeangles[sa_i]+pa), R*np.sin(wedgeangles[sa_i]+pa))
            plt.plot(endx + cx, endy + cy, color = cmap(colorind[sa_i]), linewidth = 0.7)
            endx, endy = (R*np.cos(wedgeangles[sa_i]+pa + wedgewidth/2), R*np.sin(wedgeangles[sa_i]+pa + wedgewidth/2))
            plt.plot(endx + cx, endy + cy, color = cmap(colorind[sa_i]), linestyle = '--', linewidth = 0.5)
            endx, endy = (R*np.cos(wedgeangles[sa_i]+pa - wedgewidth/2), R*np.sin(wedgeangles[sa_i]+pa - wedgewidth/2))
            plt.plot(endx + cx, endy + cy, color = cmap(colorind[sa_i]), linestyle = '--', linewidth = 0.5)
            
        plt.xlim([0,ranges[0][1] - ranges[0][0]])
        plt.ylim([0,ranges[1][1] - ranges[1][0]])
        plt.savefig('%sradial_sample_wedges_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), dpi = 300)
        plt.close()
        
    return {'prof header': newprofheader, 'prof units': newprofunits, 'prof data': newprofdata, 'prof format': newprofformat}
