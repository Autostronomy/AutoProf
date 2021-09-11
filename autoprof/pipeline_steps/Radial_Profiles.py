import numpy as np
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _iso_extract, _iso_between, Angle_TwoAngles, LSBImage, AddLogo, _average, _scatter, flux_to_sb
from autoprofutils.Diagnostic_Plots import Plot_Radial_Profiles
from scipy.stats import iqr
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
import matplotlib
import logging

def Radial_Profiles(IMG, results, options):
    """Extracts SB profiles along lines radiating from the galaxy center.

    For some applications, such as examining edge on galaxies, it is
    beneficial to observe the structure in a disk as well as (or
    instead of) the average isophotal profile. This can done with
    radial profiles which sample along lines radiating form the galaxy
    center. These lines are by default placed on the 4 semi-axes of
    the galaxy. The lines are actually wedges with increasing width as
    a function of radius. This helps keep roughly constant S/N in the
    bins, allowing the profile to extend far into the outskirts of a
    galaxy. The user may increase the number of wedgest to extract
    more stucture from the galaxy, however at some point the wedges
    will begin to cross each other. AutoProf will warn the user when
    this happens, but will carry on anyway.
    
    Arguments
    -----------------
    ap_radialprofiles_nwedges: int
      number of radial wedges to sample. Recommended choosing a power
      of 2.

      :default:
        4

    ap_radialprofiles_width: float
      User set width of radial sampling wedges in degrees.

      :default:
        15

    ap_radialprofiles_pa: float
      user set position angle at which to measure radial wedges
      relative to the global position angle, in degrees.

      :default:
        0

    ap_radialprofiles_expwidth: bool
      Tell AutoProf to use exponentially increasing widths for radial
      samples. In this case *ap_radialprofiles_width* corresponds to
      the final width of the radial sampling.

      :default:
        False

    ap_radialprofiles_variable_pa: bool
      Tell AutoProf to rotate radial sampling wedges with the position
      angle profile of the galaxy.

      :default:
        False

    References
    ----------
    - 'prof header'
    - 'prof units'
    - 'prof data'
    - 'mask' (optional)
    - 'background'
    - 'center'
    - 'init pa' (optional)
    
    Returns
    -------
    IMG: ndarray
      Unaltered galaxy image
    
    results: dict
      No results provided as this method writes its own profile
    
      .. code-block:: python
     
        {'prof header': , # Previously extracted SB profile, with extra columns appended for radial profiles (list)
         'prof units': , # Previously extracted SB profile, with extra units appended for radial profiles (dict)
         'prof data': # Previously extracted SB profile, with extra columns appended for radial profiles (dict)
    
        }

    """
    

    mask = results['mask'] if 'mask' in results else None
    nwedges = options['ap_radialprofiles_nwedges'] if 'ap_radialprofiles_nwedges' in options else 4
    wedgeangles = np.linspace(0, 2*np.pi*(1 - 1./nwedges), nwedges)

    zeropoint = options['ap_zeropoint'] if 'ap_zeropoint' in options else 22.5

    R = np.array(results['prof data']['R'])/options['ap_pixscale']
    SBE = np.array(results['prof data']['SB_e'])
    if 'ap_radialprofiles_variable_pa' in options and options['ap_radialprofiles_variable_pa']:
        pa = np.array(results['prof data']['pa'])*np.pi/180
    else:
        pa = np.ones(len(R))*((options['ap_radialprofiles_pa']*np.pi/180) if 'ap_radialprofiles_pa' in options else results['init pa'])
    dat = IMG - results['background']

    maxwedgewidth = options['ap_radialprofiles_width'] if 'ap_radialprofiles_width' in options else 15.
    maxwedgewidth *= np.pi/180
    if 'ap_radialprofiles_expwidth' in options and options['ap_radialprofiles_expwidth']:
        wedgewidth = maxwedgewidth*np.exp(R/R[-1] - 1)
    else:
        wedgewidth = np.ones(len(R)) * maxwedgewidth

    if wedgewidth[-1]*nwedges > 2*np.pi:
        logging.warning('%s: Radial sampling wedges are overlapping! %i wedges with a maximum width of %.3f rad' % (nwedges, wedgewidth[-1]))
        
    sb = list([] for i in wedgeangles)
    sbE = list([] for i in wedgeangles)

    for i in range(len(R)):
        if R[i] < 100:
            isovals = list(_iso_extract(dat, R[i], {'ellip': 0, 'pa': 0}, results['center'], more = True, minN = int(5*2*np.pi/wedgewidth[i]), mask = mask))
        else:
            isobandwidth = R[i]*(options['ap_isoband_width'] if 'ap_isoband_width' in options else 0.025)
            isovals = list(_iso_between(dat, R[i] - isobandwidth, R[i] + isobandwidth, {'ellip': 0, 'pa': 0}, results['center'], more = True, mask = mask))
        isovals[1] -= pa[i]
        
        for sa_i in range(len(wedgeangles)):
            aselect = np.abs(Angle_TwoAngles(wedgeangles[sa_i], isovals[1])) < (wedgewidth[i]/2)
            if np.sum(aselect) == 0:
                sb[sa_i].append(99.999)
                sbE[sa_i].append(99.999)
                continue
            medflux = _average(isovals[0][aselect], options['ap_isoaverage_method'] if 'ap_isoaverage_method' in options else 'median')
            scatflux = _scatter(isovals[0][aselect], options['ap_isoaverage_method'] if 'ap_isoaverage_method' in options else 'median')
            sb[sa_i].append(flux_to_sb(medflux, options['ap_pixscale'], zeropoint) if medflux > 0 else 99.999)
            sbE[sa_i].append((2.5*scatflux / (np.sqrt(np.sum(aselect))*medflux*np.log(10))) if medflux > 0 else 99.999)

    newprofheader = results['prof header']
    newprofunits = results['prof units']
    newprofdata = results['prof data']
    for sa_i in range(len(wedgeangles)):
        p1, p2 = ('SB_rad[%.1f]' % (wedgeangles[sa_i]*180/np.pi), 'SB_rad_e[%.1f]' % (wedgeangles[sa_i]*180/np.pi))
        newprofheader.append(p1)
        newprofheader.append(p2)
        newprofunits[p1] = 'mag*arcsec^-2'
        newprofunits[p2] = 'mag*arcsec^-2'
        newprofdata[p1] = sb[sa_i]
        newprofdata[p2] = sbE[sa_i]
        
    if 'ap_doplot' in options and options['ap_doplot']:
        Plot_Radial_Profiles(dat, sb, sbE, pa, nwedges, wedgeangles, wedgewidth, results, options)
        
    return IMG, {'prof header': newprofheader, 'prof units': newprofunits, 'prof data': newprofdata}
