import sys
import os
from scipy.integrate import trapz
from scipy.stats import iqr
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from photutils.isophote import EllipseSample, EllipseGeometry, Isophote, IsophoteList
from photutils.isophote import Ellipse as Photutils_Ellipse

Abs_Mag_Sun = {'u': 6.39,
               'g': 5.11,
               'r': 4.65,
               'i': 4.53,
               'z': 4.50,
               'U': 6.33,
               'B': 5.31,
               'V': 4.80,
               'R': 4.60,
               'I': 4.51,
               'J': 4.54,
               'H': 4.66,
               'K': 5.08,
               '3.6um': 3.24}     # mips.as.arizona.edu/~cnaw/sun.html # also see: http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
Au_to_Pc    = 4.84814e-6         # pc au^-1
Pc_to_m     = 3.086e16           # m pc^-1
Pc_to_Au    = 206265.            # Au pc^-1


def magperarcsec2_to_mag(mu, a = None, b = None, A = None):
    """
    Converts mag/arcsec^2 to mag
    mu: mag/arcsec^2
    a: semi major axis radius (arcsec)
    b: semi minor axis radius (arcsec)
    A: pre-calculated area (arcsec^2)
    returns: mag
    """
    assert (not A is None) or (not a is None and not b is None)
    if A is None:
        A = np.pi * a * b
    return mu - 2.5*np.log10(A) # https://en.wikipedia.org/wiki/Surface_brightness#Calculating_surface_brightness

def mag_to_magperarcsec2(m, a = None, b = None, R = None, A = None):
    """
    Converts mag to mag/arcsec^2
    m: mag
    a: semi major axis radius (arcsec)
    b: semi minor axis radius (arcsec)
    A: pre-calculated area (arcsec^2)
    returns: mag/arcsec^2
    """
    assert (not A is None) or (not a is None and not b is None) or (not R is None)
    if not R is None:
        A = np.pi * (R**2)
    elif A is None:
        A = np.pi * a * b
    return m + 2.5*np.log10(A) # https://en.wikipedia.org/wiki/Surface_brightness#Calculating_surface_brightness

def halfmag(mag):
    """
    Computes the magnitude corresponding to half in log space.
    Effectively, converts to luminosity, divides by 2, then
    converts back to magnitude. Distance is not needed as it
    cancels out. Here is a basic walk through:
    m_1 - m_ref = -2.5log10(I_1/I_ref)
    m_2 - m_ref = -2.5log10(I_1/2I_ref)
                = -2.5log10(I_1/I_ref) + 2.5log10(2)
    m_2 = m_1 + 2.5log10(2)
    """

    return mag + 2.5 * np.log10(2)

def pc_to_arcsec(R, D, Re = 0.0, De = 0.0):
    """
    Converts a size in parsec to arcseconds

    R: length in pc
    D: distance in pc
    """

    theta = R / (D * Au_to_Pc)
    if np.all(Re == 0) and np.all(De == 0):
        return theta
    else:
        e = theta * np.sqrt((Re/R)**2 + (De/D)**2)
        return theta, e

def arcsec_to_pc(theta, D, thetae = 0.0, De = 0.0):
    """
    Converts a size in arcseconds to parsec

    theta: angle in arcsec
    D: distance in pc
    """
    r = theta * D * Au_to_Pc
    if np.all(thetae == 0) and np.all(De == 0):
        return r
    else:
        e = r * np.sqrt((thetae / theta)**2 + (De / D)**2)
        return r, e

def ISB_to_muSB(I, band, IE = None):
    """
    Converts surface brightness in Lsolar pc^-2 into mag arcsec^-2

    I: surface brightness, (L/Lsun) pc^-2
    band: Photometric band in which measurements were taken
    returns: surface brightness in mag arcsec^-2
    """

    muSB = 21.571 + Abs_Mag_Sun[band] - 2.5 * np.log10(I)
    if IE is None:
        return muSB
    else:
        return muSB, (2.5/np.log(10)) * IE / I

def muSB_to_ISB(mu, band, muE = None):
    """
    Converts surface brightness in mag arcsec^-2 into Lsolar pc^-2

    mu: surface brightness, mag arcsec^-2
    band: Photometric band in which measurements were taken
    returns: surface brightness in (L/Lsun) pc^-2
    """

    ISB = 10**((21.571 + Abs_Mag_Sun[band] - mu)/2.5)
    if muE is None:
        return ISB
    else:
        return ISB, (np.log(10)/2.5) * ISB * muE

def app_mag_to_abs_mag(m, D, me = 0., De = 0.):
    """
    Converts an apparent magnitude to an absolute magnitude
    m: Apparent magnitude
    D: Distance to object in parsecs
    returns: Absolute magnitude at 10 parcecs
    """

    M = m - 5.0 * np.log10(D / 10.0)
    if np.all(me == 0) and np.all(De == 0):
        return M
    else:
        return M, np.sqrt(me**2 + (5. * De / (D * np.log(10)))**2)

def abs_mag_to_app_mag(M, D, Me = 0., De = 0.):
    """
    Converts an absolute magnitude to an apparent magnitude
    M: Absolute magnitude at 10 parcecs
    D: Distance to object in parsecs
    returns: Apparent magnitude
    """

    m = M + 5.0 * np.log10(D / 10.0)
    if np.all(Me == 0) and np.all(De == 0):
        return m
    else:
        return m, np.sqrt(Me**2 + (5. * De / (D * np.log(10)))**2)

    
def mag_to_L(mag, band, mage = None, zeropoint = None):
    """
    Returns the luminosity (in solar luminosities) given the absolute magnitude and reference point.
    mag: Absolute magnitude
    band: Photometric band in which measurements were taken
    mage: uncertainty in absolute magnitude
    zeropoint: user defined zero point
    returns: Luminosity in solar luminosities
    """

    L = 10**(((Abs_Mag_Sun[band] if zeropoint is None else zeropoint) - mag)/2.5)
    if mage is None:
        return L
    else:
        Le = np.abs(L * mage * np.log(10) / 2.5)
        return L, Le

def L_to_mag(L, band, Le = None, zeropoint = None):
    """
    Returns the Absolute magnitude of a star given its luminosity and distance
    L: Luminosity in solar luminosities
    band: Photometric band in which measurements were taken
    Le: Uncertainty in luminosity
    zeropoint: user defined zero point
    
    returns: Absolute magnitude
    """

    mag = (Abs_Mag_Sun[band]if zeropoint is None else zeropoint) - 2.5 * np.log10(L)
    
    if Le is None:
        return mag
    else:
        mage = np.abs(2.5 * Le / (L * np.log(10)))
        return mag, mage


def _iso_extract(IMG, sma, eps, pa, c, more = False):
    """
    Internal, basic function for extracting the pixel fluxes along and isophote
    """

    # Geometry of the isophote
    geo = EllipseGeometry(x0 = c['x'], y0 = c['y'],
                          sma = sma,
                          eps = eps,
                          pa = pa)
    # Extract the isophote information
    ES = EllipseSample(IMG,
                       sma = sma,
                       geometry = geo,
                       integrmode = 'bilinear' if sma < 50 else 'nearest_neighbor')
    ES.extract(max_samples = 100)
    # Return the desited vlaues, either just SB values,
    # or SB values and angles
    if more:
        return ES.values[2], ES.values[0]
    else:
        return ES.values[2]


def _x_to_pa(x):
    """
    Internal, basic function to ensure position angles remain
    in the proper parameter space (0, pi)
    """
    return x % np.pi #np.pi / (1. + np.exp(-(x-np.pi/2)))
def _inv_x_to_pa(pa):
    """
    Internal, reverse _x_to_pa, although this is just a filler
    function as _x_to_pa is non reversible.
    """
    return pa % np.pi
#
def _x_to_eps(x):
    """
    Internal, function to map the reals to the range (0.05,0.95)
    as the range of reasonable ellipticity values.
    """
    return 0.02 + 0.96*(0.5 + np.arctan(x-0.5)/np.pi) #0.02 + 0.96/(1. + np.exp(-(x - 0.5))) 
#
def _inv_x_to_eps(eps):
    """
    Internal, inverse of _x_to_eps function
    """
    return 0.5 + np.tan(np.pi*((eps - 0.02)/0.96 - 0.5)) #0.5 - np.log(0.96/(eps - 0.02) - 1.) 


def Read_Image(filename, **kwargs):
    """
    Reads a galaxy image given a file name. In a fits image the data is assumed to exist in the
    primary HDU unless given 'hdulelement'. In a numpy file, it is assumed that only one image
    is in the file.
    
    filename: A string containing the full path to an image file

    returns: Extracted image data as numpy 2D array
    """

    # Read a fits file
    if filename[filename.rfind('.')+1:].lower() == 'fits':
        hdul = fits.open(filename)
        dat = hdul[kwargs['hdulelement'] if 'hdulelement' in kwargs else 0].data
    # Read a numpy array file
    if filename[filename.rfind('.')+1:].lower() == 'npy':
        dat = np.load(filename)
            
    return dat

def Angle_TwoAngles(a1, a2):
    """
    Compute the angle between two vectors at angles a1 and a2
    """

    return np.arccos(np.sin(a1)*np.sin(a2) + np.cos(a1)*np.cos(a2))
    

def Angle_Average(a):
    """
    Compute the average for a list of angles, which may wrap around a cyclic boundary.

    a: list of angles in the range [0,2pi]
    """
    i = np.cos(a) + 1j*np.sin(a)
    return np.angle(np.mean(i))

def Angle_Scatter(a):
    """
    Compute the average for a list of angles, which may wrap around a cyclic boundary.

    a: list of angles in the range [0,2pi]
    """
    i = np.cos(a) + 1j*np.sin(a)
    return iqr(np.angle(1j*i/np.mean(i)),rng = [16,84])


def GetKwargs(c):

    newkwargs = {}

    try:
        newkwargs['saveto'] = c.saveto
    except:
        newkwargs['saveto'] = None
    try:
        newkwargs['name'] = c.name
    except:
        newkwargs['name'] = None
    try:
        newkwargs['n_procs'] = c.n_procs
    except:
        newkwargs['n_procs'] = 1
    try:
        newkwargs['mask_file'] = c.mask_file
    except:
        newkwargs['mask_file'] = None
    try:
        newkwargs['savemask'] = c.savemask
    except:
        pass
    try:
        newkwargs['autodetectoverflow'] = c.autodetectoverflow
    except:
        pass
    try:
        newkwargs['overflowval'] = c.overflowval
    except:
        pass
    try:
        newkwargs['forcing_profile'] = c.forcing_profile
    except:
        pass
    try:
        newkwargs['plotpath'] = c.plotpath
    except:
        pass
    try:
        newkwargs['doplot'] = c.doplot
    except:
        pass
    try:
        newkwargs['hdulelement'] = c.hdulelement
    except:
        pass
    try:
        newkwargs['given_centers'] = c.given_centers
    except:
        pass
    try:
        newkwargs['scale'] = c.scale
    except:
        pass
    try:
        newkwargs['samplegeometricscale'] = c.samplegeometricscale
    except:
        pass
    try:
        newkwargs['samplelinearscale'] = c.samplelinearscale
    except:
        pass
    try:
        newkwargs['samplestyle'] = c.samplestyle
    except:
        pass
    try:
        newkwargs['sampleinitR'] = c.sampleinitR
    except:
        pass
    try:
        newkwargs['sampleendR'] = c.sampleendR
    except:
        pass
    try:
        newkwargs['sampleerrorlim'] = c.sampleerrorlim
    except:
        pass
    try:
        newkwargs['zeropoint'] = c.zeropoint
    except:
        pass
    return newkwargs



def SBprof_to_COG(R, SB, axisratio, method = 0):
    """
    Converts a surface brightness profile to a curve of growth by integrating
    the SB profile in flux units then converting back to mag units. Two methods
    are implemented, one using the trapezoid method and one assuming constant
    SB between isophotes. Trapezoid method is in principle more accurate, but
    may become unstable with erratic data.

    R: Radius in arcsec
    SB: surface brightness in mag arcsec^-2
    axisratio: b/a indicating degree of isophote ellipticity
    method: 0 for trapezoid, 1 for constant

    returns: magnitude values at each radius of the profile in mag
    """
    
    m = np.zeros(len(R))
    # Dummy band for conversions, cancelled out on return
    band = 'r'

    # Method 0 uses trapezoid method to integrate in flux space
    if method == 0:
        # Compute the starting point assuming constant SB within first isophote
        m[0] = magperarcsec2_to_mag(SB[0], A = np.pi*axisratio[0]*(R[0]**2))
        # Dummy distance value for integral
        D = 10
        # Convert to flux space
        I = muSB_to_ISB(np.array(SB), band)
        # Ensure numpy array
        axisratio = np.array(axisratio)
        # Convert to physical radius using dummy distance
        R = arcsec_to_pc(np.array(R), D)
        # Integrate up to each radius in the profile
        for i in range(1,len(R)):
            m[i] = abs_mag_to_app_mag(L_to_mag(trapz(2*np.pi*I[:i+1]*R[:i+1]*axisratio[:i+1],R[:i+1]) + \
                                               mag_to_L(app_mag_to_abs_mag(m[0], D), band), band),D)
    elif method == 1:
        # Compute the starting point assuming constant SB within first isophote
        m[0] = magperarcsec2_to_mag(SB[0], A = np.pi*axisratio[0]*(R[0]**2))
        # Progress through each radius and add the contribution from each isophote individually
        for i in range(1,len(R)):
            m[i] = L_to_mag(mag_to_L(magperarcsec2_to_mag(SB[i], A = np.pi*axisratio[i]*(R[i]**2)), band) - \
                            mag_to_L(magperarcsec2_to_mag(SB[i], A = np.pi*axisratio[i-1]*(R[i-1]**2)), band) + \
                            mag_to_L(m[i-1], band), band)

    return m


def SBprof_to_COG_errorprop(R, SB, SBE, axisratio, axisratioE = None, N = 100, method = 0, symmetric_error = True):
    """
    Converts a surface brightness profile to a curve of growth by integrating
    the SB profile in flux units then converting back to mag units. Two methods
    are implemented, one using the trapezoid method and one assuming constant
    SB between isophotes. Trapezoid method is in principle more accurate, but
    may become unstable with erratic data. An uncertainty profile is also
    computed, from a given SB uncertainty profile and optional axisratio
    uncertainty profile.
    
    R: Radius in arcsec
    SB: surface brightness in mag arcsec^-2
    SBE: surface brightness uncertainty relative mag arcsec^-2
    axisratio: b/a indicating degree of isophote ellipticity
    axisratioE: uncertainty in b/a
    N: number of iterations for computing uncertainty
    method: 0 for trapezoid, 1 for constant
    
    returns: magnitude and uncertainty profile in mag
    """

    # If not provided, axis ratio error is assumed to be zero
    if axisratioE is None:
        axisratioE = np.zeros(len(R))
        
    # Create container for the monte-carlo iterations
    COG_results = np.zeros((N, len(R))) + 99.999
    SB_CHOOSE = np.logical_and(np.isfinite(SB), SB < 50)
    if np.sum(SB_CHOOSE) < 5:
        return (None, None) if symmetric_error else (None, None, None)
    COG_results[0][SB_CHOOSE] = SBprof_to_COG(R[SB_CHOOSE], SB[SB_CHOOSE], axisratio[SB_CHOOSE], method = method)
    for i in range(1,N):
        # Randomly sampled SB profile
        tempSB = np.random.normal(loc = SB, scale = SBE)
        # Randomly sampled axis ratio profile
        tempq = np.random.normal(loc = axisratio, scale = axisratioE)
        # Compute COG with sampled data
        COG_results[i][SB_CHOOSE] = SBprof_to_COG(R[SB_CHOOSE], tempSB[SB_CHOOSE], tempq[SB_CHOOSE], method = method)

    # Condense monte-carlo evaluations into profile and uncertainty envelope
    COG_profile = COG_results[0]
    COG_lower = np.median(COG_results, axis = 0) - np.quantile(COG_results, 0.317310507863/2, axis = 0)
    COG_upper = np.quantile(COG_results, 1. - 0.317310507863/2, axis = 0) - np.median(COG_results, axis = 0)

    # Return requested uncertainty format
    if symmetric_error:
        return COG_profile, np.abs(COG_lower + COG_upper)/2
    else:
        return COG_profile, COG_lower, COG_upper
    
