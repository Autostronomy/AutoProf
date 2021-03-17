import numpy as np
from astropy.io import fits


def EllipseModel_Fix(IMG, pixscale, name, results, **kwargs):

    zeropoint = kwargs['zeropoint'] if 'zeropoint' in kwargs else 22.5
    pa = results['init pa']
    eps = results['init ellip']
    
    CHOOSE = np.array(results['prof data']['SB_e']) < 0.3
    R = np.array(results['prof data']['R'])[CHOOSE]/pixscale
    SB = np.array(results['prof data']['SB'])

    Model = np.zeros(IMG.shape, dtype = IMG.dtype)
    ranges = [[max(0,int(results['center']['x']-R[-1]-2)), min(IMG.shape[1],int(results['center']['x']+R[-1]+2))],
              [max(0,int(results['center']['y']-R[-1]-2)), min(IMG.shape[0],int(results['center']['y']+R[-1]+2))]]
    XX, YY = np.meshgrid(np.arange(ranges[0][1] - ranges[0][0], dtype = float), np.arange(ranges[1][1] - ranges[1][0], dtype = float))
    XX -= results['center']['x'] - float(ranges[0][0])
    YY -= results['center']['y'] - float(ranges[1][0])
    XX, YY = (XX*np.cos(-pa) - YY*np.sin(-pa), XX*np.sin(-pa) + YY*np.cos(-pa))
    YY /= 1 - eps
    RR = np.sqrt(XX**2 + YY**2)

    MM = np.interp(RR.ravel(), R, SB).reshape(RR.shape)
    MM = 10**(-(MM - zeropoint - 5*np.log10(pixscale))/2.5)
    MM[RR > R[-1]] = 0
    Model[ranges[1][0]:ranges[1][1],ranges[0][0]:ranges[0][1]] = MM

    header = fits.Header()
    hdul = fits.HDUList([fits.PrimaryHDU(header=header),
                         fits.ImageHDU(Model)])
    
    hdul.writeto('%s%s_fixmodel.fits' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), overwrite = True)
    
    return {'ellipse model': Model}
    
