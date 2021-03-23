import numpy as np
from astropy.io import fits
from scipy.interpolate import SmoothBivariateSpline
from copy import deepcopy
import matplotlib.pyplot as plt

def EllipseModel_Fix(IMG, pixscale, name, results, **kwargs):

    zeropoint = kwargs['zeropoint'] if 'zeropoint' in kwargs else 22.5
    pa = results['init pa']
    eps = results['init ellip']
    
    CHOOSE = np.array(results['prof data']['SB_e']) < 0.3
    R = np.array(results['prof data']['R'])[CHOOSE]/pixscale
    SB = np.array(results['prof data']['SB'])[CHOOSE]

    Model = np.zeros(IMG.shape, dtype = np.float32)
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
    

def EllipseModel_General(IMG, pixscale, name, results, **kwargs):
    
    zeropoint = kwargs['zeropoint'] if 'zeropoint' in kwargs else 22.5
    
    CHOOSE = np.array(results['prof data']['SB_e']) < 0.5
    R = np.array(results['prof data']['R'])[CHOOSE]/pixscale
    SB = np.array(results['prof data']['SB'])[CHOOSE]
    SB_e = np.clip(np.array(results['prof data']['SB_e'])[CHOOSE], a_min = 1e-3, a_max = None)
    PA = np.array(results['prof data']['pa'])[CHOOSE]*np.pi/180
    ellip = np.array(results['prof data']['ellip'])[CHOOSE]
    
    X = []
    Y = []
    XY_R = []
    M = []
    M_e = []
    for i in range(len(R)):
        N = max(4,int(R[i]))
        theta = np.linspace(0, 2*np.pi*(1 - 1/N), N)
        x = R[i]*np.cos(theta)
        y = R[i]*(1 - ellip[i])*np.sin(theta)
        x,y = (x*np.cos(PA[i]) - y*np.sin(PA[i]), x*np.sin(PA[i]) + y*np.cos(PA[i]))
        XY_R += list(np.sqrt(x**2 + y**2))
        X += list(x)
        Y += list(y)
        M += list(np.ones(len(x))*SB[i])
        M_e += list(np.ones(len(x))*SB_e[i])
    XY_R = np.array(XY_R)
    X = np.array(X)
    Y = np.array(Y)
    M = np.array(M)
    
    ranges = [[max(0,int(results['center']['x']-R[-1]-2)), min(IMG.shape[1],int(results['center']['x']+R[-1]+2))],
              [max(0,int(results['center']['y']-R[-1]-2)), min(IMG.shape[0],int(results['center']['y']+R[-1]+2))]]
    XX, YY = np.meshgrid(np.arange(ranges[0][1] - ranges[0][0], dtype = float), np.arange(ranges[1][1] - ranges[1][0], dtype = float))
    S = deepcopy(XX.shape)
    XX -= results['center']['x'] - float(ranges[0][0])
    YY -= results['center']['y'] - float(ranges[1][0])
    
    MM = np.zeros(XX.shape)
    for i in range(XX.shape[0]):
        CHOOSE = abs(Y - YY[i,int(YY.shape[1]/2)]) < (10*results['psf fwhm'])
        K = -((XX[i,:].reshape(XX.shape[1],-1) - X[CHOOSE])**2 + (YY[i,:].reshape(XX.shape[1],-1) - Y[CHOOSE])**2)/(results['psf fwhm'])**2
        K = np.exp(K - (np.max(K,axis = 1)).reshape(K.shape[0],-1))
        MM[i,:] = np.sum(M[CHOOSE]*K,axis = 1) / np.sum(K,axis = 1)
        
    MM = 10**(-(MM - zeropoint - 5*np.log10(pixscale))/2.5)
    
    XX, YY = (XX*np.cos(-PA[-1]) - YY*np.sin(-PA[-1]), XX*np.sin(-PA[-1]) + YY*np.cos(-PA[-1]))
    YY /= 1 - ellip[-1]
    RR = np.sqrt(XX**2 + YY**2)
    MM[RR > R[-1]] = 0
    
    Model = np.zeros(IMG.shape, dtype = np.float32)
    Model[ranges[1][0]:ranges[1][1],ranges[0][0]:ranges[0][1]] = MM
    
    header = fits.Header()
    hdul = fits.HDUList([fits.PrimaryHDU(header=header),
                         fits.ImageHDU(Model)])
    
    hdul.writeto('%s%s_model.fits' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), overwrite = True)
    
    return {'ellipse model': Model}
