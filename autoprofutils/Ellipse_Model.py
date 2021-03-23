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
    
    CHOOSE = np.array(results['prof data']['SB_e']) < 0.3
    R = np.array(results['prof data']['R'])[CHOOSE]/pixscale
    SB = np.array(results['prof data']['SB'])[CHOOSE]
    SB_e = np.clip(np.array(results['prof data']['SB_e'])[CHOOSE], a_min = 1e-4, a_max = None)
    PA = np.array(results['prof data']['pa'])[CHOOSE]*np.pi/180
    ellip = np.array(results['prof data']['ellip'])[CHOOSE]

    X = []
    Y = []
    M = []
    M_e = []
    for i in range(len(R)):
        N = int(2*np.pi*R[i])
        theta = np.linspace(0, 2*np.pi*(1 - 1/N), N)
        x = R[i]*np.cos(theta)
        y = R[i]*(1 - ellip[i])*np.sin(theta)
        x,y = (x*np.cos(PA[i]) - y*np.sin(PA[i]), x*np.sin(PA[i]) + y*np.cos(PA[i]))
        X += list(x)
        Y += list(y)
        M += list(np.ones(len(x))*SB[i])
        M_e += list(np.ones(len(x))*SB_e[i])

    plt.scatter(X,Y,c = M)
    plt.savefig('%sellipsemodelpoints_%s.jpg' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name))
    plt.close()
    
    smooth = SmoothBivariateSpline(X,Y,M,w = 1/np.array(M_e), grid = False)

    Model = np.zeros(IMG.shape, dtype = np.float32)
    ranges = [[max(0,int(results['center']['x']-R[-1]-2)), min(IMG.shape[1],int(results['center']['x']+R[-1]+2))],
              [max(0,int(results['center']['y']-R[-1]-2)), min(IMG.shape[0],int(results['center']['y']+R[-1]+2))]]
    XX, YY = np.meshgrid(np.arange(ranges[0][1] - ranges[0][0], dtype = float), np.arange(ranges[1][1] - ranges[1][0], dtype = float))
    S = deepcopy(XX.shape)
    XX -= results['center']['x'] - float(ranges[0][0])
    YY -= results['center']['y'] - float(ranges[1][0])
    
    MM = []
    L = len(XX.ravel())
    for i in range(0,L, 100):
        MM += list(smooth(XX.ravel()[i:min(i+100,L)], YY.ravel()[i:min(i+100,L)])[0])

    MM = np.array(MM)
    MM = np.reshape(MM,(S[0],S[1]))
    MM = 10**(-(MM - zeropoint - 5*np.log10(pixscale))/2.5)
    XX, YY = (XX*np.cos(-PA[-1]) - YY*np.sin(-PA[-1]), XX*np.sin(-PA[-1]) + YY*np.cos(-PA[-1]))
    YY /= 1 - ellip[-1]
    RR = np.sqrt(XX**2 + YY**2)
    del XX
    del YY
    MM[RR > (R[-1]*0.99)] = 0
    
    Model[ranges[1][0]:ranges[1][1],ranges[0][0]:ranges[0][1]] = MM
    print(Model.shape)
    header = fits.Header()
    hdul = fits.HDUList([fits.PrimaryHDU(header=header),
                         fits.ImageHDU(Model)])
    
    hdul.writeto('%s%s_model.fits' % (kwargs['plotpath'] if 'plotpath' in kwargs else '', name), overwrite = True)
    
    return {'ellipse model': Model}
