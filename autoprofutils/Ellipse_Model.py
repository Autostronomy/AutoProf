import numpy as np
from astropy.io import fits
from scipy.interpolate import SmoothBivariateSpline, interp2d, Rbf, UnivariateSpline
from copy import deepcopy
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import AddLogo, autocmap, LSBImage
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

def EllipseModel_Fix(IMG, results, options):

    zeropoint = options['ap_zeropoint'] if 'ap_zeropoint' in options else 22.5
    pa = results['init pa']
    eps = results['init ellip']
    
    CHOOSE = np.array(results['prof data']['SB_e']) < 0.3
    R = np.array(results['prof data']['R'])[CHOOSE]/options['ap_pixscale']
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
    MM = 10**(-(MM - zeropoint - 5*np.log10(options['ap_pixscale']))/2.5)
    MM[RR > R[-1]] = 0
    Model[ranges[1][0]:ranges[1][1],ranges[0][0]:ranges[0][1]] = MM

    header = fits.Header()
    hdul = fits.HDUList([fits.PrimaryHDU(header=header),
                         fits.ImageHDU(Model)])
    
    hdul.writeto('%s%s_fixmodel.fits' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), overwrite = True)


    if 'ap_doplot' in options and options['ap_doplot']:
        plt.figure(figsize = (7,7))
        plt.imshow(np.clip(Model[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]],a_min = 0, a_max = None),
                   origin = 'lower', cmap = autocmap, norm = ImageNormalize(stretch=LogStretch(), clip = False))
        plt.axis('off')
        plt.tight_layout()
        if not ('ap_nologo' in options and options['ap_nologo']):
            AddLogo(plt.gcf())
        plt.savefig('%sellipsemodel_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
        plt.close()
        
        residual = IMG[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]] - results['background'] - Model[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]]
        plt.figure(figsize = (7,7))
        plt.imshow(residual, origin = 'lower', cmap = 'PuBu',
                   vmin = np.quantile(residual, 0.0001), vmax = 0)
        autocmap.set_under('k', alpha=0)
        plt.imshow(np.clip(residual,a_min = 0, a_max = np.quantile(residual,0.9999)),
                   origin = 'lower', cmap = autocmap, norm = ImageNormalize(stretch=LogStretch(), clip = False),
                   interpolation = 'none', clim = [1e-5, None])        
        plt.axis('off')
        plt.tight_layout()
        if not ('ap_nologo' in options and options['ap_nologo']):
            AddLogo(plt.gcf())
        plt.savefig('%sellipseresidual_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
        plt.close()
    
    return IMG, {'ellipse model': Model}
    

# def EllipseModel_General(IMG, results, options):
    
#     zeropoint = options['ap_zeropoint'] if 'ap_zeropoint' in options else 22.5
    
#     CHOOSE = np.array(results['prof data']['SB_e']) < 0.5
#     R = np.array(results['prof data']['R'])[CHOOSE]/options['ap_pixscale']
#     SB = np.array(results['prof data']['SB'])[CHOOSE]
#     SB_e = np.clip(np.array(results['prof data']['SB_e'])[CHOOSE], a_min = 1e-3, a_max = None)
#     PA = np.array(results['prof data']['pa'])[CHOOSE]*np.pi/180
#     ellip = np.array(results['prof data']['ellip'])[CHOOSE]
    
#     X = []
#     Y = []
#     XY_R = []
#     M = []
#     M_e = []
#     for i in range(len(R)):
#         N = max(4,int(R[i]))
#         theta = np.linspace(0, 2*np.pi*(1 - 1/N), N)
#         x = R[i]*np.cos(theta)
#         y = R[i]*(1 - ellip[i])*np.sin(theta)
#         x,y = (x*np.cos(PA[i]) - y*np.sin(PA[i]), x*np.sin(PA[i]) + y*np.cos(PA[i]))
#         XY_R += list(np.sqrt(x**2 + y**2))
#         X += list(x)
#         Y += list(y)
#         M += list(np.ones(len(x))*SB[i])
#         M_e += list(np.ones(len(x))*SB_e[i])

#     XY_R = np.array(XY_R)
#     X = np.array(X)
#     Y = np.array(Y)
#     M = np.array(M)
    
#     ranges = [[max(0,int(results['center']['x']-R[-1]-2)), min(IMG.shape[1],int(results['center']['x']+R[-1]+2))],
#               [max(0,int(results['center']['y']-R[-1]-2)), min(IMG.shape[0],int(results['center']['y']+R[-1]+2))]]
#     XX, YY = np.meshgrid(np.arange(ranges[0][1] - ranges[0][0], dtype = float), np.arange(ranges[1][1] - ranges[1][0], dtype = float))
#     S = deepcopy(XX.shape)
#     XX -= results['center']['x'] - float(ranges[0][0])
#     YY -= results['center']['y'] - float(ranges[1][0])
    
#     # MM = np.zeros(XX.shape)
#     # for i in range(XX.shape[0]):
#     #     CHOOSE = abs(Y - YY[i,int(YY.shape[1]/2)]) < (10*results['psf fwhm'])
#     #     K = -((XX[i,:].reshape(XX.shape[1],-1) - X[CHOOSE])**2 + (YY[i,:].reshape(XX.shape[1],-1) - Y[CHOOSE])**2)/((1+np.sqrt(XY_R[CHOOSE]))*results['psf fwhm']/4)**2
#     #     K = np.exp(K - (np.max(K,axis = 1)).reshape(K.shape[0],-1))
#     #     MM[i,:] = np.sum(M[CHOOSE]*K,axis = 1) / np.sum(K,axis = 1)

#     sp_interp = Rbf(X,Y,M, smooth = 1, function = 'linear') #interp2d(X,Y,M, fill_value = 0)
#     MM = sp_interp(XX,YY) #np.reshape(sp_interp(np.arange(ranges[0][1] - ranges[0][0], dtype = float), np.arange(ranges[1][1] - ranges[1][0], dtype = float)), XX.shape)
    
#     MM = 10**(-(MM - zeropoint - 5*np.log10(options['ap_pixscale']))/2.5)
    
#     XX, YY = (XX*np.cos(-PA[-1]) - YY*np.sin(-PA[-1]), XX*np.sin(-PA[-1]) + YY*np.cos(-PA[-1]))
#     YY /= 1 - ellip[-1]
#     RR = np.sqrt(XX**2 + YY**2)
#     MM[RR > R[-2]] = 0
    
#     Model = np.zeros(IMG.shape, dtype = np.float32)
#     Model[ranges[1][0]:ranges[1][1],ranges[0][0]:ranges[0][1]] = MM
    
#     header = fits.Header()
#     hdul = fits.HDUList([fits.PrimaryHDU(header=header),
#                          fits.ImageHDU(Model)])
    
#     hdul.writeto('%s%s_model.fits' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), overwrite = True)

#     if 'ap_doplot' in options and options['ap_doplot']:
#         plt.figure(figsize = (7,7))
#         plt.imshow(np.clip(Model[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]],a_min = 0, a_max = None),
#                    origin = 'lower', cmap = autocmap, norm = ImageNormalize(stretch=LogStretch(), clip = False))
#         plt.axis('off')
#         plt.tight_layout()
#         if not ('ap_nologo' in options and options['ap_nologo']):
#             AddLogo(plt.gcf())
#         plt.savefig('%sellipsemodel_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
#         plt.close()
        
#         residual = IMG[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]] - results['background'] - Model[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]]
#         plt.figure(figsize = (7,7))
#         plt.imshow(residual, origin = 'lower', cmap = 'PuBu',
#                    vmin = np.quantile(residual, 0.0001), vmax = 0)
#         autocmap.set_under('k', alpha=0)
#         plt.imshow(np.clip(residual,a_min = 0, a_max = np.quantile(residual,0.9999)),
#                    origin = 'lower', cmap = autocmap, norm = ImageNormalize(stretch=LogStretch(), clip = False),
#            interpolation = 'none', clim = [1e-5, None])        
#         plt.axis('off')
#         plt.tight_layout()
#         if not ('ap_nologo' in options and options['ap_nologo']):
#             AddLogo(plt.gcf())
#         plt.savefig('%sellipseresidual_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
#         plt.close()

#     if 'ap_paperplots' in options and options['ap_paperplots']:    
#         plt.figure(figsize = (7,7))
#         LSBImage(IMG[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]] - results['background'], results['background noise'])
#         plt.axis('off')
#         plt.tight_layout()
#         plt.savefig('%sclearimage_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
#         plt.close()

    
#     return IMG, {'ellipse model': Model}

def EllipseModel_General(IMG, results, options):

    zeropoint = options['ap_zeropoint'] if 'ap_zeropoint' in options else 22.5
    
    CHOOSE = np.array(results['prof data']['SB_e']) < 0.5
    R = np.array(results['prof data']['R'])[CHOOSE]/options['ap_pixscale']

    sb = UnivariateSpline(R, np.array(results['prof data']['SB'])[CHOOSE], ext = 3, s = 0)
    pa = UnivariateSpline(R, np.array(results['prof data']['pa'])[CHOOSE]*np.pi/180, ext = 3, s = 0)
    q = UnivariateSpline(R, 1 - np.array(results['prof data']['ellip'])[CHOOSE], ext = 3, s = 0)
    
    ranges = [[max(0,int(results['center']['x']-R[-1]-2)), min(IMG.shape[1],int(results['center']['x']+R[-1]+2))],
              [max(0,int(results['center']['y']-R[-1]-2)), min(IMG.shape[0],int(results['center']['y']+R[-1]+2))]]

    XX, YY = np.meshgrid(np.arange(ranges[0][1] - ranges[0][0], dtype = float), np.arange(ranges[1][1] - ranges[1][0], dtype = np.float32))
    XX -= results['center']['x'] - float(ranges[0][0])
    YY -= results['center']['y'] - float(ranges[1][0])
    MM = np.zeros(XX.shape, dtype = np.float32)
    Prox = np.zeros(XX.shape, dtype = np.float32) + np.inf

    for r in reversed(np.logspace(np.log10(R[0]/2),np.log10(R[-1]),len(R)*2)):
        RR = np.sqrt((XX*np.cos(-pa(r)) - YY*np.sin(-pa(r)))**2 + ((XX*np.sin(-pa(r)) + YY*np.cos(-pa(r)))/np.clip(q(r),a_min = 0.03,a_max = 1))**2)
        D = np.abs(RR - r)
        CLOSE = D < Prox
        if np.any(CLOSE):
            MM[CLOSE] = sb(RR[CLOSE])
            Prox[CLOSE] = D[CLOSE]
        
    MM = 10**(-(MM - zeropoint - 5*np.log10(options['ap_pixscale']))/2.5)
    RR = np.sqrt((XX*np.cos(-pa(R[-1])) - YY*np.sin(-pa(R[-1])))**2 + ((XX*np.sin(-pa(R[-1])) + YY*np.cos(-pa(R[-1])))/np.clip(q(R[-1]),a_min = 0.03,a_max = 1))**2)
    MM[RR > R[-1]] = 0
    
    Model = np.zeros(IMG.shape, dtype = np.float32)
    Model[ranges[1][0]:ranges[1][1],ranges[0][0]:ranges[0][1]] = MM
    
    header = fits.Header()
    hdul = fits.HDUList([fits.PrimaryHDU(header=header),
                         fits.ImageHDU(Model)])
    
    hdul.writeto('%s%s_model.fits' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), overwrite = True)

    if 'ap_doplot' in options and options['ap_doplot']:
        ranges = [[max(0,int(results['center']['x']-R[-1]*1.2)), min(IMG.shape[1],int(results['center']['x']+R[-1]*1.2))],
                  [max(0,int(results['center']['y']-R[-1]*1.2)), min(IMG.shape[0],int(results['center']['y']+R[-1]*1.2))]]
        plt.figure(figsize = (7,7))
        plt.imshow(np.clip(Model[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]],a_min = 0, a_max = None),
                   origin = 'lower', cmap = autocmap, norm = ImageNormalize(stretch=LogStretch(), clip = False))
        plt.axis('off')
        plt.tight_layout()
        if not ('ap_nologo' in options and options['ap_nologo']):
            AddLogo(plt.gcf())
        plt.savefig('%sellipsemodel_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
        plt.close()
        
        residual = IMG[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]] - results['background'] - Model[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]]
        plt.figure(figsize = (7,7))
        plt.imshow(residual, origin = 'lower', cmap = 'PuBu',
                   vmin = np.quantile(residual, 0.0001), vmax = 0)
        autocmap.set_under('k', alpha=0)
        plt.imshow(np.clip(residual,a_min = 0, a_max = np.quantile(residual,0.9999)),
                   origin = 'lower', cmap = autocmap, norm = ImageNormalize(stretch=LogStretch(), clip = False),
           interpolation = 'none', clim = [1e-5, None])        
        plt.axis('off')
        plt.tight_layout()
        if not ('ap_nologo' in options and options['ap_nologo']):
            AddLogo(plt.gcf())
        plt.savefig('%sellipseresidual_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
        plt.close()

    if 'ap_paperplots' in options and options['ap_paperplots']:    
        plt.figure(figsize = (7,7))
        LSBImage(IMG[ranges[1][0]: ranges[1][1], ranges[0][0]: ranges[0][1]] - results['background'], results['background noise'])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('%sclearimage_%s.jpg' % (options['ap_plotpath'] if 'ap_plotpath' in options else '', options['ap_name']), dpi = options['ap_plotdpi'] if 'ap_plotdpi'in options else 300)
        plt.close()

    return IMG, {'ellipse model': Model}
