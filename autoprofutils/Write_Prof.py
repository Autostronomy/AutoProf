from astropy.io import fits
import numpy as np
import os

def WriteProf(IMG, results, **kwargs):
    """
    Writes the photometry information for disk given a photutils isolist object
    """
    
    saveto = kwargs['saveto'] if 'saveto' in kwargs else './'
    
    # Write aux file
    with open(saveto + kwargs['name'] + '.aux', 'w') as f:
        # write profile info
        f.write('name: %s\n' % str(kwargs['name']))
        f.write('pixel scale: %.3e arcsec/pix\n' % kwargs['pixscale'])
        if 'checkfit' in results:
            for k in results['checkfit'].keys():
                f.write('check fit %s: %s\n' % (k, 'pass' if results['checkfit'][k] else 'fail'))
        f.write('psf fwhm: %.3f pix\n' % (results['psf fwhm']))
        try:
            f.write('background: %.5e +- %.2e flux/pix, noise: %.5e flux/pix\n' % (results['background'], results['background uncertainty'], results['background noise']))
        except:
            pass
        use_center = results['center']
        f.write('center x: %.2f pix, y: %.2f pix\n' % (use_center['x'], use_center['y']))
        if 'init ellip_err' in results and 'init pa_err' in results:
            f.write('global ellipticity: %.3f +- %.3f, pa: %.3f +- %.3f deg\n' % (results['init ellip'], results['init ellip_err'],
                                                                                  PA_shift_convention(results['init pa'])*180/np.pi,
                                                                                  results['init pa_err']*180/np.pi))
        else:
            f.write('global ellipticity: %.3f, pa: %.3f deg\n' % (results['init ellip'], PA_shift_convention(results['init pa'])*180/np.pi))
        f.write('fit limit semi-major axis: %.2f pix\n' % results['fit R'][-1])
        if len(kwargs) > 0:
            for k in kwargs.keys():
                f.write('settings %s: %s\n' % (k,str(kwargs[k])))
            
    # Write the profile
    delim = kwargs['delimiter'] if 'delimiter' in kwargs else ','
    with open(saveto + kwargs['name'] + '.prof', 'w') as f:
        # Write profile header
        f.write(delim.join(results['prof header']) + '\n')
        if 'prof units' in results:
            f.write(delim.join(results['prof units'][h] for h in results['prof header']) + '\n')
        for i in range(len(results['prof data'][results['prof header'][0]])):
            line = []
            for h in results['prof header']:
                if h == 'pa':
                    line.append(results['prof format'][h] % PA_shift_convention(results['prof data'][h][i], deg = True))
                else:
                    line.append(results['prof format'][h] % results['prof data'][h][i])
            f.write(delim.join(line) + '\n')
                
    # Write the mask data, if provided
    if 'mask' in results and (not results['mask'] is None) and 'savemask' in kwargs and kwargs['savemask']:
        header = fits.Header()
        header['IMAGE 1'] = 'star mask'
        header['IMAGE 2'] = 'overflow values mask'
        hdul = fits.HDUList([fits.PrimaryHDU(header=header),
                             fits.ImageHDU(results['mask'].astype(int)),
                             fits.ImageHDU(results['overflow mask'].astype(int))])
        hdul.writeto(saveto + kwargs['name'] + '_mask.fits', overwrite = True)
        sleep(1)
        # Zip the mask file because it can be large and take a lot of memory, but in principle
        # is very easy to compress
        os.system('gzip -fq '+ saveto + kwargs['name'] + '_mask.fits')
    return IMG, {}
