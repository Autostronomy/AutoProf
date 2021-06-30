from astropy.io import fits
from astropy.table import Table
import numpy as np
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import PA_shift_convention
from datetime import datetime

def WriteProf(IMG, results, options):
    """
    Writes the photometry information for disk given a photutils isolist object
    """
    
    saveto = options['ap_saveto'] if 'ap_saveto' in options else './'
    
    # Write aux file
    with open(os.path.join(saveto, options['ap_name'] + '.aux'), 'w') as f:
        # write profile info
        f.write('written on: %s\n' % str(datetime.now()))
        f.write('name: %s\n' % str(options['ap_name']))

        for r in sorted(results.keys()):
            if 'auxfile' in r:
                f.write(results[r] + '\n')
        for k in sorted(options.keys()):
            if k == 'ap_name':
                continue
            f.write('option %s: %s\n' % (k,str(options[k])))
            
    # Write the profile
    delim = options['ap_delimiter'] if 'ap_delimiter' in options else ','
    delim = options['ap_delimiter'] if 'ap_delimiter' in options else ','
    results['prof data']['pa'] = list(PA_shift_convention(np.array(results['prof data']['pa']), deg = True))
    T = Table(data = results['prof data'], names = results['prof header'])
    if 'ap_profile_format' in options and options['ap_profile_format'].lower() == 'fits':
        T.meta['UNITS'] = delim.join(results['prof units'][h] for h in results['prof header'])
        T.write(os.path.join(saveto, options['ap_name'] + '_prof.fits'), format = 'fits', overwrite = True)
    else:
        T.write(os.path.join(saveto, options['ap_name'] + '.prof'), format = 'ascii.commented_header',
                delimiter = delim, overwrite = True,
                comment = '# ' + delim.join(results['prof units'][h] for h in results['prof header']) + '\n')
    results['prof data']['pa'] = list(PA_shift_convention(np.array(results['prof data']['pa']), deg = True))
                
    # Write the mask data, if provided
    if 'mask' in results and (not results['mask'] is None) and 'ap_savemask' in options and options['ap_savemask']:
        header = fits.Header()
        header['IMAGE 1'] = 'star mask'
        header['IMAGE 2'] = 'overflow values mask'
        hdul = fits.HDUList([fits.PrimaryHDU(header=header),
                             fits.ImageHDU(results['mask'].astype(int)),
                             fits.ImageHDU(results['overflow mask'].astype(int))])
        hdul.writeto(saveto + options['ap_name'] + '_mask.fits', overwrite = True)
        sleep(1)
        # Zip the mask file because it can be large and take a lot of memory, but in principle
        # is very easy to compress
        os.system('gzip -fq '+ saveto + options['ap_name'] + '_mask.fits')
    return IMG, {}
