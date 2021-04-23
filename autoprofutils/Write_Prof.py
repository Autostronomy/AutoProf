from astropy.io import fits
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
    with open(saveto + options['ap_name'] + '_AP.aux', 'w') as f:
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
    with open(saveto + options['ap_name'] + '_AP.prof', 'w') as f:
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
