import numpy as np
from scipy.stats import iqr
from scipy.fftpack import fft, ifft
import logging
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _iso_extract, _x_to_pa, _x_to_eps, _inv_x_to_eps, _inv_x_to_pa

def Check_Fit(IMG, results, options):
    """
    Check for failed fit with various measures.
    1) compare iqr of isophotes with that of a simple global fit
    2) compare iqr of isophotes with median flux to check for high variability
    3) measure signal in 2nd and 4th FFT coefficient which should be minimal
    4) measure signal in 1st FFT coefficient which should be minimal
    5) Compare integrated SB profile with simple flux summing for total magnitude
    """
    tests = {}
    # subtract background from image during processing
    dat = IMG - results['background']

    # Compare variability of flux values along isophotes
    ######################################################################
    use_center = results['center']
    count_variable = 0
    count_initrelative = 0
    f2_compare = []
    f1_compare = []
    checkson = {'R': results['fit R'] if 'fit R' in results else results['prof data']['R'],
                'pa': results['fit pa'] if 'fit pa' in results else results['prof data']['pa'],
                'ellip': results['fit ellip'] if 'fit ellip' in results else results['prof data']['ellip']}
    for i in range(len(checkson)): 
        init_isovals = _iso_extract(dat,checkson['R'][i],results['init ellip'], # fixme, use mask
                                    results['init pa'],use_center)
        isovals = _iso_extract(dat,checkson['R'][i],checkson['ellip'][i],
                               checkson['pa'][i],use_center)
        coefs = fft(np.clip(isovals, a_max = np.quantile(isovals,0.85), a_min = None))

        if np.median(isovals) < (iqr(isovals)-results['background noise']):
            count_variable += 1
        if ((iqr(isovals) - results['background noise'])/(np.median(isovals)+results['background noise'])) > (iqr(init_isovals)/(np.median(init_isovals)+results['background noise'])):
            count_initrelative += 1
        f2_compare.append(np.sum(np.abs(coefs[2]))/np.abs(coefs[0]))
        f1_compare.append(np.abs(coefs[1])/np.abs(coefs[0]))
        
    f1_compare = np.array(f1_compare)
    f2_compare = np.array(f2_compare)
    if count_variable > (0.2*len(checkson['R'])):
        logging.warning('%s: Possible failed fit! flux values highly variable along isophotes' % options['ap_name'])
        tests['isophote variability'] = False
    else:
        tests['isophote variability'] = True
    if count_initrelative > (0.5*len(checkson['R'])):
        logging.warning('%s: Possible failed fit! flux values highly variable relative to initialization' % options['ap_name'])
        tests['initial fit compare'] = False
    else:
        tests['initial fit compare'] = True
    if np.sum(f2_compare > 0.3) > 2 or np.sum(f2_compare > 0.2) > (0.3*len(checkson['R'])) or np.sum(f2_compare > 0.1) > (0.8*len(checkson['R'])):
        logging.warning('%s: Possible failed fit! poor convergence of FFT coefficients' % options['ap_name'])
        tests['FFT coefficients'] = False
    else:
        tests['FFT coefficients'] = True
    if np.sum(f1_compare > 0.3) > 2 or np.sum(f1_compare > 0.2) > (0.3*len(checkson['R'])) or np.sum(f1_compare > 0.1) > (0.8*len(checkson['R'])):
        logging.warning('%s: Possible failed fit! possible failed center or lopsided galaxy' % options['ap_name'])
        tests['Light symmetry'] = False
    else:
        tests['Light symmetry'] = True
        
    # Compare integrated total magnitude with summed total magnitude
    try:
        SB = np.array(results['prof data']['SB'])
        SBe = np.array(results['prof data']['SB_e'])
        Mint = np.array(results['prof data']['totmag'])
        Msum = np.array(results['prof data']['totmag_direct'])
        CHOOSE = np.logical_and(SB < 99, SBe < 0.1)
        if np.sum(np.abs(Mint[CHOOSE][-4:] - Msum[CHOOSE][-4:]) > 0.2) > 2:
            logging.warning('%s: Possible failed fit! Inconsistent results for curve of growth, bad fit or foreground star' % options['ap_name'])
            tests['curve of growth consistency'] = False
        else:
            tests['curve of growth consistency'] = True
    except:
        logging.info('%s: Check fit could not check SB profile consistency')
    
    res = {'checkfit': tests}
    for t in tests:
        res['auxfile checkfit %s' % t] = 'checkfit %s: %s' % (t, 'pass' if tests[t] else 'fail')
    return IMG, res
