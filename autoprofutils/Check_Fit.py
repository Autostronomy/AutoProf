import numpy as np
from scipy.stats import iqr
from scipy.fftpack import fft, ifft
import logging
import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.SharedFunctions import _iso_extract, _x_to_pa, _x_to_eps, _inv_x_to_eps, _inv_x_to_pa


def Check_Fit_Simple(IMG, pixscale, name, results, **kwargs):
    tests = {}
    # subtract background from image during processing
    dat = IMG - results['background']['background']
    
    # Compare integrated total magnitude with summed total magnitude
    try:
        SB = np.array(results['isophoteextract']['data']['SB'])
        SBe = np.array(results['isophoteextract']['data']['SB_e'])
        Mint = np.array(results['isophoteextract']['data']['totmag'])
        Msum = np.array(results['isophoteextract']['data']['totmag_direct'])
        CHOOSE = np.logical_and(SB < 99, SBe < 0.1)
        if np.sum(np.abs(Mint[CHOOSE][-4:] - Msum[CHOOSE][-4:]) > 0.2) > 2:
            logging.warning('%s: Possible failed fit! Inconsistent results for curve of growth, bad fit or foreground star' % name)
            tests['curve of growth consistency'] = False
        else:
            tests['curve of growth consistency'] = True
    except:
        logging.info('%s: Check fit could not check SB profile consistency')
    return tests

def Check_Fit_IQR(IMG, pixscale, name, results, **kwargs):

    tests = {}
    # subtract background from image during processing
    dat = IMG - results['background']['background']

    # Compare variability of flux values along isophotes
    ######################################################################
    use_center = results['isophotefit']['center'] if 'center' in results['isophotefit'] else results['center']
    count_variable = 0
    count_initrelative = 0
    f2_compare = []
    f1_compare = []
    for i in range(len(results['isophotefit']['R'])):
        init_isovals = _iso_extract(dat,results['isophotefit']['R'][i],results['isophoteinit']['ellip'],
                                    results['isophoteinit']['pa'],use_center)
        isovals = _iso_extract(dat,results['isophotefit']['R'][i],results['isophotefit']['ellip'][i],
                               results['isophotefit']['pa'][i],use_center)
        coefs = fft(np.clip(isovals, a_max = np.quantile(isovals,0.85), a_min = None))

        if np.median(isovals) < (iqr(isovals)-results['background']['noise']):
            count_variable += 1
        if ((iqr(isovals) - results['background']['noise'])/(np.median(isovals)+results['background']['noise'])) > (iqr(init_isovals)/(np.median(init_isovals)+results['background']['noise'])):
            count_initrelative += 1
        f2_compare.append(np.sum(np.abs(coefs[[2,4]]))/np.abs(coefs[0]))
        f1_compare.append(np.abs(coefs[1])/np.abs(coefs[0]))
        
    f1_compare = np.array(f1_compare)
    f2_compare = np.array(f2_compare)
    if count_variable > (0.2*len(results['isophotefit']['R'])):
        logging.warning('%s: Possible failed fit! flux values highly variable along isophotes' % name)
        tests['isophote variability'] = False
    else:
        tests['isophote variability'] = True
    if count_initrelative > (0.5*len(results['isophotefit']['R'])):
        logging.warning('%s: Possible failed fit! flux values highly variable relative to initialization' % name)
        tests['initial fit compare'] = False
    else:
        tests['initial fit compare'] = True
    if np.sum(f2_compare > 0.3) > 2 or np.sum(f2_compare > 0.2) > (0.3*len(results['isophotefit']['R'])) or np.sum(f2_compare > 0.1) > (0.8*len(results['isophotefit']['R'])):
        logging.warning('%s: Possible failed fit! poor convergence of FFT coefficients' % name)
        tests['FFT coefficients'] = False
    else:
        tests['FFT coefficients'] = True
    if np.sum(f1_compare > 0.3) > 2 or np.sum(f1_compare > 0.2) > (0.3*len(results['isophotefit']['R'])) or np.sum(f1_compare > 0.1) > (0.8*len(results['isophotefit']['R'])):
        logging.warning('%s: Possible failed fit! possible failed center or lopsided galaxy' % name)
        tests['Light symmetry'] = False
    else:
        tests['Light symmetry'] = True
        
    # Compare integrated total magnitude with summed total magnitude
    try:
        SB = np.array(results['isophoteextract']['data']['SB'])
        SBe = np.array(results['isophoteextract']['data']['SB_e'])
        Mint = np.array(results['isophoteextract']['data']['totmag'])
        Msum = np.array(results['isophoteextract']['data']['totmag_direct'])
        CHOOSE = np.logical_and(SB < 99, SBe < 0.1)
        if np.sum(np.abs(Mint[CHOOSE][-4:] - Msum[CHOOSE][-4:]) > 0.2) > 2:
            logging.warning('%s: Possible failed fit! Inconsistent results for curve of growth, bad fit or foreground star' % name)
            tests['curve of growth consistency'] = False
        else:
            tests['curve of growth consistency'] = True
    except:
        logging.info('%s: Check fit could not check SB profile consistency')
    
    
    return tests
