import numpy as np
import sys
import os
sys.path.append(os.environ['AUTOPROF'])

def Crop(IMG, results, options):
    """
    Crop the edges of an image about a center point.

    If a 'Center' method has been applied before in the pipeline we use that to
    define the galaxy center, otherwise we define the center as half the image.

    ap_cropto states the new size of the image after cropping. 
    We default to 512*512.
    """

    if 'center' in results:
        x = results['center']['x']
        y = results['center']['y']
        center = np.array((x, y)).astype('int')
    else:
        center = np.array(IMG.shape)//2

    cropto = options['ap_cropto'] if 'ap_cropto' in options else (512, 512)

    IMG = IMG[center[0] - cropto[0]//2:center[0] + cropto[0]//2,
              center[1] - cropto[1]//2:center[1] + cropto[1]//2]

    return IMG, {}

def MinMaxNorm(IMG, results, options):
    """
    Perform a min max norm on the image, normalising all pixels to values
    within the closed interval [0, 1].
    """
    IMG = (IMG - np.min(IMG))/(np.max(IMG) - np.min(IMG))

    return IMG, {}

def PercentileClip(IMG, results, options):
    """
    Perform an upper bound percentile clip on the image.

    ap_percentileclip states the percentile to clip pixels at.
    We default to 99.9%.
    """
    percentile = options['ap_percentileclip'] if 'ap_percentileclip' in options else 99.9
    p = np.percentile(IMG, percentile)
    IMG = np.where(IMG <= p, IMG, p)

    return IMG, {'clipped at': p}
