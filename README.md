Author: Connor Stone
Date: 4 Feb 2021

# Introduction

This README explains basic use of the autoprof code for extracting photometry
from galaxy images. The photometry is extracted via elliptical isophotes which
are fit with variable position angle and ellipticity, but a fixed center. The
isophotes are found by minimizing low order Fourier coefficients along
with regularization applied to minimize chaotic swings of isophote parameters.
The regularization penalizes isophotes for having PA and ellipticity values
very different from their neighbours.

# Installation

### Requirements

numpy, scipy, matplotlib, astropy, photutils

### Basic Install

1. Download the package from: https://github.com/ConnorStoneAstro/AutoProf
1. Set an environment variable and alias the autoprof function:
    ```bash
    export AUTOPROF='/path/to/AutoProf/'
    alias autoprof='/path/to/AutoProf/autoprof.py'
    ```
1. Run the test case to see that all is well:
    ```bash
    cd /path/to/AutoProf/
    autoprof test/test_config.py
    ```

### Issues

Contact connor.stone@queensu.ca if you experience issues. The code has been
tested on Linux Mint and Mac machines.

# Using AutoProf

### Getting Started

### Optional Commands
- autodetectoverflow: Will try to guess the pixel saturation flux value from the mode
   		       in the image. In principle if all overflow pixels have the same
		       value then it would show up as the mode, but this is not
		       guaranteed (bool).
- plotpath: Path to file where diagnostic plots should be written, see also "doplot" (string)
- doplot: Generate diagnostic plots during processing (bool).
- hdulelement: index for hdul of fits file where image exists (int).
- given_centers: user provided centers for isophote fitting, used with "GivenCenters" center
                  function (set_center_f). Should be a dictionary with galaxy names as each
		  key and the center as the value. Center should be formatted as:
		  {'x':float, 'y': float}, where the floats are the center coordinates in pixels
- scale: growth scale when fitting isophotes, not the same as "sample---scale" (float, (0,inf))
- samplegeometricscale: growth scale for isophotes when sampling for the final output profile.
                         Used when sampling geometrically (float, (0,inf))
- samplelinearscale: growth scale (in pixels) for isophotes when sampling for the final output
                      profile. Used when sampling linearly (float, (0,inf))
- samplestyle: indicate if isophote sampling radii should grow linearly or geometrically. Can
                also do geometric sampling at the center and linear sampling once geometric step
		size equals linear (string, ['linear', 'geometric', 'geometric-linear'])
- sampleinitR: Starting radius (in pixels) for isophote sampling from the image. Note that
   		a starting radius of zero is not advised. (float, (0, imgsize/2))
- sampleendR: End radius (in pixels) for isophote sampling from the image (float, (0,imgsize/2))
- sampleerrorlim: Surface Brightness uncertainty cutoff (in mag arcsec^-2) for profile
                extraction (float, (0,inf))