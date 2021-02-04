Author: Connor Stone
Date: 9 June 2020

This README explains basic use of the autoprof code for extracting photometry
from galaxy images. The photometry is extracted via elliptical isophotes which
are fit with variable position angle and ellipticity, but a fixed center. The
isophotes are found by minimizing low order, even, Fourier coefficients along
with regularization applied to minimize chaotic swings of isophote parameters.
The regularization penalizes isophotes for having PA and ellipticity values
very different from their neighbours.

Requirements: numpy, scipy, matplotlib, astropy, photutils

The primary interface is "Pipeline.py" which contains the "Isophote_Pipeline"
class. A basic use of the class would be:
PIPELINE = Isophote_Pipeline(loggername = 'ImJustTesting.log')
PIPELINE.Process_Image(IMG = <path to image>,
		       seeing = <seeing on observation night>,
		       pixscale = <angular size of pixels>,
		       saveto = <file path to save profile to>)

This will run the standard analysis pipeline on the given image. To alter the
pipeline, you can give functions as arguments to the pipeline class init
function. The replaceable sections of the pipeline are described in the init
function for the pipeline class, for example set_background_f will set the
function that computes the background of the image. Each subprocess in the
pipeline has multiple functions already, and you can write more so long as
they have the same input and output format.

To run PIPELINE.Process_List you provide the same information, just collected
into lists instead of single entries. The pipeline object will parallelize
the PIPELINE.Process_Image function for you with the specified number of
processes (default 1).

Each function describes its arguments in its description, however there are some
extra Kwargs that can be passed to Process_Image that will impact various
elements of the processing and override default behaviour:
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