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

If you have difficulty running AutoProf, it is possible that one of these dependencies is not in its latest (Python3) version and you should try updating.

### Basic Install

1. Download the package from: https://github.com/ConnorStoneAstro/AutoProf
1. Set an environment variable and alias the autoprof function. To make this permanent, include these lines in your .bashrc file (or equivalent for your OS). 
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

If you get *Permission Denied*, it is possible that the file is not listed as exicutable and you need to run:
```bash
cd /path/to/AutoProf/
chmod 777 autoprof.py
```

For other issues contact connor.stone@queensu.ca for help. The code has been tested on Linux Mint and Mac machines.

# Using AutoProf

### Getting Started On A Single Image

The fastest way to run AutoProf for yourself will be for a single image.
The steps below describe how to get AutoProf to run on an image that you provide.
To run in batch mode for many images there isn't much to change, but that will be described later.

1. Copy the *config.py.example* script to the directory with your image. And remove the *.example* part from the filename.
1. In the config file, edit the following lines:
    ```python
    pixscale = # your image scale in arcsec/pix
    image_file = # filename of your image
    ```
1. Run AutoProf on the configuration file:
    ```bash
    autoprof config.py
    ```
1. Check the .prof file for the surface brightness profile.
Check the .aux file for extra information, including checks on the success of the fit.
Check the .log file for messages about the progress of the fit, which are updated throughout the fitting procedure.
Also, look at the diagnostic plots to see if the fit appears to have worked.

### Other Processing Modes

There are 4 main processing modes for AutoProf: image, image list, forced image, forced image list.
The subsections below will outline how to use each mode.

#### Running AutoProf In Batch Mode

Running AutoProf in batch mode is relatively simple once you have applied it to a single image.
You must modify the *process_mode* command to:
```python
process_mode = 'image list'
```
Then anything which you think should be specified for each galaxy should be a list, instead of a single value.
For example, the *image_file* variable should now be a list of image files.
If it doesn't need to be different for each galaxy, then simply leave the argument as a single value.
For example, the *pixscale* variable can be left as a float value and AutoProf will use that same value for all images.
Also unique to batch processing is the availability of parallel processing with the *n_procs* variable.
Since image analysis is an "embarrassingly parallel problem" AutoProf can analyze many images simultaneously.
It is suggested that you set *n_procs* equal to the number of processors you have, although you may need to experiment.
Especially if you don't have much ram, this may be the limiting factor.

Note that AutoProf has a list of arguments that it is expecting (see *List Of AutoProf Arguments* for a full list) and it only checks for those.
You can therefore make any variables you need in the config file to construct your list of image files so long as they don't conflict with any of the expected AutoProf arguments.

#### Forced Photometry

Forced photometry allows one to take an isophotal solution from one image and apply it (kind of) blindly to another image.
This can be used for multiband images from the same telescope, or between telescopes.
One may need to adjust for different pixel scales, or have to re-center between images, but the ultimate goal is to apply the same ellipticity and position angle profile to the galaxy in each band.
Running forced photometry is very similar to image processing.
You will, however, need the .prof and .aux files from an AutoProf run in order to try out the forced photometry.
Once you have that, you may follow these basic instructions to run forced photometry on a single image:

1. Copy the *config.py.example* script to the directory with your image. And remove the *.example* part from the filename.
1. In the config file, edit the following lines:
    ```python
    process_mode = 'forced image'
    pixscale = # your image scale in arcsec/pix
    image_file = # filename of your image
    forcing_profile = # filename for the .prof output
    ```
1. Run AutoProf on the configuration file:
    ```bash
    autoprof config.py
    ```
1. Check the .prof file for the surface brightness profile.
Check the .aux file for extra information, including checks on the success of the fit.
Check the .log file for messages about the progress of the fit, which are updated throughout the fitting procedure.
Also, look at the diagnostic plots to see if the fit appears to have worked.

To run forced photometry in batch mode is very similar to running image processing in batch mode.
Modify the *process_mode* variable to 'forced image list', then you must make *image_file* and *forcing_profile* into lists with the matching images and profiles.
And of course, any other arguments can be made into lists as well if appropriate.

### List Of AutoProf Arguments

This is a list of all arguments that AutoProf will check for and what they do.
In your config file, do not use any of these names unless you intend for AutoProf to interpret those values in it's image processing pipeline.

- pixscale: pixel scale in arcsec/pixel (float)
- image_file: path to fits file with image data (string)
- saveto: path to directory where final profile should be saved (string)
- name: name to use for the galaxy, this will be the name used in output files and in the log file (string)
- process_mode: analysis mode for AutoProf to run in (string)
- n_procs: number of processes to create when running in batch mode (int)
- overflowval: flux value that corresponds to an overflow pixel, used to identify bad pixels and mask them (float)
- mask_file: path to fits file which is a mask for the image. Must have the same dimensions as the main image (string)
- savemask: indicates if the star mask should be saved after fitting (bool)
- autodetectoverflow: Will try to guess the pixel saturation flux value from the mode
   		       in the image. In principle if all overflow pixels have the same
		       value then it would show up as the mode, but this is not
		       guaranteed (bool).
- plotpath: Path to file where diagnostic plots should be written, see also "doplot" (string)
- forced_recenter: when doing forced photometry indicates if AutoProf should re-calculate the galaxy center in the image (bool)
- doplot: Generate diagnostic plots during processing (bool).
- hdulelement: index for hdul of fits file where image exists (int).
- given_center: user provided center for isophote fitting. Center should be formatted as:
		{'x':float, 'y': float}, where the floats are the center coordinates in pixels. Also see *fit_center* (dict)
- fit_center: indicates if AutoProf should attempt to find the center. It will start at the center of the image unless *given_center* is provided
  	      in which case it will start there. This argument is ignored for forced photometry, in the event that a *given_center* is provided,
	      AutoProf will automatically use that value, if not given then it will read from the .aux file (bool)
- scale: growth scale when fitting isophotes, not the same as "sample---scale" (float)
- samplegeometricscale: growth scale for isophotes when sampling for the final output profile.
                         Used when sampling geometrically (float)
- samplelinearscale: growth scale (in pixels) for isophotes when sampling for the final output
                      profile. Used when sampling linearly (float)
- samplestyle: indicate if isophote sampling radii should grow linearly or geometrically. Can
                also do geometric sampling at the center and linear sampling once geometric step
		size equals linear (string, ['linear', 'geometric', 'geometric-linear'])
- sampleinitR: Starting radius (in pixels) for isophote sampling from the image. Note that
   		a starting radius of zero is not advised. (float)
- sampleendR: End radius (in pixels) for isophote sampling from the image (float)
- new_pipeline_functions: Allows user to set functions for the AutoProf pipeline analysis. See *Modifying Pipeline Functions* for more information (dict)
- new_pipieline_steps: Allows user to change the AutoProf analysis pipeline by adding, removing, or re-ordering steps. See *Modifying Pipeline Steps* for more information (list)

# How Does AutoProf Work?

More to come

### Background

Find peak of noise flux pedestal

### PSF

IRAF star finder, median fwhm

### Star Masking

IRAF star finder, block based on fwhm

### Centering

Follow hill climbing first fft coefficient for isophotal rings

### Global Isophote Fitting

Using circular ellipses, determine position angle using phase of 2nd fft coefficient. Fit ellipse which minimizes amplitude of 2nd fft coefficient relative to median flux on isophote

### Isophotal Fitting

Starting with global fit, randomly update isophotes and individually minimize relative 2nd fft coefficient amplitude, plus regularization term.

### Isophotal Profile Extraction

Using photutils, median filter flux values along isophote.

### Checking Isophotal Solution

Check for large flux variation along isophote. Check for large 2nd fft coefficient values despite minimization procedure. Check for large disagreement in integrated curve of growth and pixel summed curve of growth.


# Advanced Usage

### Modifying Pipeline Functions

This is done with the *new_pipeline_functions* argument, which is formatted as a dictionary with string keys and functions as values.
In this way you can alter the functions used by AutoProf in it's pipeline.

**This is hard to do right**

### Modifying Pipeline Steps

This is done with the *new_pipeline_steps* argument, which is formatted as a list of strings which tells AutoProf what order to run it's pipeline functions.
In this way you can alter the order of operations used by AutoProf in it's pipeline.

**This is hard to do right**

