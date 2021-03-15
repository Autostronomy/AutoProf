# AutoProf

This pipeline for non-parametric Galaxy image analysis was written with the goal
of making a tool that is easy for anyone to get started with, yet flexible
enough to prototype new ideas and accommodate advanced users. It was written
by [Connor Stone](https://connorjstone.com/) along with lots of testing/help from
[Nikhil Arora](https://orcid.org/0000-0002-3929-9316),
[Stephane Courteau](https://www.physics.queensu.ca/facultysites/courteau/),
[Simon Diaz Garcia](https://orcid.org/0000-0002-4662-1289),
and [Jean-Charles Cuillandre](https://www.cfht.hawaii.edu/~jcc/).

# Introduction

This README explains basic use of the AutoProf code for extracting photometry
from galaxy images. The photometry is extracted via elliptical isophotes which
are fit with variable position angle and ellipticity, but a fixed center. The
isophotes are found by minimizing low order Fourier coefficients along
with regularization applied to minimize chaotic swings of isophote parameters.
The regularization penalizes isophotes for having PA and ellipticity values
very different from their neighbours.

# Installation

### Requirements

numpy, scipy, matplotlib, astropy, photutils, scikit-learn

If you have difficulty running AutoProf, it is possible that one of these dependencies is not in its latest (Python3) version and you should try updating.

### Basic Install

1. Download the package from: https://github.com/ConnorStoneAstro/AutoProf
    ```bash
    cd /where/you/want/AutoProf/to/live/
    git clone git@github.com:ConnorStoneAstro/AutoProf.git
    ```
    If you are having difficulty cloning the package, it is also possible to download a zip file of the package from the github page.
1. Set an environment variable and alias the autoprof function. To make this permanent, include these lines in your .bashrc file (or equivalent for your OS). 
    ```bash
    export AUTOPROF='/path/to/AutoProf/'
    alias autoprof='/path/to/AutoProf/autoprof.py'
    ```
1. Run the test cases to see that all is well:
    ```bash
    cd /path/to/AutoProf/test/
    autoprof test_config.py
    autoprof test_forced_config.py Forced.log
    autoprof test_batch_config.py Batch.log
    ```
    This will test a basic AutoProf run on a single galaxy, forced photometry of the galaxy on itself, and batch photometry for multiple images (which are actually the same in this case) respectively.

### Issues

* If you get *Permission Denied*, it is possible that the file is not listed as executable and you need to run:
    ```bash
    cd /path/to/AutoProf/
    chmod 755 autoprof.py
    ```
* If you have everything set up, but are getting strange errors such as *ImportError: No module named photutils* even when photutils is already installed it is possible that your python3 installation lives somewhere unusual. Try executing:
    ```bash
    which python3
    ```
    to see where your python3 installation lives. If the result is something other than */usr/bin/python3* then you need to make a small edit to the *autoprof.py* file. In the first line make the change from:
    ```python
    #!/bin/bash/python3
    ```
    to instead be:
    ```python
    #!/wherever/your/python3/is/installed
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
This can be used for multi-band images from the same telescope, or between telescopes.
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
- zeropoint: Photometric zero point, AB magnitude is assumed if none given, corresponding to a zero point of 22.5 (float)
- delimiter: Delimiter character used to separate values in output profile. Will default to a comma (",") if not given (string)
- new_pipeline_functions: Allows user to set functions for the AutoProf pipeline analysis. See *Modifying Pipeline Functions* for more information (dict)
- new_pipeline_steps: Allows user to change the AutoProf analysis pipeline by adding, removing, or re-ordering steps. See *Modifying Pipeline Steps* for more information (list)

There is one argument that AutoProf can take in the command line, which is the name of the log file.
The log file stores information about everything that AutoProf is doing, this is useful for diagnostic purposes.
By default, AutoProf will name the log file *AutoProf.log*, if you wish to make it a different filename then add the filename when running AutoProf:
```bash
autoprof config.py newlogfilename.log
```

# How Does AutoProf Work?

At it's core AutoProf is a simple pipeline object that loads in an image, blindly runs a list of functions, and saves the resulting information.
It is equipped with a powerful default set of functions which can fit an isophotal solution to most galaxies, even with relatively complex features.
Below is a high level description for each function, in case something goes wrong this may help you troubleshoot the issue.

### Background

**pipeline label: background**

The default background calculation is done by searching for the "mode" of the pixel flux values.
First, the method extracts the border of the image, taking all pixels that are within 1/5th the image width of the edge.
Then it constructs a density profile in flux space and finds the peak.
This peak is used as the background level, a few rounds of sigma clipping are applied to remove bright signals before taking the background noise level (measured as an interquartile range).

Output format:
```python
{'background': , # flux value representing the background level
'background noise': # measure of scatter around the background level
}
```

### PSF

**pipeline label: psf**

Using the IRAF star finder wrapper from photutils, bright stars in the image are identified (at most 30).
An FFT is used to identify non-circular stars or artifacts which may have been picked up by IRAF.
Circular appertures are placed around the star until the background brightness is nearly reached.
The brightness of these apertures as a function of radius are fit with a Gaussian and the sigma is converted into a fwhm.


Output format:
```python
{'psf fwhm': # estimate of the fwhm of the PSF
}
```

### Centering

**pipeline label: center**

Depending on the specified parameters, this function will start at the center of the image or at a user specified center.
From the starting point, the function will create 10 circular isophotes out to 10 times the PSF size and sample flux values around each isophote.
An FFT is taken for the flux values around each circular isophote and the phase of the first FFT coefficient is used to determine a direction on the image of increasing brightness.
Taking the average direction, flux values are sampled from the current center out to 10 times the PSF.
A parabola is fit to the flux values and the center is then updated to the maximum of the parabola.
This is repeated until the update steps become negligible.
At this point, tiny random perturbations are used to fine tune the center.
The random perturbations continue until a minimum is found in FFT first coefficient magnitude.

Output format:
```python
{'center': {'x': , # x coordinate of the center (pix)
	    'y': } # y coordinate of the center (pix)
}
```

### Global Isophote Fitting

**pipeline label: isophoteinit**

A global position angle and ellipticity are fit in a two step process.
First, a series of circular isophotes are geometrically sampled until they approach the background level of the image.
An FFT is taken for the flux values around each isophote and the phase of the second coefficient is used to determine a direction.
The average direction for the outer isophotes is taken as the position angle of the galaxy.
Second, with fixed position angle the ellipticity is optimized to minimize the amplitude of the second FFT coefficient relative to the median flux in an isophote.

To compute the error on position angle we use the standard deviation of the outer values from step one.
For ellipticity the error is computed by optimizing the ellipticity for multiple isophotes within 1 PSF length of each other.

Output format:
```python
{'init ellip': , # Ellipticity of the global fit (float)
 'init pa': # Position angle of the global fit (float)
}
```

### Isophotal Fitting

**pipeline label: isophotefit**

A series of isophotes are constructed which grow geometrically until they begin to reach the background level.
Then the algorithm iteratively updates the position angle and ellipticity of each isophote individually for many rounds.
Each round updates every isophote in a random order.
Each round cycles between three options: optimizing position angle, ellipticity, or both.
To optimize the parameters, 5 values (pa, ellip, or both) are randomly sampled and the "loss" is computed.
The loss is a combination of the relative amplitude of the second FFT coefficient (compared to the median flux), and a regularization term.
The regularization term penalizes adjacent isophotes for having different position angle or ellipticity (using the l1 norm).
Thus, all the isophotes are coupled and tend to fit smoothly varying isophotes.
When the optimization has completed three rounds without any isophotes updating, the profile is assumed to have converged.

An uncertainty for each ellipticity and position angle value is determined by taking the RMS between the fitted values and a smoothed polynomial fit values for 4 points.
This is a very rough estimate of the uncertainty, but works sufficiently well in the outskirts.

Output format:
```python
{'fit R': , # Semi-major axis for ellip and pa profile (list)
'fit ellip': , # Ellipticity values at each corresponding R value (list)
'fit ellip_err': , # Optional, uncertainty on ellipticity values (list)
'fit pa': , # Position angle values at each corresponding R value (list)
'fit pa_err': , # Optional, uncertainty on position angle values (list)
}
```

### Star Masking

**pipeline label: starmask**

A cutout of the full image is identified which encloses the full isophotal solution.
The IRAF star finder wrapper from photutils is used to identify stars in the cutout.
The stars are then masked with a variable size circle based on the brightness of the stars.
This routine also identifies saturated pixels if the user provides an *overflowval* as an argument.

Output format:
```python
{'mask': , # 2D array with same dimensions as the image indicating which pixels should be masked (ndarray)
'overflow mask': # 2D array with same dimensions as the image indicating which pixels were saturated (ndarray)
}
```

### Isophotal Profile Extraction

**pipeline label: isophoteextract**

The user may specify a variety of sampling arguments for the photometry extraction.
For example, a start or end radius in pixels, or whether to sample geometrically or linearly in radius.
Geometric sampling is the default as it is faster.
Once the sampling profile of semi-major axis values has been chosen, the function interpolates (spline) the position angle and ellipticity profiles at the requested values.
For any sampling beyond the outer radius from the *Isophotal Fitting* step, a constant value is used.
Within 1 PSF, a circular isophote is used.

Output format:
```python
{'prof header': , # List of strings indicating the order to write the .prof file data (list)
'prof units': , # Dictionary with keys from header, values are strings that give the units for each variable (dict)
'prof data': , # Dictionary with keys from header, values are lists with the data (dict)
'prof format': # Dictionary with keys from header, values are format strings for precision of writing the data (dict)
}
```

### Checking Isophotal Solution

**pipeline label: checkfit**

A variety of checks are applied to ensure that the fit has converged to a reasonable solution.
If a fit passes all of these checks then it is typically an acceptable fit.
However if it fails one or more of the checks then the fit likely either failed or the galaxy has strong non-axisymmetric features (and the fit itself may be acceptable).

One check samples the fitted isophotes and looks for cases with high variability of flux values along the isophote.
This is done by comparing the interquartile range to the median flux, if the interquartile range is larger then that isophote is flagged.
If enough isophotes are flagged then the fit may have failed.

A second check operates similarly, checking the second and fourth FFT coefficient amplitudes relative to the median flux.
If many of the isophotes have large FFT coefficients, or if a few of the isophotes have very large FFT coefficients then the fit is flagged as potentially failed.

A third check is similar to the first, except that it compares the interquartile range from the fitted isophotes to those using just the global position angle and ellipticity values.

Finally, the fourth check compares the total magnitude of the galaxy based on integrating the surface brightness profile against a simple sum of the flux within the isophotes (with a star mask applied).

Output format:
```python
{'checkfit': {'anything': , # True if the test was passed, False if the test failed (bool)
	      'you': , # True if the test was passed, False if the test failed (bool)
	      'want': , # True if the test was passed, False if the test failed (bool)
	      'to': , # True if the test was passed, False if the test failed (bool)
	      'put': } # True if the test was passed, False if the test failed (bool)
}
```

# Advanced Usage

### Modifying Pipeline Functions

This is done with the *new_pipeline_functions* argument, which is formatted as a dictionary with string keys and functions as values.
In this way you can alter the functions used by AutoProf in it's pipeline.

**This is hard to do right**

Each of the functions in *How Does AutoProf Work?* has a pipeline label, this is how the code identifies the functions and their outputs.
Thus, one can create their own version of any function and modify the pipeline by assigning the function to that label.
For example, if you wrote a new center finding function, you could update the pipeline by including:
```python
new_pipeline_functions = {'center': My_Center_Finding_Function}
```
in your config file.
You can also make up any other functions and add them to the pipeline functions list, assigning whatever key you like.
However, AutoProf will only look for functions that are in the pipeline steps object, so see *Modifying Pipeline Steps* for how to add/remove/reorder steps in the pipeline.

Every function in the pipeline has the same template.
To add a new function, or replace an existing one, you must format it as:
```python
def My_New_Function(IMG, pixscale, name, results, **kwargs):
    # Code here
    return {'results': of, 'the': calculations}
```
where *IMG* is the unaltered input image, *pixscale* is the pixel scale in arcsec/pix, *name* is a string that identifies the galaxy, *results* is a dictionary containing the output of all previous pipeline steps, and *kwargs* is a dictionary with all user specified arguments from *List Of AutoProf Arguments* if they have non-default values.
The output of every function in the pipeline is a dictionary with strings for keys.
If you wish to replace a function, make sure to have the output follow the same format.
So long as your output dictionary has the same keys/value format, it should be able to seamlessly replace that step in the pipeline.
If you wish to include more information, you can include as many other entries in the dictionary as you like, the default pipeline functions will ignore them.
See *How Does AutoProf Work?* for the expected outputs from each function.

### Modifying Pipeline Steps

This is done with the *new_pipeline_steps* argument, which is formatted as a list of strings which tells AutoProf what order to run it's pipeline functions.
In this way you can alter the order of operations used by AutoProf in it's pipeline.

**This is hard to do right**

Each function must be run in a specific order as they often rely on the output from another step.
The basic pipeline step order is:
```python
['background', 'psf', 'center', 'isophoteinit', 'isophotefit', 'starmask', 'isophoteextract', 'checkfit']
```
For forced photomettry the pipeline step order is:
```python
['background', 'psf', 'center forced', 'isophoteinit', 'isophotefit forced', 'starmask forced', 'isophoteextract forced']
```
So the background, psf, and global ellip/pa are always fit directly to the image, but for forced photometry the center, isophote parameter, and star mask are fixed.
If you would like to change this behaviour, just provide a new pipeline steps list.
For example if you wished to re-fit the center for an image you can change *center forced* back to *center*.

You can create your own order, or add in new functions by supplying a new list.
For example, if you had your own function to run after the centering function you could do so by including:
```python
new_pipeline_functions = {'myfunction': My_New_Function}
new_pipeline_steps = ['background', 'psf', 'center', 'myfunction', 'isophoteinit', 'isophotefit', 'starmask', 'isophoteextract', 'checkfit']
```
in your config file.
Note that for *new_pipeline_functions* you need only include the new function, while for *new_pipeline_steps* you must write out the full pipeline steps.
If you wish to skip a step, it is sometimes better to write your own "null" version of the function (and change *new_pipeline_functions*) that just returns do-nothing values for it's dictionary as the other functions may still look for the output and could crash. 
