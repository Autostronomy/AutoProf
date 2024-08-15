=========================
AutoProf Pipeline Control
=========================

Modifying Pipeline Methods
--------------------------

This is done with the *ap_new_pipeline_methods* argument, which is formatted as a dictionary with string keys and functions as values.
In this way you can add to or alter the methods used by AutoProf in it's pipeline.

Each of the methods in :doc:`defaultpipeline` has a pipeline label, this is how the code identifies the functions and their outputs.
Thus, one can create their own version of any method and modify the pipeline by assigning the function to that label.
For example, if you wrote a new center finding method, you could update the pipeline by including:

.. code-block:: python
   
  ap_new_pipeline_methods = {'center': My_Center_Finding_Method}
		
in your config file.
You can also make up any other methods and add them to the pipeline functions list, assigning whatever key you like.
However, AutoProf will only look for methods that are in the *pipeline_steps* object, so see `Modifying Pipeline Steps`_ for how to add/remove/reorder steps in the pipeline.

Pipeline Method Template
------------------------

Every function in the pipeline has the same template.
To add a new function, or replace an existing one, you must format it as:

.. code-block:: python
   
  def My_New_Function(IMG, results, options):
      # Code here
      return IMG, {'results': of, 'the': calculations}

where *IMG* is the input image, *results* is a dictionary containing the output of all previous pipeline steps, and *options* is a dictionary with all user specified arguments (any variable in the config file that starts with *ap\_*) if they have non-default values (None).
The output of every method in the pipeline is an image and a dictionary with strings for keys.
The output image is assigned to replace the input image, so if you wish to alter the input image you can do so in a way that all future steps will see.
The dictionary output is used to update the *results* dictionary that is passed to all future methods, you can therefore add new elements to the dictionary or replace older ones. 
If you wish to replace a method, make sure to have the output follow this format.
So long as your output dictionary has the same keys/value format, it should be able to seamlessly replace a step in the pipeline.
If you wish to include more information, you can include as many other entries in the dictionary as you like, the default methods functions will ignore them.
See the corresponding documentation for the expected outputs from each function.

Modifying Pipeline Steps
------------------------

This is done with the *ap_new_pipeline_steps* argument, which is formatted as a list of strings which tells AutoProf what order to run it's pipeline methods.
In this way you can alter the order of operations used by AutoProf in it's pipeline.

Each function must be run in a specific order as they often rely on the output from another step.
The basic pipeline step order is:

.. code-block:: python
   
  ['background', 'psf', 'center', 'isophoteinit', 'isophotefit', 'isophoteextract', 'checkfit', 'writeprof']

For forced photometry the default pipeline step order is:

.. code-block:: python
   
  ['background', 'psf', 'center forced', 'isophoteinit', 'isophotefit forced', 'isophoteextract forced', 'writeprof']

If you would like to change this behaviour, just provide a *ap_new_pipeline_steps* list.
For example if you wished to use forced photometry but you want to re-fit the center you can change :func:`~autoprof.pipeline_steps.Center.Center_Forced` back to :func:`~autoprof.pipeline_steps.Center.Center_HillClimb` with:

.. code-block:: python
   
  ap_new_pipeline_steps = ['background', 'psf', 'center', 'isophoteinit', 'isophotefit forced', 'isophoteextract forced', 'writeprof']

in your config file.

You can create your own order, or add in new functions by supplying a new list.
For example, if you had your own method to run after the centering function you could do so by including:

.. code-block:: python
   
  ap_new_pipeline_methods = {'mymethod': My_New_Method}
  ap_new_pipeline_steps = ['background', 'psf', 'center', 'mymethod', 'isophoteinit', 'isophotefit', 'isophoteextract', 'checkfit', 'writeprof']

in your config file.
Note that for *ap_new_pipeline_methods* you need only include the new function, while for *ap_new_pipeline_steps* you must write out the full pipeline steps.
If you wish to skip a step, it is sometimes better to write your own "null" version of the function (and change *ap_new_pipeline_methods*) that just returns do-nothing values for it's dictionary as the other functions may still look for the output and could crash.

Example Custom Pipeline
-----------------------

This config file demonstrates the flexability of AutoProf pipelines to perform
custom tasks. In this case a mask is produced which removes any pixel with a
negative flux value, then computes the background level of the image. This
background is written to a custom aux file and the pixel mask is saved. No
isophote fitting is performed, only measurement of background level and a count
of pixels within a flux range.

.. code-block:: python
   
  import os
  from datetime import datetime
  from astropy.io import fits
  from time import sleep
  import logging
  import numpy as np

  ap_process_mode = "image"
  ap_doplot = True
  ap_image_file = "ESO479-G1_r.fits"
  ap_name = "testcustomprocessing"
  ap_pixscale = 0.262
  ap_zeropoint = 22.5
  ap_badpixel_low = 0


  def mywriteoutput(IMG, results, options):
      saveto = options["ap_saveto"] if "ap_saveto" in options else "./"
      with open(os.path.join(saveto, options["ap_name"] + ".aux"), "w") as f:
          # write profile info
          f.write("written on: %s\n" % str(datetime.now()))
          f.write("name: %s\n" % str(options["ap_name"]))
          for r in sorted(results.keys()):
              if "auxfile" in r:
                  f.write(results[r] + "\n")
          for k in sorted(options.keys()):
              if k == "ap_name":
                  continue
          f.write("option %s: %s\n" % (k, str(options[k])))
      # Write the mask data, if provided
      if "mask" in results and (not results["mask"] is None):
          header = fits.Header()
          header["IMAGE 1"] = "mask"
          hdul = fits.HDUList(
              [fits.PrimaryHDU(header=header), fits.ImageHDU(results["mask"].astype(int))]
          )
          hdul.writeto(saveto + options["ap_name"] + "_mask.fits", overwrite=True)
          sleep(1)
          # Zip the mask file because it can be large and take a lot of memory, but in principle
          # is very easy to compress
          os.system("gzip -fq " + saveto + options["ap_name"] + "_mask.fits")

      return IMG, {}


  def count_pixel_range(IMG, results, options):

      count = np.sum(
          np.logical_and(IMG > options["ap_mycountrange_low"], IMG < options["ap_mycountrange_high"])
      )

      logging.info("%s: counted %i pixels in custom range" % (options["ap_name"], count))

      return IMG, {
          "auxfile count pixels in range": "In range from %.2f to %.2f there were %i pixels"
          % (options["ap_mycountrange_low"], options["ap_mycountrange_low"], count)
      }


  ap_new_pipeline_steps = {
      "head": [
          "mask badpixels",
          "background",
          "count pixel range",
          "custom writebackground",
      ]
  }
  ap_new_pipeline_methods = {
      "custom writebackground": mywriteoutput,
      "count pixel range": count_pixel_range,
  }

  # note these parameters are not standard for AutoProf, they are only used in the custom function.
  # Users can create any such parameters that they like so long as the variable begins with 'ap_'
  ap_mycountrange_low = 0.2
  ap_mycountrange_high = 0.3

To try out this pipeline, download the `test file <https://github.com/Autostronomy/AutoProf/raw/main/tests/ESO479-G1_r.fits>`_.

Example Tree Pipeline
---------------------

This example shows how to create a tree pipeline, which dynamically changes the
pipeline based on the results of previous steps.

.. code-block:: python

  import numpy as np

  ap_process_mode = "image"

  ap_image_file = "ESO479-G1_r.fits"
  ap_pixscale = 0.262
  ap_name = "testtreeimage"
  ap_doplot = True
  ap_isoband_width = 0.05
  ap_samplegeometricscale = 0.05
  ap_truncate_evaluation = True
  ap_ellipsemodel_resolution = 2.0

  ap_fouriermodes = 4
  ap_slice_anchor = {"x": 1700.0, "y": 1350.0}
  ap_slice_length = 300.0
  ap_isoclip = True


  def My_Edgon_Fit_Method(IMG, results, options):
      N = 100
      return IMG, {
          "fit ellip": np.array([results["init ellip"]] * N),
          "fit pa": np.array([results["init pa"]] * N),
          "fit ellip_err": np.array([0.05] * N),
          "fit pa_err": np.array([5 * np.pi / 180] * N),
          "fit R": np.logspace(0, np.log10(results["init R"] * 2), N),
      }


  def whenrerun(IMG, results, options):
      count_checks = 0
      for k in results["checkfit"].keys():
          if not results["checkfit"][k]:
              count_checks += 1

      if count_checks <= 0:  # if checks all passed, carry on
          return None, {"onloop": options["onloop"] if "onloop" in options else 0}
      elif (
          not "onloop" in options
      ):  # start by simply re-running the analysis to see if AutoProf got stuck
          return "head", {"onloop": 1}
      elif options["onloop"] == 1 and (
          not results["checkfit"]["FFT coefficients"]
          or not results["checkfit"]["isophote variability"]
      ):  # Try smoothing the fit the result was chaotic
          return "head", {"onloop": 2, "ap_regularize_scale": 3, "ap_fit_limit": 5}
      elif (
          options["onloop"] == 1 and not results["checkfit"]["Light symmetry"]
      ):  # Try testing larger area to find center if fit found high asymmetry (possibly stuck on a star)
          return "head", {"onloop": 2, "ap_centeringring": 20}
      else:  # Don't try a third time, just give up
          return None, {"onloop": options["onloop"] if "onloop" in options else 0}


  ap_new_pipeline_methods = {
      "branch edgeon": lambda IMG, results, options: (
          "edgeon" if results["init ellip"] > 0.8 else "standard",
          {},
      ),
      "branch rerun": whenrerun,
      "edgeonfit": My_Edgon_Fit_Method,
  }
  ap_new_pipeline_steps = {
      "head": [
          "background",
          "psf",
          "center",
          "isophoteinit",
          "branch edgeon",
      ],
      "standard": [
          "isophotefit",
          "starmask",
          "isophoteextract",
          "checkfit",
          "branch rerun",
          "writeprof",
          "plot image",
          "ellipsemodel",
          "axialprofiles",
          "radialprofiles",
          "sliceprofile",
      ],
      "edgeon": [
          "edgeonfit",
          "isophoteextract",
          "radsample",
          "axialprofiles",
          "writeprof",
      ],
  }

To try out this pipeline, download the `test file <https://github.com/Autostronomy/AutoProf/raw/main/tests/ESO479-G1_r.fits>`_.