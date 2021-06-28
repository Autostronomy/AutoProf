==================
Fit Success Checks
==================

Description
-----------

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
.. code-block:: python
   
  {'checkfit': {'anything': , # True if the test was passed, False if the test failed (bool)
  	        'you': , # True if the test was passed, False if the test failed (bool)
  	        'want': , # True if the test was passed, False if the test failed (bool)
	        'to': , # True if the test was passed, False if the test failed (bool)
	        'put': }, # True if the test was passed, False if the test failed (bool)
  
   'auxfile checkfit anything': ,# optional aux file message for pass/fail of test (string) 
   'auxfile checkfit you': ,# optional aux file message for pass/fail of test (string) 
   'auxfile checkfit want': ,# optional aux file message for pass/fail of test (string) 
   'auxfile checkfit to': ,# optional aux file message for pass/fail of test (string) 
   'auxfile checkfit put': # optional aux file message for pass/fail of test (string)
  
  }

Config Parameters
-----------------
