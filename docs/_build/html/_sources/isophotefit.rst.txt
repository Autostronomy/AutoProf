============================
Fitting Elliptical Isophotes
============================


Description
-----------

**pipeline label: isophotefit**

The :func:`defualt isophotal fitting
<~autoprofutils.Isophote_Fit.Isophote_Fit_FFT_Robust>` routine
simultaneously optimizes a collection of elliptical isophotes by
minimizing the 2nd FFT coefficient power, regularized for
robustness. A series of isophotes are constructed which grow
geometrically until they begin to reach the background level.  Then
the algorithm iteratively updates the position angle and ellipticity
of each isophote individually for many rounds.  Each round updates
every isophote in a random order.  Each round cycles between three
options: optimizing position angle, ellipticity, or both.  To optimize
the parameters, 5 values (pa, ellip, or both) are randomly sampled and
the "loss" is computed.  The loss is a combination of the relative
amplitude of the second FFT coefficient (compared to the median flux),
and a regularization term.  The regularization term penalizes adjacent
isophotes for having different position angle or ellipticity (using
the l1 norm).  Thus, all the isophotes are coupled and tend to fit
smoothly varying isophotes.  When the optimization has completed three
rounds without any isophotes updating, the profile is assumed to have
converged.

An uncertainty for each ellipticity and position angle value is
determined by taking the RMS between the fitted values and a smoothed
polynomial fit values for 4 points.  This is a very rough estimate of
the uncertainty, but works sufficiently well in the outskirts.

Output format:

.. code-block:: python
   
  {'fit R': , # Semi-major axis for ellip and pa profile (list)
   'fit ellip': , # Ellipticity values at each corresponding R value (list)
   'fit ellip_err': , # Optional, uncertainty on ellipticity values (list)
   'fit pa': , # Position angle values at each corresponding R value (list)
   'fit pa_err': , # Optional, uncertainty on position angle values (list)
   'auxfile fitlimit': # optional, message ofr aux file to record fitting limit semi-major axis (string)
  
  }


Config Parameters
-----------------

ap_scale
  growth scale when fitting isophotes, not the same as *ap_sample---scale*. Default is 0.2. (float)

ap_fit_limit
  noise level out to which to extend the fit in units of pixel background noise level. Default is 2, smaller values will end fitting further out in the galaxy image. (float)

ap_regularize_scale
  scale factor to apply to regularization coupling factor between isophotes.
  Default of 1, larger values make smoother fits, smaller values give more chaotic fits. (float)
