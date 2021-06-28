.. the "raw" directive below is used to hide the title in favor of
   just the logo being visible
.. raw:: html

    <style media="screen" type="text/css">
      h1 {display:none;}
    </style>

.. |br| raw:: html

    <div style="min-height:0.1em;"></div>

*********
AutoProf
*********

.. raw:: html

   <img src="_static/AP_logo.png";" width="495"/>

.. only:: latex

    .. image:: _static/AP_logo.png

|br|

.. Important::
    If you use AutoProf for a project that leads to a publication,
    whether directly or as a dependency of another package, please
    include an :doc:`acknowledgment and/or citation <citation>`.

|br|

Getting Started
===============

.. toctree::
    :maxdepth: 1

    install.rst
    ..
       getting_started.rst
       contributing.rst
       citation.rst
       license.rst

..
   User Documentation
   ==================

   .. toctree::
       :maxdepth: 1

       background.rst
       detection.rst
       grouping.rst
       aperture.rst
       psf.rst
       epsf.rst
       psf_matching.rst
       segmentation.rst
       centroids.rst
       morphology.rst
       isophote.rst
       geometry.rst
       datasets.rst
       utils.rst

|br|

.. note::

    Like much astronomy software, AutoProf is an evolving package.
    I try to keep the API stable and consistent, however I will make
    changes to the interface if it considerably improves things
    going forward. Please contact connor.stone@queensu.ca if you experience
    issues. If you would like to be notified of major changes send an email
    with the subject line "AUTOPROF MAILING LIST".
