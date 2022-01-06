========================
Troubleshooting AutoProf
========================

Here we list some common troubles that users can run into when getting started or when beginning to access some of AutoProf's more advanced features. By it's nature this is an evolving page, so if you run into something that you think should be addressed on this page please reach out to `Connor Stone <https://connorjstone.com/>`_! 

The error bars on the SB profile diagnostic plot are too large
--------------------------------------------------------------

The diagnostic SB plot is intended for evaluating the quality of a fit. With many SB profiles, the statistical errors are too small to see in a plot and so they are scaled up arbitrarily to make them visible. The errorbar scaling factor can be seen in the legend as *(err.xx)* where xx is the scaling factor. When scaling is applied it is done such that the largest error bar on the plot is now 1 mag arcsec^-2.

I want to hold the PA and ellipticity fixed at a global value instead of fitting each isophote
----------------------------------------------------------------------------------------------

This can be done with the "isophotefit fixed" pipeline step replacing the standard "isophotefit" step. For information on adjusting the AutoProf pipeline see :doc:`pipelinemanipulation`.

The fitted isophotes are too smooth, they are missing features
--------------------------------------------------------------

AutoProf is intended for use on large surveys with thousands to millions of objects.
By default it applies a certain amount of profile smoothing to achieve relaiable automated results.
In some cases this smoothing can overdo it and will cause AutoProf to skip over a real feature (it will still fit the global structure though).
This smoothing is controlled by *ap_regularize_scale* which by default is 1.
Decreasing the value will reduce the smoothing strength, at 0 there will be no smoothing applied.
If the isophotes continue to come out too smooth then the culprit is likely the *ap_isofit_robustclip* parameter, which clips high flux values while fitting isophotes.
It defaults to 0.15 meaning that the 15% highest flux values are clipped down to lower values.
This parameter can also be set to 0 to turn off the effect.
It is not advised to turn off these features when batch processing large numbers of galaxies since it will make the fit susceptible to spurious image features. 

