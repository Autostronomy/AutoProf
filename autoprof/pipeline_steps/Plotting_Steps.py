import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from ..autoprofutils.SharedFunctions import (
    _iso_extract,
    _x_to_eps,
    _x_to_pa,
    _inv_x_to_pa,
    _inv_x_to_eps,
    LSBImage,
    Angle_Average,
    Angle_Median,
    AddLogo,
    PA_shift_convention,
    Sigma_Clip_Upper,
)
import logging

__all__ = ("Plot_Galaxy_Image", )

def Plot_Galaxy_Image(IMG, results, options):
    """Generate a plain image of the galaxy

    Plots an LSB image of the object without anything else drawn above
    it.  Useful for inspecting images for spurious features. This step
    can be run at any point in the pipeline. It will take advantage of
    whatever information has been determined so far. So if it is the
    first pipeline step, it has little to work from and will simply
    plot the whole image, if it is run after the isophote
    initialization step then the plotted image will be cropped to
    focus on the galaxy.

    Parameters
    -----------------
    ap_guess_center : dict, default None
      user provided starting point for center fitting. Center should
      be formatted as:

      .. code-block:: python

        {'x':float, 'y': float}

      , where the floats are the center coordinates in pixels. If not
      given, Autoprof will default to a guess of the image center.

    ap_set_center : dict, default None
      user provided fixed center for rest of calculations. Center
      should be formatted as:

      .. code-block:: python

        {'x':float, 'y': float}

      , where the floats are the center coordinates in pixels. If not
      given, Autoprof will default to a guess of the image center.

    Notes
    --------------
    :References:
    - 'background'
    - 'background noise'
    - 'center' (optional)
    - 'init R' (optional)

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {}
    """

    if "center" in results:
        center = results["center"]
    elif "ap_set_center" in options:
        center = options["ap_set_center"]
    elif "ap_guess_center" in options:
        center = options["ap_guess_center"]
    else:
        center = {"x": IMG.shape[1] / 2, "y": IMG.shape[0] / 2}

    if "prof data" in results:
        edge = 1.2 * results["prof data"]["R"][-1] / options["ap_pixscale"]
    elif "init R" in results:
        edge = 3 * results["init R"]
    elif "fit R" in results:
        edge = 2 * results["fit R"]
    else:
        edge = max(IMG.shape) / 2
    edge = min(
        [
            edge,
            abs(center["x"] - IMG.shape[1]),
            center["x"],
            abs(center["y"] - IMG.shape[0]),
            center["y"],
        ]
    )

    ranges = [
        [max(0, int(center["x"] - edge)), min(IMG.shape[1], int(center["x"] + edge))],
        [max(0, int(center["y"] - edge)), min(IMG.shape[0], int(center["y"] + edge))],
    ]

    LSBImage(
        IMG[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]]
        - results["background"],
        results["background noise"],
    )
    if not ("ap_nologo" in options and options["ap_nologo"]):
        AddLogo(plt.gcf())
    plt.savefig(
        f"{options.get('ap_plotpath','')}clean_image_{options['ap_name']}.{options.get('ap_plot_extension', 'jpg')}",
        dpi=options["ap_plotdpi"] if "ap_plotdpi" in options else 300,
    )
    plt.close()

    return IMG, {}
