import numpy as np
import sys
import os

from .SharedFunctions import interpolate_Lanczos

__all__ = ("Crop", "Resolution")

def Crop(IMG, results, options):
    """
    Crop the edges of an image about a center point.

    If a 'Center' method has been applied before in the pipeline we use that to
    define the galaxy center, otherwise we define the center as half the image.

    ap_cropto states the new size of the image after cropping.
    We default to 512*512.
    """

    if "center" in results:
        x = results["center"]["x"]
        y = results["center"]["y"]
        center = np.array((x, y)).astype("int")
    else:
        center = np.array(IMG.shape) // 2

    cropto = options["ap_cropto"] if "ap_cropto" in options else (512, 512)

    IMG = IMG[
        center[0] - cropto[0] // 2 : center[0] + cropto[0] // 2,
        center[1] - cropto[1] // 2 : center[1] + cropto[1] // 2,
    ]

    return IMG, {}


def Resolution(IMG, results, options):
    """Change the resolution of an image

    Can be used to resample an image at a new pixel scale. The new
    image can have finer sampling (smaller pixels) or coarser sampling
    (larger pixels) and this can be accomplished with a variety of
    methods. First, there are pooling methods which can only reduce
    the resolution, these methods take a certain block as defined by
    *ap_resolution_shape* and perform a mean, median, or max operation
    on those pixels. That value then becomes the pixel value for a
    downsampled image. Alternatively, one can use either bicubic or
    Lanczos interpolation, which can upsample or downsample the
    image. The parameter *ap_resolution_shape* is then the desired new
    shape of the image, which will be fit to represent the same area
    on the sky. To match the same area on the sky, each pixel is
    assumed to have a width of 1, thus representing a 1x1 square on
    the sky.

    Parameters
    -----------------
    ap_resolution_method : string, default 'lanczos'
      Method which is used to perform resolution resampling. Options
      are: 'max pool', 'mean pool', 'median pool', 'bicubic', and
      'lanczos'. Lanczos is very slow, but represents the best
      interpolation method according to information theory.

    ap_resolution_shape : tuple, default None
      Shape used for resampling. For pooling methods, this represents
      the size of the pool/box in which a calcualtion is done. For
      other methods this is the size of the final desired image. This
      parameter is required.

    ap_resolution_dtype : object, default None
      Optional parameter to set a new dtype for the image after
      resampling. This can be used to reduce precision if it is
      unnecessary, limiting the size of the final image if it is very
      finely resampled.

    ap_iso_interpolate_window : int, default 5
      Only used by Lanczos interpolation, this will set the area of
      pixels used to calculate the Lanczos interpolated values. Larger
      values are more accurate, but require quadratically more
      computation time.

    Note
    ----
    The inputted pixel scale will be incorrect after this operation if
    it is set for the original image.

    Returns
    -------
    IMG : ndarray
      Resampled galaxy image according to user specified sampling
      method.

    results : dict
      .. code-block:: python

        {}

    """

    if (
        "ap_resolution_method" in options
        and options["ap_resolution_method"] == "max pool"
    ):
        M, N = IMG.shape
        K, L = options["ap_resolution_shape"]
        MK = M // K
        NL = N // L
        newIMG = mat[: MK * K, : NL * L].reshape(MK, K, NL, L).max(axis=(1, 3))
    if (
        "ap_resolution_method" in options
        and options["ap_resolution_method"] == "mean pool"
    ):
        M, N = IMG.shape
        K, L = options["ap_resolution_shape"]
        MK = M // K
        NL = N // L
        newIMG = mat[: MK * K, : NL * L].reshape(MK, K, NL, L).mean(axis=(1, 3))
    if (
        "ap_resolution_method" in options
        and options["ap_resolution_method"] == "median pool"
    ):
        M, N = IMG.shape
        K, L = options["ap_resolution_shape"]
        MK = M // K
        NL = N // L
        newIMG = mat[: MK * K, : NL * L].reshape(MK, K, NL, L).median(axis=(1, 3))
    if (
        "ap_resolution_method" in options
        and options["ap_resolution_method"] == "bicubic"
    ):
        newIMG = np.zeros(
            options["ap_resolution_shape"],
            dtype=options["ap_resolution_dtype"]
            if "ap_resolution_dtype" in options
            else IMG.dtype,
        )
        XX, YY = np.meshgrid(
            np.linspace(-0.5, IMG.shape[0] - 0.5, options["ap_resolution_shape"][0]),
            np.linspace(-0.5, IMG.shape[1] - 0.5, options["ap_resolution_shape"][1]),
        )
        newIMG = interpolate_bicubic(IMG, XX.ravel(), YY.ravel()).reshape(newIMG.shape)
    else:
        newIMG = np.zeros(
            options["ap_resolution_shape"],
            dtype=options["ap_resolution_dtype"]
            if "ap_resolution_dtype" in options
            else IMG.dtype,
        )
        XX, YY = np.meshgrid(
            np.linspace(-0.5, IMG.shape[0] - 0.5, options["ap_resolution_shape"][0]),
            np.linspace(-0.5, IMG.shape[1] - 0.5, options["ap_resolution_shape"][1]),
        )
        for i in range(len(newIMG.shape[0])):
            newIMG[i] = interpolate_Lanczos(
                IMG,
                XX[i],
                YY[i],
                scale=int(options["ap_iso_interpolate_window"])
                if "ap_iso_interpolate_window" in options
                else 3,
            )

    return newIMG, {}
