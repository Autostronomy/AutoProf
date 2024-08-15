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
