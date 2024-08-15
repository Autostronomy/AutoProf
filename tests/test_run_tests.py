import sys
from os.path import isfile
from os import remove

from autoprof import run_from_terminal


def test_basic_config():
    sys.argv = ["", "basic_config.py"]
    run_from_terminal()
    for checkfile in [
        "Background_hist_testimage.jpg",
        "fit_ellipse_testimage.jpg",
        "initialize_ellipse_optimize_testimage.jpg",
        "initialize_ellipse_testimage.jpg",
        "phase_profile_testimage.jpg",
        "photometry_ellipse_testimage.jpg",
        "photometry_testimage.jpg",
        "testimage.aux",
        "testimage.prof",
    ]:
        assert isfile(checkfile)
        remove(checkfile)


def test_batch_config():
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    sys.argv = ["", "batch_config.py"]
    run_from_terminal()
    for imgname in ["testbatchimage1", "testbatchimage2", "testbatchimage3", "testbatchimage4"]:
        for checkfile in [
            f"Background_hist_{imgname}.jpg",
            f"fit_ellipse_{imgname}.jpg",
            f"initialize_ellipse_optimize_{imgname}.jpg",
            f"initialize_ellipse_{imgname}.jpg",
            f"phase_profile_{imgname}.jpg",
            f"photometry_ellipse_{imgname}.jpg",
            f"photometry_{imgname}.jpg",
            f"{imgname}.aux",
            f"{imgname}.prof",
        ]:
            assert isfile(checkfile)
            remove(checkfile)


def test_custom_config():
    sys.argv = ["", "custom_config.py"]
    run_from_terminal()
    for checkfile in [
        "Background_hist_testcustomprocessing.jpg",
        "testcustomprocessing_mask.fits.gz",
        "testcustomprocessing.aux",
    ]:
        print(checkfile)
        assert isfile(checkfile)
        remove(checkfile)


def test_forced_config():
    sys.argv = ["", "basic_config.py"]
    run_from_terminal()
    sys.argv = ["", "forced_config.py"]
    run_from_terminal()
    for checkfile in [
        "Background_hist_testforcedimage.jpg",
        "phase_profile_testforcedimage.jpg",
        "photometry_ellipse_testforcedimage.jpg",
        "photometry_testforcedimage.jpg",
        "testforcedimage.aux",
        "testforcedimage.prof",
    ]:
        print(checkfile)
        assert isfile(checkfile)
        remove(checkfile)

    for checkfile in [
        "Background_hist_testimage.jpg",
        "fit_ellipse_testimage.jpg",
        "initialize_ellipse_optimize_testimage.jpg",
        "initialize_ellipse_testimage.jpg",
        "phase_profile_testimage.jpg",
        "photometry_ellipse_testimage.jpg",
        "photometry_testimage.jpg",
        "testimage.aux",
        "testimage.prof",
    ]:
        remove(checkfile)


def test_tree_config():
    sys.argv = ["", "tree_config.py"]
    run_from_terminal()
    for checkfile in [
        "axial_profile_0_testtreeimage.jpg",
        "axial_profile_1_testtreeimage.jpg",
        "axial_profile_2_testtreeimage.jpg",
        "axial_profile_3_testtreeimage.jpg",
        "axial_profile_lines_testtreeimage.jpg",
        "Background_hist_testtreeimage.jpg",
        "clean_image_testtreeimage.jpg",
        "ellipsemodel_gen_testtreeimage.jpg",
        "ellipseresidual_gen_testtreeimage.jpg",
        "fit_ellipse_testtreeimage.jpg",
        "initialize_ellipse_optimize_testtreeimage.jpg",
        "initialize_ellipse_testtreeimage.jpg",
        "mask_testtreeimage.jpg",
        "phase_profile_testtreeimage.jpg",
        "photometry_ellipse_testtreeimage.jpg",
        "photometry_testtreeimage.jpg",
        "radial_profiles_testtreeimage.jpg",
        "radial_profiles_wedges_testtreeimage.jpg",
        "slice_profile_testtreeimage.jpg",
        "slice_profile_window_testtreeimage.jpg",
        "testtreeimage.aux",
        "testtreeimage.prof",
        "testtreeimage_axial_profile.prof",
        "testtreeimage_genmodel.fits",
        "testtreeimage_slice_profile.prof",
    ]:
        print(checkfile)
        assert isfile(checkfile)
        remove(checkfile)
