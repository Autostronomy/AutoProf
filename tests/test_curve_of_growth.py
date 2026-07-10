import unittest
from unittest.mock import patch

import numpy as np

from autoprof.pipeline_steps.Isophote_Extract import _Generate_Profile


class TestMaskedProfileCurveOfGrowth(unittest.TestCase):
    def setUp(self):
        self.image = np.ones((20, 20), dtype=float)
        self.results = {
            "background": 0.0,
            "background noise": 1.0,
            "center": {"x": 10.0, "y": 10.0},
            "psf fwhm": 2.0,
        }
        self.options = {
            "ap_name": "masked-profile-test",
            "ap_pixscale": 1.0,
            "ap_doplot": False,
        }

    @staticmethod
    def parameters(count):
        return [{"ellip": 0.0, "pa": 0.0} for _ in range(count)]

    def test_fully_masked_profiles_have_no_curve_of_growth(self):
        self.results["mask"] = np.ones_like(self.image, dtype=bool)
        for count in (3, 6):
            with self.subTest(count=count):
                radii = np.arange(1, count + 1, dtype=float)
                profile = _Generate_Profile(
                    self.image,
                    self.results,
                    radii,
                    self.parameters(count),
                    self.options,
                )["prof data"]

                for column in ("SB", "SB_e", "totmag", "totmag_e", "totmag_direct"):
                    np.testing.assert_array_equal(profile[column], np.full(count, 99.999))
                np.testing.assert_array_equal(profile["pixels"], np.zeros(count))

    def test_fully_masked_intensity_profile_keeps_existing_sentinels(self):
        self.results["mask"] = np.ones_like(self.image, dtype=bool)
        radii = np.arange(1, 4, dtype=float)
        options = dict(self.options, ap_fluxunits="intensity")
        profile = _Generate_Profile(
            self.image,
            self.results,
            radii,
            self.parameters(len(radii)),
            options,
        )["prof data"]

        for column in ("I", "I_e", "totflux_direct"):
            self.assertTrue(np.all(np.isnan(profile[column])))
        for column in ("totflux", "totflux_e"):
            np.testing.assert_array_equal(profile[column], np.full(len(radii), -99.999))

    @patch("autoprof.pipeline_steps.Isophote_Extract._iso_between")
    @patch("autoprof.pipeline_steps.Isophote_Extract._iso_extract")
    def test_masked_row_is_excluded_from_curve_of_growth(self, mock_extract, mock_between):
        def extract(_, radius, *__args, **__kwargs):
            if radius == 3:
                return np.array([]), np.array([]), 15
            return np.linspace(0.9, 1.1, 15), np.linspace(0, 2 * np.pi, 15), 0

        mock_extract.side_effect = extract
        mock_between.return_value = np.ones(20)
        radii = np.arange(1, 7, dtype=float)
        profile = _Generate_Profile(
            self.image,
            self.results,
            radii,
            self.parameters(len(radii)),
            self.options,
        )["prof data"]

        for column in ("SB", "SB_e", "totmag", "totmag_e", "totmag_direct"):
            self.assertEqual(profile[column][2], 99.999)
        self.assertTrue(np.all(np.asarray(profile["totmag"])[[0, 1, 3, 4, 5]] < 99))

    @patch("autoprof.pipeline_steps.Isophote_Extract._iso_between")
    @patch("autoprof.pipeline_steps.Isophote_Extract._iso_extract")
    def test_truncated_profile_columns_keep_matching_lengths(self, mock_extract, mock_between):
        def extract(_, radius, *__args, **__kwargs):
            flux = 1.0 if radius == 1 else -1.0
            return np.full(15, flux), np.linspace(0, 2 * np.pi, 15), 0

        mock_extract.side_effect = extract
        mock_between.return_value = np.ones(20)
        radii = np.arange(1, 7, dtype=float)
        options = dict(self.options, ap_truncate_evaluation=True)
        result = _Generate_Profile(
            self.image,
            self.results,
            radii,
            self.parameters(len(radii)),
            options,
        )

        self.assertEqual(len(result["prof data"]["R"]), 3)
        for column in result["prof header"]:
            self.assertEqual(len(result["prof data"][column]), 3)


if __name__ == "__main__":
    unittest.main()
