
import unittest
import numpy as np

from detection_engine.two_dimensional_cfar import ca_cfar_2d


class TestTwoDimensionalCFAR(unittest.TestCase):

    def test_detects_strong_target(self):
        """A strong target should exceed the local CFAR threshold."""

        power_map = np.ones((64, 64))

        # Insert a strong target at the centre.
        power_map[32, 32] = 1000

        detections, threshold, noise = ca_cfar_2d(power_map)

        self.assertTrue(detections[32, 32])

    def test_does_not_detect_uniform_noise(self):
        """Uniform power should not produce false detections."""

        power_map = np.ones((64, 64))

        detections, threshold, noise = ca_cfar_2d(power_map)

        self.assertFalse(np.any(detections))

    def test_boundary_cells_are_not_processed(self):
        """Cells without a complete CFAR window remain unprocessed."""

        power_map = np.ones((64, 64))

        detections, threshold, noise = ca_cfar_2d(
            power_map,
            guard_cells=(2, 2),
            training_cells=(4, 4)
        )

        # Outer half-window = 2 + 4 = 6 cells.
        self.assertTrue(np.all(np.isinf(threshold[:6, :])))
        self.assertTrue(np.all(np.isinf(threshold[-6:, :])))

        self.assertTrue(np.all(np.isnan(noise[:6, :])))

    def test_training_cell_count_and_threshold(self):
        """Verify that the default window uses exactly 144 training cells."""

        power_map = np.ones((64, 64))

        detections, threshold, noise = ca_cfar_2d(
            power_map,
            guard_cells=(2, 2),
            training_cells=(4, 4),
            pfa=0.01
        )

        # Outer window = 13 x 13 = 169 cells.
        # Guard region + CUT = 5 x 5 = 25 cells.
        # Training cells = 169 - 25 = 144.
        expected_n = 144

        expected_alpha = expected_n * (
            0.01 ** (-1 / expected_n) - 1
        )

        # Every training cell has power 1.
        # Therefore, estimated noise must be 1.
        self.assertAlmostEqual(noise[32, 32], 1.0)

        # Threshold must equal alpha * noise estimate.
        self.assertAlmostEqual(
            threshold[32, 32],
            expected_alpha
        )

    def test_rejects_invalid_pfa(self):
        """Pfa must be strictly between zero and one."""

        power_map = np.ones((64, 64))

        with self.assertRaises(ValueError):
            ca_cfar_2d(power_map, pfa=1.5)


if __name__ == "__main__":
    unittest.main()
