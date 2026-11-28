
import unittest
import numpy as np

from detection_engine.target_extraction import extract_targets


class TestTargetExtraction(unittest.TestCase):

    def test_extracts_single_target(self):

        detections = np.zeros((64, 64), dtype=bool)
        power_map = np.ones((64, 64))

        # Create a connected cluster around one target.
        detections[30:33, 30:33] = True
        power_map[31, 31] = 1000

        targets = extract_targets(
            detections,
            power_map,
            np.arange(64) * 15.0,
            np.arange(64) * 0.2
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].range_bin, 31)
        self.assertEqual(targets[0].doppler_bin, 31)
        self.assertEqual(targets[0].num_detection_cells, 9)

    def test_extracts_two_separate_targets(self):

        detections = np.zeros((64, 64), dtype=bool)
        power_map = np.ones((64, 64))

        detections[10:12, 10:12] = True
        detections[40:42, 40:42] = True

        power_map[10, 10] = 500
        power_map[40, 40] = 800

        targets = extract_targets(
            detections,
            power_map,
            np.arange(64) * 15.0,
            np.arange(64) * 0.2
        )

        self.assertEqual(len(targets), 2)

    def test_selects_strongest_cell(self):

        detections = np.zeros((32, 32), dtype=bool)
        power_map = np.ones((32, 32))

        detections[15:18, 15:18] = True

        power_map[15, 15] = 100
        power_map[16, 16] = 900
        power_map[17, 17] = 200

        targets = extract_targets(
            detections,
            power_map,
            np.arange(32) * 15.0,
            np.arange(32) * 0.2
        )

        self.assertEqual(targets[0].range_bin, 16)
        self.assertEqual(targets[0].doppler_bin, 16)

    def test_minimum_cluster_size(self):

        detections = np.zeros((32, 32), dtype=bool)
        power_map = np.ones((32, 32))

        detections[10, 10] = True
        detections[20:22, 20:22] = True

        targets = extract_targets(
            detections,
            power_map,
            np.arange(32) * 15.0,
            np.arange(32) * 0.2,
            min_cluster_size=2
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].num_detection_cells, 4)

    def test_empty_detection_map(self):

        detections = np.zeros((32, 32), dtype=bool)
        power_map = np.ones((32, 32))

        targets = extract_targets(
            detections,
            power_map,
            np.arange(32) * 15.0,
            np.arange(32) * 0.2
        )

        self.assertEqual(len(targets), 0)


if __name__ == "__main__":
    unittest.main()
