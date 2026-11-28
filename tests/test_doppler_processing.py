import unittest

import numpy as np

from signal_processing.doppler_processing import (
    RadarConfig,
    Target,
    process_range_doppler,
    simulate_received_pulses,
)


class DopplerProcessingTests(unittest.TestCase):
    def setUp(self):
        self.config = RadarConfig(
            num_pulses=64,
            noise_std=0.0,
            random_seed=7,
        )

    def test_recovers_range_and_positive_negative_velocity_bins(self):
        targets = (
            Target(range_m=1500.0, velocity_mps=-1.875, amplitude=1.0),
            Target(range_m=3000.0, velocity_mps=0.9375, amplitude=0.7),
            Target(range_m=4500.0, velocity_mps=2.8125, amplitude=0.5),
        )
        received = simulate_received_pulses(self.config, targets)
        result = process_range_doppler(
            received,
            self.config,
            apply_window=False,
        )

        for target in targets:
            expected_range_bin = round(
                target.range_m / self.config.range_bin_spacing_m
            )
            observed_doppler_bin = np.argmax(
                np.abs(result.range_doppler_map[:, expected_range_bin])
            )
            observed_velocity = result.velocity_axis_mps[
                observed_doppler_bin
            ]

            self.assertAlmostEqual(
                result.range_axis_m[expected_range_bin],
                target.range_m,
                places=8,
            )
            self.assertAlmostEqual(
                observed_velocity,
                target.velocity_mps,
                places=8,
            )

    def test_velocity_axis_is_symmetric(self):
        received = simulate_received_pulses(
            self.config,
            [Target(range_m=1500.0, velocity_mps=0.0)],
        )
        result = process_range_doppler(received, self.config)

        self.assertAlmostEqual(
            result.velocity_axis_mps[0],
            -self.config.max_unambiguous_velocity_mps,
        )
        self.assertAlmostEqual(
            result.velocity_axis_mps[-1],
            self.config.max_unambiguous_velocity_mps
            - self.config.velocity_bin_spacing_mps,
        )

    def test_rejects_target_outside_fast_time_window(self):
        with self.assertRaises(ValueError):
            simulate_received_pulses(
                self.config,
                [Target(range_m=20_000.0, velocity_mps=0.0)],
            )


if __name__ == "__main__":
    unittest.main()
