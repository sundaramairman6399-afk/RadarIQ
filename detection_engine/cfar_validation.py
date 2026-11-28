
"""
RadarIQ - 2D CA-CFAR Monte Carlo Validation

Purpose:
Measure the empirical Probability of False Alarm (Pfa)
using independent noise-only radar simulations.

Each trial uses a different random seed.
"""

from dataclasses import replace

import numpy as np

from signal_processing.doppler_processing import (
    RadarConfig,
    simulate_received_pulses,
    process_range_doppler,
)

from detection_engine.two_dimensional_cfar import ca_cfar_2d


def run_cfar_validation(
    number_of_trials: int = 100,
    desired_pfa: float = 0.01,
):

    # ----------------------------------------------------------
    # STEP 1: VALIDATE INPUTS
    # ----------------------------------------------------------

    if number_of_trials < 1:
        raise ValueError("Number of trials must be at least 1.")

    if not 0 < desired_pfa < 1:
        raise ValueError("Desired Pfa must be between 0 and 1.")

    # Base configuration contains all radar parameters.
    base_config = RadarConfig()

    total_false_detections = 0
    total_valid_cells = 0

    trial_pfa_values = []

    print("\nRADARIQ - 2D CA-CFAR MONTE CARLO VALIDATION")
    print("============================================")

    print(f"Monte Carlo trials: {number_of_trials}")
    print(f"Desired Pfa: {desired_pfa:.5f}")

    # ----------------------------------------------------------
    # STEP 2: RUN INDEPENDENT NOISE-ONLY TRIALS
    # ----------------------------------------------------------

    for trial in range(number_of_trials):

        # Create a new configuration for this trial.
        #
        # IMPORTANT:
        # Your RadarConfig uses "random_seed", not "seed".
        # Every trial gets a different random seed.

        trial_config = replace(
            base_config,
            random_seed=base_config.random_seed + trial
        )

        # Generate noise-only radar data.
        # Empty targets means no simulated target echoes.
        received_pulses = simulate_received_pulses(
            trial_config,
            targets=[]
        )

        # Generate Range-Doppler map.
        result = process_range_doppler(
            received_pulses,
            trial_config
        )

        # Convert complex values into power.
        power_map = np.abs(
            result.range_doppler_map
        ) ** 2

        # Apply 2D CA-CFAR.
        detections, threshold_map, noise_map = ca_cfar_2d(
            power_map=power_map,
            guard_cells=(2, 2),
            training_cells=(4, 4),
            pfa=desired_pfa
        )

        # Only include cells where a CFAR threshold exists.
        valid_cells = np.isfinite(threshold_map)

        valid_count = np.count_nonzero(valid_cells)

        false_count = np.count_nonzero(
            detections & valid_cells
        )

        # Calculate empirical Pfa for this trial.
        trial_pfa = (
            false_count / valid_count
            if valid_count > 0
            else 0.0
        )

        total_false_detections += false_count
        total_valid_cells += valid_count

        trial_pfa_values.append(trial_pfa)

        print(
            f"Trial {trial + 1:03d}/{number_of_trials} | "
            f"Seed: {trial_config.random_seed} | "
            f"False detections: {false_count:4d} | "
            f"Empirical Pfa: {trial_pfa:.5f}"
        )

    # ----------------------------------------------------------
    # STEP 3: CALCULATE FINAL STATISTICS
    # ----------------------------------------------------------

    empirical_pfa = (
        total_false_detections / total_valid_cells
        if total_valid_cells > 0
        else 0.0
    )

    mean_trial_pfa = float(np.mean(trial_pfa_values))

    std_trial_pfa = float(np.std(trial_pfa_values))

    # ----------------------------------------------------------
    # STEP 4: DISPLAY FINAL RESULTS
    # ----------------------------------------------------------

    print("\nFINAL VALIDATION RESULTS")
    print("========================")

    print(f"Total trials: {number_of_trials}")

    print(f"Total valid cells: {total_valid_cells}")

    print(
        f"Total false detections: "
        f"{total_false_detections}"
    )

    print(f"Desired Pfa: {desired_pfa:.5f}")

    print(f"Empirical Pfa: {empirical_pfa:.5f}")

    print(
        f"Empirical Pfa (%): "
        f"{empirical_pfa * 100:.3f}%"
    )

    print(f"Mean trial Pfa: {mean_trial_pfa:.5f}")

    print(f"Standard deviation: {std_trial_pfa:.5f}")

    print("\nVALIDATION COMPLETE")


if __name__ == "__main__":

    run_cfar_validation(
        number_of_trials=100,
        desired_pfa=0.01
    )
