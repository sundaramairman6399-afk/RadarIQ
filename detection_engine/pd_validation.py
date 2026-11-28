
"""
RadarIQ - Probability of Detection (Pd) Validation

Purpose:
    Measure how reliably the 2D CA-CFAR detector detects targets
    at different ranges and signal strengths.

Method:
    1. Simulate radar echoes containing one target at a time.
    2. Use a different random seed for every trial.
    3. Process the received pulses into a Range-Doppler map.
    4. Apply 2D CA-CFAR.
    5. Extract detected target clusters.
    6. Check whether a detection is sufficiently close to the
       configured target range and velocity.
    7. Calculate Pd = successful detections / total trials.

Important:
    This script does not modify the existing CFAR implementation.
"""

from dataclasses import replace

from signal_processing.doppler_processing import (
    RadarConfig,
    Target,
    simulate_received_pulses,
    process_range_doppler,
)

from detection_engine.two_dimensional_cfar import ca_cfar_2d
from detection_engine.target_extraction import extract_targets


# ---------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ---------------------------------------------------------

NUM_TRIALS = 100

PFA = 0.01

GUARD_CELLS = (2, 2)
TRAINING_CELLS = (4, 4)

MIN_CLUSTER_SIZE = 2

# A detection is considered successful if its extracted
# range and velocity are sufficiently close to the true target.
RANGE_TOLERANCE_M = 30.0
VELOCITY_TOLERANCE_MPS = 0.5

BASE_CONFIG = RadarConfig()


# ---------------------------------------------------------
# TARGET CONFIGURATION
# ---------------------------------------------------------

TARGETS = [
    Target(
        range_m=1500.0,
        velocity_mps=-1.875,
        amplitude=1.0,
    ),
    Target(
        range_m=3000.0,
        velocity_mps=0.9375,
        amplitude=0.7,
    ),
    Target(
        range_m=4500.0,
        velocity_mps=2.8125,
        amplitude=0.5,
    ),
]


# ---------------------------------------------------------
# CHECK WHETHER A TARGET WAS DETECTED
# ---------------------------------------------------------

def target_was_detected(true_target, detected_targets):
    """
    Compare extracted detections against the known target.

    A detection counts as successful when BOTH conditions hold:

        Absolute range error <= range tolerance

        Absolute velocity error <= velocity tolerance

    This allows small estimation differences without counting
    distant noise clusters as successful detections.
    """

    for detection in detected_targets:

        range_error = abs(
            detection.range_m - true_target.range_m
        )

        velocity_error = abs(
            detection.velocity_mps - true_target.velocity_mps
        )

        if (
            range_error <= RANGE_TOLERANCE_M
            and velocity_error <= VELOCITY_TOLERANCE_MPS
        ):
            return True

    return False


# ---------------------------------------------------------
# RUN MONTE CARLO VALIDATION FOR ONE TARGET
# ---------------------------------------------------------

def validate_target(target, target_number):

    successful_detections = 0

    print("\n" + "=" * 60)
    print(f"TARGET {target_number}")
    print("=" * 60)

    print(f"True range: {target.range_m:.2f} m")
    print(f"True velocity: {target.velocity_mps:+.4f} m/s")
    print(f"Target amplitude: {target.amplitude}")
    print(f"Monte Carlo trials: {NUM_TRIALS}")

    for trial in range(NUM_TRIALS):

        # -------------------------------------------------
        # 1. Generate a new random seed for every trial.
        # -------------------------------------------------

        trial_config = replace(
            BASE_CONFIG,
            random_seed=BASE_CONFIG.random_seed
            + (target_number * NUM_TRIALS)
            + trial,
        )

        # -------------------------------------------------
        # 2. Simulate radar pulses containing this target.
        # -------------------------------------------------

        received_pulses = simulate_received_pulses(
            trial_config,
            [target],
        )

        # -------------------------------------------------
        # 3. Generate the Range-Doppler map.
        # -------------------------------------------------

        result = process_range_doppler(
            received_pulses,
            trial_config,
        )

        # CFAR operates on power, not complex magnitude.
        power_map = abs(result.range_doppler_map) ** 2

        # -------------------------------------------------
        # 4. Apply 2D CA-CFAR.
        # -------------------------------------------------

        detections, threshold_map, noise_map = ca_cfar_2d(
            power_map,
            guard_cells=GUARD_CELLS,
            training_cells=TRAINING_CELLS,
            pfa=PFA,
        )

        # -------------------------------------------------
        # 5. Extract detection clusters.
        # -------------------------------------------------

        extracted_targets = extract_targets(
            detections,
            power_map,
            result.range_axis_m,
            result.velocity_axis_mps,
            min_cluster_size=MIN_CLUSTER_SIZE,
        )

        # -------------------------------------------------
        # 6. Check whether the actual target was detected.
        # -------------------------------------------------

        if target_was_detected(target, extracted_targets):
            successful_detections += 1

        # Display progress every 20 trials.
        if (trial + 1) % 20 == 0:
            print(
                f"Completed {trial + 1}/{NUM_TRIALS} trials"
            )

    # -----------------------------------------------------
    # 7. Calculate Probability of Detection.
    # -----------------------------------------------------

    pd = successful_detections / NUM_TRIALS

    print("\nTARGET RESULTS")
    print(f"Successful detections: {successful_detections}")
    print(f"Failed detections: {NUM_TRIALS - successful_detections}")
    print(f"Probability of Detection: {pd:.4f}")
    print(f"Probability of Detection (%): {pd * 100:.2f}%")

    return pd


# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("RADARIQ - PROBABILITY OF DETECTION VALIDATION")
    print("=" * 60)

    print(f"Monte Carlo trials per target: {NUM_TRIALS}")
    print(f"Desired Pfa: {PFA}")
    print(f"Range tolerance: {RANGE_TOLERANCE_M} m")
    print(f"Velocity tolerance: {VELOCITY_TOLERANCE_MPS} m/s")

    results = []

    # Test every target independently.
    for index, target in enumerate(TARGETS, start=1):

        pd = validate_target(target, index)

        results.append((target, pd))

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL PROBABILITY OF DETECTION RESULTS")
    print("=" * 60)

    print(
        f"{'Range (m)':<15}"
        f"{'Velocity (m/s)':<20}"
        f"{'Amplitude':<15}"
        f"{'Pd (%)':<10}"
    )

    print("-" * 60)

    for target, pd in results:

        print(
            f"{target.range_m:<15.2f}"
            f"{target.velocity_mps:<20.4f}"
            f"{target.amplitude:<15.2f}"
            f"{pd * 100:<10.2f}"
        )

    print("=" * 60)
    print("PD VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
