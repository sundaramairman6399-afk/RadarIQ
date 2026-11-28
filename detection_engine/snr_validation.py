
"""
RadarIQ - SNR versus Probability of Detection Validation

Purpose:
    Study how target signal strength affects the probability
    that the 2D CA-CFAR detector successfully detects a target.

Experiment:
    1. Keep noise level fixed.
    2. Change target amplitude.
    3. Run independent Monte Carlo trials.
    4. Measure target detection probability.
    5. Estimate input echo-to-noise ratio.
    6. Plot Pd versus SNR.

SNR is measured on raw received complex samples,
before matched filtering.
"""

from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from signal_processing.doppler_processing import (
    RadarConfig,
    Target,
    simulate_received_pulses,
    process_range_doppler,
)

from detection_engine.two_dimensional_cfar import ca_cfar_2d
from detection_engine.target_extraction import extract_targets


# =========================================================
# 1. EXPERIMENT CONFIGURATION
# =========================================================

NUM_TRIALS = 100

PFA = 0.01

GUARD_CELLS = (2, 2)
TRAINING_CELLS = (4, 4)

MIN_CLUSTER_SIZE = 2

RANGE_TOLERANCE_M = 30.0
VELOCITY_TOLERANCE_MPS = 0.5

# Test much weaker signals than the previous experiment.
AMPLITUDES = [
    0.005,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
]

BASE_CONFIG = RadarConfig()

TARGET_RANGE_M = 1500.0
TARGET_VELOCITY_MPS = -1.875


# =========================================================
# 2. CALCULATE INPUT SIGNAL-TO-NOISE RATIO
# =========================================================

def calculate_input_snr_db(received_with_target, noise_only):
    """
    Estimate raw received-sample SNR.

    received_with_target = target echo + noise
    noise_only = noise

    Subtracting the two signals isolates the target echo.

    SNR = target echo power / noise power

    SNR in dB = 10 * log10(SNR)
    """

    target_echo = received_with_target - noise_only

    signal_power = np.mean(np.abs(target_echo) ** 2)

    noise_power = np.mean(np.abs(noise_only) ** 2)

    if noise_power <= 0:
        raise ValueError("Noise power must be positive.")

    snr_linear = signal_power / noise_power

    snr_db = 10.0 * np.log10(snr_linear)

    return snr_db


# =========================================================
# 3. CHECK WHETHER THE TARGET WAS DETECTED
# =========================================================

def target_was_detected(true_target, detected_targets):
    """
    A detection is successful when BOTH conditions hold:

        Range error <= range tolerance

        Velocity error <= velocity tolerance

    This prevents distant noise clusters from being counted
    as successful target detections.
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


# =========================================================
# 4. RUN MONTE CARLO EXPERIMENT FOR ONE AMPLITUDE
# =========================================================

def validate_amplitude(amplitude, amplitude_index):

    successful_detections = 0

    measured_snr_values = []

    true_target = Target(
        range_m=TARGET_RANGE_M,
        velocity_mps=TARGET_VELOCITY_MPS,
        amplitude=amplitude,
    )

    print("\n" + "=" * 60)
    print(f"TESTING TARGET AMPLITUDE: {amplitude}")
    print("=" * 60)

    for trial in range(NUM_TRIALS):

        # Generate a different random seed for every trial.
        #
        # The same seed is used for the target and noise-only
        # simulations within one trial. This ensures that
        # both simulations contain identical noise.

        trial_seed = (
            BASE_CONFIG.random_seed
            + trial
            + 10000 * amplitude_index
        )

        trial_config = replace(
            BASE_CONFIG,
            random_seed=trial_seed,
        )

        # -------------------------------------------------
        # A. Generate noise-only received pulses.
        # -------------------------------------------------

        noise_only = simulate_received_pulses(
            trial_config,
            [],
        )

        # -------------------------------------------------
        # B. Generate received pulses containing the target.
        # -------------------------------------------------

        received_with_target = simulate_received_pulses(
            trial_config,
            [true_target],
        )

        # -------------------------------------------------
        # C. Measure input SNR.
        # -------------------------------------------------

        snr_db = calculate_input_snr_db(
            received_with_target,
            noise_only,
        )

        measured_snr_values.append(snr_db)

        # -------------------------------------------------
        # D. Generate the Range-Doppler map.
        # -------------------------------------------------

        result = process_range_doppler(
            received_with_target,
            trial_config,
        )

        # Convert complex Range-Doppler values into power.
        power_map = np.abs(result.range_doppler_map) ** 2

        # -------------------------------------------------
        # E. Apply 2D CA-CFAR.
        # -------------------------------------------------

        detections, threshold_map, noise_map = ca_cfar_2d(
            power_map,
            guard_cells=GUARD_CELLS,
            training_cells=TRAINING_CELLS,
            pfa=PFA,
        )

        # -------------------------------------------------
        # F. Extract detection clusters.
        # -------------------------------------------------

        extracted_targets = extract_targets(
            detections,
            power_map,
            result.range_axis_m,
            result.velocity_axis_mps,
            min_cluster_size=MIN_CLUSTER_SIZE,
        )

        # -------------------------------------------------
        # G. Check whether the actual target was detected.
        # -------------------------------------------------

        if target_was_detected(true_target, extracted_targets):
            successful_detections += 1

        # Display progress every 20 trials.
        if (trial + 1) % 20 == 0:
            print(
                f"Completed {trial + 1}/{NUM_TRIALS} trials"
            )

    # -----------------------------------------------------
    # H. Calculate Pd and average input SNR.
    # -----------------------------------------------------

    pd = successful_detections / NUM_TRIALS

    average_snr_db = float(np.mean(measured_snr_values))

    print("\nAMPLITUDE RESULTS")
    print(f"Amplitude: {amplitude}")
    print(f"Average input SNR: {average_snr_db:.2f} dB")
    print(f"Successful detections: {successful_detections}")
    print(f"Failed detections: {NUM_TRIALS - successful_detections}")
    print(f"Probability of Detection: {pd:.4f}")
    print(f"Probability of Detection (%): {pd * 100:.2f}%")

    return average_snr_db, pd


# =========================================================
# 5. MAIN EXPERIMENT
# =========================================================

def main():

    print("=" * 65)
    print("RADARIQ - SNR VERSUS PROBABILITY OF DETECTION")
    print("=" * 65)

    print(f"Monte Carlo trials per amplitude: {NUM_TRIALS}")
    print(f"Number of amplitude levels: {len(AMPLITUDES)}")
    print(f"Total trials: {NUM_TRIALS * len(AMPLITUDES)}")
    print(f"Desired Pfa: {PFA}")

    print(f"Target range: {TARGET_RANGE_M} m")
    print(f"Target velocity: {TARGET_VELOCITY_MPS} m/s")

    snr_results = []
    pd_results = []

    # Test every amplitude independently.
    for index, amplitude in enumerate(AMPLITUDES):

        average_snr, pd = validate_amplitude(
            amplitude,
            index + 1,
        )

        snr_results.append(average_snr)
        pd_results.append(pd)

    # -----------------------------------------------------
    # 6. PRINT FINAL RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL SNR VERSUS PD RESULTS")
    print("=" * 65)

    print(
        f"{'Amplitude':<15}"
        f"{'Input SNR (dB)':<20}"
        f"{'Pd (%)':<15}"
    )

    print("-" * 65)

    for amplitude, snr, pd in zip(
        AMPLITUDES,
        snr_results,
        pd_results,
    ):

        print(
            f"{amplitude:<15.3f}"
            f"{snr:<20.2f}"
            f"{pd * 100:<15.2f}"
        )

    # -----------------------------------------------------
    # 7. PLOT SNR VERSUS PROBABILITY OF DETECTION
    # -----------------------------------------------------

    # Sort results by SNR for a clear graph.
    sorted_indices = np.argsort(snr_results)

    sorted_snr = np.array(snr_results)[sorted_indices]

    sorted_pd = np.array(pd_results)[sorted_indices]

    plt.figure(figsize=(10, 6))

    plt.plot(
        sorted_snr,
        sorted_pd * 100,
        marker="o",
        linewidth=2,
        label="2D CA-CFAR",
    )

    plt.xlabel("Input SNR (dB)")
    plt.ylabel("Probability of Detection (%)")

    plt.title("RadarIQ - Pd versus SNR Performance")

    plt.ylim(-5, 105)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Save the graph in the project root directory.
    plt.savefig(
        "pd_vs_snr.png",
        dpi=300,
    )

    print("\nGraph saved as: pd_vs_snr.png")

    plt.show()

    print("\nSNR VALIDATION COMPLETE")


# =========================================================
# 8. PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
