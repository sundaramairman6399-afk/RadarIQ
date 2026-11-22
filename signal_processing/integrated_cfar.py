
import numpy as np


# ==========================================
# RADARIQ - COHERENT INTEGRATION + CFAR
# ==========================================

# Signal settings
FS = 1000
DURATION = 1.0
SAMPLES = int(FS * DURATION)

TARGET_FREQUENCIES = [50, 120, 250]
TARGET_AMPLITUDES = [1.0, 1.0, 0.05]

NOISE_STD = 1.0

# CFAR settings
TRAINING_CELLS = 10
GUARD_CELLS = 2
TOTAL_TRAINING = TRAINING_CELLS * 2

PFA_DESIRED = 0.01
OS_RANK = 15

# Evaluation settings
NUM_TRIALS = 100
NUM_PULSES = 10
CALIBRATION_SAMPLES = 200_000

# Target detection window
TARGET_WINDOW_HZ = 2


# ==========================================
# CALCULATE CFAR ALPHA
# ==========================================

def calibrate_alpha(num_pulses, method):
    """
    Calibrate alpha using Monte Carlo simulation.

    The sum of independent complex Gaussian noise
    samples remains complex Gaussian.

    Therefore, its squared magnitude follows an
    exponential distribution, not Gamma(shape=N_pulses).
    """

    calibration_rng = np.random.default_rng(1000 + num_pulses)

    # Correct noise-power distribution after coherent integration
    cut_power = calibration_rng.exponential(
        scale=num_pulses,
        size=CALIBRATION_SAMPLES
    )

    training_power = calibration_rng.exponential(
        scale=num_pulses,
        size=(CALIBRATION_SAMPLES, TOTAL_TRAINING)
    )

    if method == "CA-CFAR":

        noise_estimate = np.mean(training_power, axis=1)

    elif method == "OS-CFAR":

        sorted_training = np.sort(training_power, axis=1)

        noise_estimate = sorted_training[:, OS_RANK - 1]

    else:
        raise ValueError("Unknown CFAR method")

    ratios = cut_power / noise_estimate

    alpha = np.quantile(
        ratios,
        1 - PFA_DESIRED
    )

    return alpha


# ==========================================
# GENERATE SIGNAL FFT
# ==========================================

def generate_target_fft():

    time = np.arange(SAMPLES) / FS

    signal = np.zeros(SAMPLES)

    for frequency, amplitude in zip(
        TARGET_FREQUENCIES,
        TARGET_AMPLITUDES
    ):

        signal += amplitude * np.sin(
            2 * np.pi * frequency * time
        )

    return np.fft.rfft(signal)


# ==========================================
# GENERATE NOISE AND INTEGRATE PULSES
# ==========================================

def generate_integrated_fft(
    num_pulses,
    rng,
    include_targets=False
):

    # Independent noise for every pulse
    noise = rng.normal(
        0,
        NOISE_STD,
        size=(num_pulses, SAMPLES)
    )

    # FFT of every pulse
    noise_fft = np.fft.rfft(
        noise,
        axis=1
    )

    # Coherent integration of complex FFT values
    integrated_fft = np.sum(
        noise_fft,
        axis=0
    )

    if include_targets:

        target_fft = generate_target_fft()

        # Perfect phase alignment across pulses
        integrated_fft += num_pulses * target_fft

    return integrated_fft


# ==========================================
# CA-CFAR / OS-CFAR DETECTION
# ==========================================

def cfar_detection(power, alpha, method):

    num_cells = len(power)

    detections = np.zeros(
        num_cells,
        dtype=bool
    )

    thresholds = np.full(
        num_cells,
        np.nan
    )

    start_index = TRAINING_CELLS + GUARD_CELLS

    end_index = (
        num_cells -
        TRAINING_CELLS -
        GUARD_CELLS
    )

    for i in range(start_index, end_index):

        # Training cells before CUT
        leading_cells = power[
            i - GUARD_CELLS - TRAINING_CELLS:
            i - GUARD_CELLS
        ]

        # Training cells after CUT
        lagging_cells = power[
            i + GUARD_CELLS + 1:
            i + GUARD_CELLS + 1 + TRAINING_CELLS
        ]

        training = np.concatenate(
            (leading_cells, lagging_cells)
        )

        # Estimate background noise
        if method == "CA-CFAR":

            noise_estimate = np.mean(training)

        elif method == "OS-CFAR":

            noise_estimate = np.sort(training)[
                OS_RANK - 1
            ]

        else:
            raise ValueError("Unknown CFAR method")

        threshold = alpha * noise_estimate

        thresholds[i] = threshold

        if power[i] > threshold:

            detections[i] = True

    return detections, thresholds


# ==========================================
# EVALUATE FALSE ALARM PROBABILITY
# ==========================================

def evaluate_false_alarms(
    num_pulses,
    alpha,
    method
):

    rng = np.random.default_rng(
        2000 + num_pulses
    )

    total_false_alarms = 0
    total_noise_cells = 0

    for _ in range(NUM_TRIALS):

        # Noise-only integrated FFT
        integrated_fft = generate_integrated_fft(
            num_pulses,
            rng,
            include_targets=False
        )

        power = np.abs(integrated_fft) ** 2

        detections, _ = cfar_detection(
            power,
            alpha,
            method
        )

        start_index = TRAINING_CELLS + GUARD_CELLS
        end_index = len(power) - TRAINING_CELLS - GUARD_CELLS

        valid_detections = detections[
            start_index:end_index
        ]

        total_false_alarms += np.count_nonzero(
            valid_detections
        )

        total_noise_cells += len(valid_detections)

    measured_pfa = (
        total_false_alarms / total_noise_cells
    )

    return (
        total_false_alarms,
        total_noise_cells,
        measured_pfa
    )


# ==========================================
# EVALUATE PROBABILITY OF DETECTION
# ==========================================

def evaluate_detection_probability(
    num_pulses,
    alpha,
    method
):

    rng = np.random.default_rng(
        3000 + num_pulses
    )

    detection_counts = {
        frequency: 0
        for frequency in TARGET_FREQUENCIES
    }

    frequency_axis = np.fft.rfftfreq(
        SAMPLES,
        d=1 / FS
    )

    for _ in range(NUM_TRIALS):

        # Generate target-present integrated signal
        integrated_fft = generate_integrated_fft(
            num_pulses,
            rng,
            include_targets=True
        )

        power = np.abs(integrated_fft) ** 2

        detections, _ = cfar_detection(
            power,
            alpha,
            method
        )

        for frequency in TARGET_FREQUENCIES:

            target_bins = np.where(
                np.abs(frequency_axis - frequency)
                <= TARGET_WINDOW_HZ
            )[0]

            if np.any(detections[target_bins]):

                detection_counts[frequency] += 1

    detection_probabilities = {}

    for frequency in TARGET_FREQUENCIES:

        detection_probabilities[frequency] = (
            detection_counts[frequency] / NUM_TRIALS
        ) * 100

    return detection_probabilities


# ==========================================
# RUN COMPLETE EXPERIMENT
# ==========================================

def run_experiment():

    print("\nRADARIQ - COHERENT INTEGRATION + CFAR")
    print("--------------------------------------")

    print(f"Number of trials: {NUM_TRIALS}")
    print(f"Coherent pulses: {NUM_PULSES}")
    print(f"Desired Pfa: {PFA_DESIRED}")

    methods = ["CA-CFAR", "OS-CFAR"]
    pulse_settings = [1, NUM_PULSES]

    results = {}

    # ======================================
    # RUN ALL EXPERIMENTS
    # ======================================

    for num_pulses in pulse_settings:

        for method in methods:

            print(
                f"\nEvaluating {method} "
                f"with {num_pulses} pulse(s)..."
            )

            alpha = calibrate_alpha(
                num_pulses,
                method
            )

            false_alarms, total_cells, measured_pfa = (
                evaluate_false_alarms(
                    num_pulses,
                    alpha,
                    method
                )
            )

            detection_probabilities = (
                evaluate_detection_probability(
                    num_pulses,
                    alpha,
                    method
                )
            )

            results[(method, num_pulses)] = {
                "alpha": alpha,
                "false_alarms": false_alarms,
                "total_cells": total_cells,
                "pfa": measured_pfa,
                "pd": detection_probabilities
            }

    # ======================================
    # FALSE ALARM COMPARISON
    # ======================================

    print("\n\nFALSE ALARM COMPARISON")
    print("----------------------")

    for method, num_pulses in [
        ("CA-CFAR", 1),
        ("OS-CFAR", 1),
        ("CA-CFAR", NUM_PULSES),
        ("OS-CFAR", NUM_PULSES)
    ]:

        result = results[(method, num_pulses)]

        print(f"\n{method} | Pulses: {num_pulses}")

        print(f"Calibrated alpha: {result['alpha']:.3f}")

        print(
            f"False alarms: {result['false_alarms']}"
        )

        print(
            f"Total noise cells: {result['total_cells']}"
        )

        print(
            f"Measured Pfa: {result['pfa']:.5f}"
        )

        print(
            f"Measured Pfa (%): {result['pfa'] * 100:.3f}%"
        )

    # ======================================
    # PROBABILITY OF DETECTION
    # ======================================

    print("\n\nPROBABILITY OF DETECTION")
    print("------------------------")

    for method, num_pulses in [
        ("CA-CFAR", 1),
        ("OS-CFAR", 1),
        ("CA-CFAR", NUM_PULSES),
        ("OS-CFAR", NUM_PULSES)
    ]:

        result = results[(method, num_pulses)]

        print(f"\n{method} | Pulses: {num_pulses}")

        for frequency in TARGET_FREQUENCIES:

            pd = result["pd"][frequency]

            detected = round(
                pd * NUM_TRIALS / 100
            )

            print(
                f"{frequency} Hz: "
                f"{detected}/{NUM_TRIALS} "
                f"({pd:.1f}%)"
            )

    # ======================================
    # WEAK TARGET IMPROVEMENT
    # ======================================

    print("\n\nWEAK TARGET IMPROVEMENT")
    print("-----------------------")

    weak_frequency = 250

    for method in methods:

        single_pd = results[
            (method, 1)
        ]["pd"][weak_frequency]

        integrated_pd = results[
            (method, NUM_PULSES)
        ]["pd"][weak_frequency]

        improvement = integrated_pd - single_pd

        print(f"\n{method}")

        print(
            f"Single-pulse Pd: {single_pd:.1f}%"
        )

        print(
            f"Integrated Pd: {integrated_pd:.1f}%"
        )

        print(
            f"Improvement: {improvement:.1f} percentage points"
        )

    print("\nINTEGRATED CFAR EVALUATION COMPLETE")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    run_experiment()
