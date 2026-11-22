import numpy as np

# ============================================================
# RADARIQ - FULL-PROFILE RANGE-DOMAIN CA-CFAR EVALUATION
# ============================================================

# Radar configuration
C = 3e8
FS = 10e6
DURATION = 100e-6
NUM_SAMPLES = int(FS * DURATION)

TARGET_DISTANCES = [1500, 3000, 4500]
TARGET_AMPLITUDES = [1.0, 0.7, 0.5]

PULSE_WIDTH = 10
NOISE_STD = 0.25
TRANSMITTED_PULSE = np.ones(PULSE_WIDTH)

# CA-CFAR configuration
TRAINING_CELLS_PER_SIDE = 10
GUARD_CELLS_PER_SIDE = 10
PFA_DESIRED = 0.01

# Monte Carlo settings
CALIBRATION_TRIALS = 2000
VALIDATION_TRIALS = 500
DETECTION_TRIALS = 500

# A detection within +/- 2 range bins is counted as a hit.
DETECTION_TOLERANCE_BINS = 2


# ============================================================
# RANGE-PROFILE AND CFAR INDICES
# ============================================================

PROFILE_LENGTH = NUM_SAMPLES - PULSE_WIDTH + 1

TOTAL_TRAINING_CELLS = 2 * TRAINING_CELLS_PER_SIDE
EDGE_CELLS = TRAINING_CELLS_PER_SIDE + GUARD_CELLS_PER_SIDE

# Only CUTs with a complete training window on both sides.
CUT_INDICES = np.arange(
    EDGE_CELLS,
    PROFILE_LENGTH - EDGE_CELLS
)

LEFT_TRAINING_OFFSETS = np.arange(
    -(TRAINING_CELLS_PER_SIDE + GUARD_CELLS_PER_SIDE),
    -GUARD_CELLS_PER_SIDE
)

RIGHT_TRAINING_OFFSETS = np.arange(
    GUARD_CELLS_PER_SIDE + 1,
    GUARD_CELLS_PER_SIDE + TRAINING_CELLS_PER_SIDE + 1
)

TRAINING_OFFSETS = np.concatenate(
    (LEFT_TRAINING_OFFSETS, RIGHT_TRAINING_OFFSETS)
)

TRAINING_INDICES = CUT_INDICES[:, None] + TRAINING_OFFSETS


# ============================================================
# MATCHED FILTER
# ============================================================

def apply_matched_filter(signal):
    """Apply a rectangular-pulse matched filter using valid convolution."""
    return np.convolve(
        signal,
        TRANSMITTED_PULSE[::-1],
        mode="valid"
    )


# ============================================================
# NOISE-ONLY RANGE PROFILE
# ============================================================

def generate_noise_profile(rng):
    noise = rng.normal(
        loc=0.0,
        scale=NOISE_STD,
        size=NUM_SAMPLES
    )

    matched_output = apply_matched_filter(noise)
    return matched_output ** 2


# ============================================================
# CA-CFAR NOISE ESTIMATE
# ============================================================

def calculate_noise_estimates(range_power):
    training_samples = range_power[TRAINING_INDICES]
    return np.mean(training_samples, axis=1)


# ============================================================
# EMPIRICAL CALIBRATION OF THE CFAR MULTIPLIER
# ============================================================

def calibrate_alpha():
    """
    Estimate alpha from the empirical CUT/noise-estimate ratio
    distribution under noise-only conditions.

    All eligible CUT positions from each calibration profile
    contribute to the empirical distribution.
    """
    rng = np.random.default_rng(1000)
    ratio_batches = []

    for _ in range(CALIBRATION_TRIALS):
        range_power = generate_noise_profile(rng)

        cut_power = range_power[CUT_INDICES]
        noise_estimate = calculate_noise_estimates(range_power)

        ratios = cut_power / np.maximum(
            noise_estimate,
            np.finfo(float).tiny
        )

        ratio_batches.append(ratios)

    all_ratios = np.concatenate(ratio_batches)

    alpha = np.quantile(
        all_ratios,
        1.0 - PFA_DESIRED,
        method="higher"
    )

    return alpha


# ============================================================
# APPLY CA-CFAR TO ONE RANGE PROFILE
# ============================================================

def apply_cfar(range_power, alpha):
    noise_estimate = calculate_noise_estimates(range_power)
    thresholds = alpha * noise_estimate

    cut_power = range_power[CUT_INDICES]
    detection_mask = cut_power > thresholds

    return detection_mask, thresholds


# ============================================================
# GROUP ADJACENT DETECTED CELLS INTO DETECTION CLUSTERS
# ============================================================

def group_detections(range_power, detection_mask):
    detected_indices = CUT_INDICES[detection_mask]

    if detected_indices.size == 0:
        return []

    groups = np.split(
        detected_indices,
        np.where(np.diff(detected_indices) > 1)[0] + 1
    )

    detection_peaks = []

    for group in groups:
        strongest_index = group[
            np.argmax(range_power[group])
        ]
        detection_peaks.append(int(strongest_index))

    return detection_peaks


# ============================================================
# TARGET-PRESENT RECEIVED SIGNAL
# ============================================================

def generate_target_signal(rng):
    received_signal = rng.normal(
        loc=0.0,
        scale=NOISE_STD,
        size=NUM_SAMPLES
    )

    target_sample_indices = []

    for distance, amplitude in zip(
        TARGET_DISTANCES,
        TARGET_AMPLITUDES
    ):
        round_trip_delay = (2.0 * distance) / C
        sample_index = int(round(round_trip_delay * FS))

        if sample_index + PULSE_WIDTH > NUM_SAMPLES:
            raise ValueError(
                f"Target at {distance} m exceeds the signal duration."
            )

        target_sample_indices.append(sample_index)

        received_signal[
            sample_index:sample_index + PULSE_WIDTH
        ] += amplitude * TRANSMITTED_PULSE

    return received_signal, target_sample_indices


# ============================================================
# FULL-PROFILE FALSE-ALARM VALIDATION
# ============================================================

def evaluate_false_alarms(alpha):
    rng = np.random.default_rng(2000)

    total_false_alarm_cells = 0
    total_tested_cells = 0
    total_false_alarm_clusters = 0

    for _ in range(VALIDATION_TRIALS):
        range_power = generate_noise_profile(rng)

        detection_mask, _ = apply_cfar(range_power, alpha)

        detection_clusters = group_detections(
            range_power,
            detection_mask
        )

        total_false_alarm_cells += int(
            np.count_nonzero(detection_mask)
        )

        total_tested_cells += len(CUT_INDICES)
        total_false_alarm_clusters += len(detection_clusters)

    measured_pfa = total_false_alarm_cells / total_tested_cells

    average_false_alarm_clusters = (
        total_false_alarm_clusters / VALIDATION_TRIALS
    )

    return (
        measured_pfa,
        average_false_alarm_clusters,
        total_false_alarm_cells,
        total_tested_cells
    )


# ============================================================
# PROBABILITY OF DETECTION EVALUATION
# ============================================================

def evaluate_detection_probability(alpha):
    rng = np.random.default_rng(3000)

    detected_targets = np.zeros(
        len(TARGET_DISTANCES),
        dtype=int
    )

    total_false_alarm_clusters = 0

    for _ in range(DETECTION_TRIALS):
        received_signal, target_indices = generate_target_signal(rng)

        matched_output = apply_matched_filter(received_signal)
        range_power = matched_output ** 2

        detection_mask, _ = apply_cfar(range_power, alpha)

        detection_clusters = group_detections(
            range_power,
            detection_mask
        )

        # Count a target as detected if a cluster peak is
        # within the specified tolerance of its range bin.
        for target_number, target_index in enumerate(target_indices):
            target_was_detected = any(
                abs(detection - target_index)
                <= DETECTION_TOLERANCE_BINS
                for detection in detection_clusters
            )

            if target_was_detected:
                detected_targets[target_number] += 1

        # Count clusters that are not close to any actual target.
        for detection in detection_clusters:
            near_any_target = any(
                abs(detection - target_index)
                <= DETECTION_TOLERANCE_BINS
                for target_index in target_indices
            )

            if not near_any_target:
                total_false_alarm_clusters += 1

    detection_probabilities = (
        detected_targets / DETECTION_TRIALS
    ) * 100.0

    average_false_clusters_with_targets = (
        total_false_alarm_clusters / DETECTION_TRIALS
    )

    return (
        detected_targets,
        detection_probabilities,
        average_false_clusters_with_targets
    )


# ============================================================
# MAIN
# ============================================================

def main():
    theoretical_alpha = TOTAL_TRAINING_CELLS * (
        PFA_DESIRED ** (-1.0 / TOTAL_TRAINING_CELLS) - 1.0
    )

    range_resolution = C / (2.0 * FS)

    print("\nRADARIQ - FULL-PROFILE RANGE-DOMAIN CA-CFAR")
    print("-------------------------------------------")
    print(f"Sampling frequency: {FS / 1e6:.2f} MHz")
    print(f"Noise standard deviation: {NOISE_STD}")
    print(f"Range-bin spacing: {range_resolution:.2f} m")

    print("\nCFAR SETTINGS")
    print("-------------")
    print(
        f"Training cells per side: "
        f"{TRAINING_CELLS_PER_SIDE}"
    )
    print(
        f"Guard cells per side: "
        f"{GUARD_CELLS_PER_SIDE}"
    )
    print(f"Total training cells: {TOTAL_TRAINING_CELLS}")
    print(f"Desired Pfa: {PFA_DESIRED:.4f}")
    print(f"Eligible CUTs per profile: {len(CUT_INDICES)}")

    print("\nTHRESHOLD CALIBRATION")
    print("---------------------")
    print(f"Calibration trials: {CALIBRATION_TRIALS}")

    alpha = calibrate_alpha()

    print(f"Theoretical multiplier (reference): {theoretical_alpha:.3f}")
    print(f"Empirically calibrated multiplier: {alpha:.3f}")

    print("\nFULL-PROFILE FALSE-ALARM VALIDATION")
    print("-----------------------------------")

    (
        measured_pfa,
        average_false_alarm_clusters,
        false_alarm_cells,
        tested_cells
    ) = evaluate_false_alarms(alpha)

    print(f"Validation trials: {VALIDATION_TRIALS}")
    print(f"False-alarm cells: {false_alarm_cells}")
    print(f"Total tested cells: {tested_cells}")
    print(f"Measured cell-level Pfa: {measured_pfa:.5f}")
    print(f"Measured cell-level Pfa: {measured_pfa * 100:.3f}%")
    print(
        "Average false detection clusters/profile: "
        f"{average_false_alarm_clusters:.3f}"
    )

    print("\nPROBABILITY OF DETECTION")
    print("------------------------")

    (
        detected_targets,
        detection_probabilities,
        average_false_clusters_with_targets
    ) = evaluate_detection_probability(alpha)

    for i, distance in enumerate(TARGET_DISTANCES):
        print(
            f"Target at {distance} m: "
            f"{detected_targets[i]}/{DETECTION_TRIALS} "
            f"({detection_probabilities[i]:.2f}%)"
        )

    print(
        "\nAverage false detection clusters/profile "
        "with targets: "
        f"{average_false_clusters_with_targets:.3f}"
    )

    print("\nFULL-PROFILE RANGE CA-CFAR EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
