
import numpy as np


# ==========================================
# RADARIQ: CA-CFAR vs OS-CFAR EVALUATION
# ==========================================

FS = 1000
DURATION = 1
SAMPLES = FS * DURATION

TRIALS = 100

TRAINING_CELLS = 10       # Per side
GUARD_CELLS = 2           # Per side
TOTAL_TRAINING = 2 * TRAINING_CELLS

PFA_DESIRED = 0.01
OS_RANK = 15              # 15th smallest training cell

TARGETS = {
    50: 1.0,
    120: 1.0,
    250: 0.05
}

NOISE_STD = 1.0

rng = np.random.default_rng(42)


# ==========================================
# CALCULATE CA-CFAR MULTIPLIER
# ==========================================

def calculate_ca_alpha(n, pfa):
    return n * (pfa ** (-1 / n) - 1)


# ==========================================
# CALCULATE OS-CFAR MULTIPLIER
# ==========================================

def calculate_os_alpha(n, k, pfa):
    low = 0.0
    high = 100.0

    for _ in range(100):
        alpha = (low + high) / 2

        probability = 1.0

        for j in range(k):
            probability *= (n - j) / (n - j + alpha)

        if probability > pfa:
            low = alpha
        else:
            high = alpha

    return (low + high) / 2


# ==========================================
# CA-CFAR THRESHOLD
# ==========================================

def ca_cfar(power, alpha):
    thresholds = np.full(len(power), np.inf)

    start = TRAINING_CELLS + GUARD_CELLS
    end = len(power) - start

    for i in range(start, end):

        left = power[
            i - GUARD_CELLS - TRAINING_CELLS:
            i - GUARD_CELLS
        ]

        right = power[
            i + GUARD_CELLS + 1:
            i + GUARD_CELLS + TRAINING_CELLS + 1
        ]

        training = np.concatenate((left, right))

        noise_estimate = np.mean(training)

        thresholds[i] = alpha * noise_estimate

    return thresholds


# ==========================================
# OS-CFAR THRESHOLD
# ==========================================

def os_cfar(power, alpha):
    thresholds = np.full(len(power), np.inf)

    start = TRAINING_CELLS + GUARD_CELLS
    end = len(power) - start

    for i in range(start, end):

        left = power[
            i - GUARD_CELLS - TRAINING_CELLS:
            i - GUARD_CELLS
        ]

        right = power[
            i + GUARD_CELLS + 1:
            i + GUARD_CELLS + TRAINING_CELLS + 1
        ]

        training = np.concatenate((left, right))

        sorted_training = np.sort(training)

        noise_estimate = sorted_training[OS_RANK - 1]

        thresholds[i] = alpha * noise_estimate

    return thresholds


# ==========================================
# GENERATE FFT POWER
# ==========================================

def get_fft_power(include_targets):

    time = np.arange(SAMPLES) / FS

    signal = np.zeros(SAMPLES)

    if include_targets:
        for frequency, amplitude in TARGETS.items():
            signal += amplitude * np.sin(
                2 * np.pi * frequency * time
            )

    noise = rng.normal(0, NOISE_STD, SAMPLES)

    received_signal = signal + noise

    fft_result = np.fft.rfft(received_signal)

    power = np.abs(fft_result) ** 2

    frequencies = np.fft.rfftfreq(SAMPLES, 1 / FS)

    return frequencies, power


# ==========================================
# SETUP CFAR PARAMETERS
# ==========================================

ca_alpha = calculate_ca_alpha(
    TOTAL_TRAINING,
    PFA_DESIRED
)

os_alpha = calculate_os_alpha(
    TOTAL_TRAINING,
    OS_RANK,
    PFA_DESIRED
)

print("\nRADARIQ - 100 TRIAL CFAR EVALUATION")
print("-----------------------------------")

print("Number of trials:", TRIALS)
print("Training cells:", TOTAL_TRAINING)
print("Guard cells:", GUARD_CELLS * 2)
print("Desired Pfa:", PFA_DESIRED)

print(f"\nCA-CFAR alpha: {ca_alpha:.3f}")
print(f"OS-CFAR alpha: {os_alpha:.3f}")


# ==========================================
# EVALUATION COUNTERS
# ==========================================

methods = {
    "CA-CFAR": {
        "false_alarms": 0,
        "total_noise_cells": 0,
        "detections": {f: 0 for f in TARGETS}
    },

    "OS-CFAR": {
        "false_alarms": 0,
        "total_noise_cells": 0,
        "detections": {f: 0 for f in TARGETS}
    }
}

alphas = {
    "CA-CFAR": ca_alpha,
    "OS-CFAR": os_alpha
}

detectors = {
    "CA-CFAR": ca_cfar,
    "OS-CFAR": os_cfar
}


# ==========================================
# RUN 100 TRIALS
# ==========================================

for trial in range(TRIALS):

    # --------------------------------------
    # A. NOISE-ONLY TRIAL: MEASURE FALSE ALARMS
    # --------------------------------------

    frequencies, noise_power = get_fft_power(
        include_targets=False
    )

    for name in methods:

        thresholds = detectors[name](
            noise_power,
            alphas[name]
        )

        valid = np.isfinite(thresholds)

        false_alarms = np.sum(
            noise_power[valid] > thresholds[valid]
        )

        methods[name]["false_alarms"] += int(false_alarms)

        methods[name]["total_noise_cells"] += int(
            np.sum(valid)
        )

    # --------------------------------------
    # B. TARGET-PRESENT TRIAL: MEASURE PD
    # --------------------------------------

    frequencies, target_power = get_fft_power(
        include_targets=True
    )

    for name in methods:

        thresholds = detectors[name](
            target_power,
            alphas[name]
        )

        for target_frequency in TARGETS:

            # Search within +/- 2 Hz of the target
            target_window = (
                (frequencies >= target_frequency - 2) &
                (frequencies <= target_frequency + 2)
            )

            detected = np.any(
                target_power[target_window] >
                thresholds[target_window]
            )

            if detected:
                methods[name]["detections"][target_frequency] += 1


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n\nFALSE ALARM RESULTS")
print("-------------------")

for name in methods:

    false_alarms = methods[name]["false_alarms"]

    total_cells = methods[name]["total_noise_cells"]

    measured_pfa = false_alarms / total_cells

    print(f"\n{name}")
    print("False alarms:", false_alarms)
    print("Total noise cells tested:", total_cells)
    print(f"Measured Pfa: {measured_pfa:.5f}")
    print(f"Measured Pfa (%): {measured_pfa * 100:.3f}%")


print("\n\nPROBABILITY OF DETECTION")
print("------------------------")

for name in methods:

    print(f"\n{name}")

    for target_frequency in TARGETS:

        detections = methods[name]["detections"][
            target_frequency
        ]

        pd = detections / TRIALS

        print(
            f"{target_frequency} Hz target: "
            f"{detections}/{TRIALS} "
            f"({pd * 100:.1f}%)"
        )


print("\n\nEVALUATION COMPLETE")
print("-------------------")
