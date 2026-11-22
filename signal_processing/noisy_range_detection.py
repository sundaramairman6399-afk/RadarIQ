
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# RADARIQ - NOISY RADAR RANGE DETECTION
# ==========================================

# Speed of light (m/s)
C = 3e8

# Sampling settings
FS = 10e6
DURATION = 100e-6
NUM_SAMPLES = int(FS * DURATION)

time = np.arange(NUM_SAMPLES) / FS

# ==========================================
# TARGET CONFIGURATION
# ==========================================

TARGET_DISTANCES = [1500, 3000, 4500]
TARGET_AMPLITUDES = [1.0, 0.7, 0.5]

PULSE_WIDTH = 10

# ==========================================
# NOISE CONFIGURATION
# ==========================================

NOISE_STD = 0.25
rng = np.random.default_rng(42)

# ==========================================
# GENERATE CLEAN RECEIVED ECHOES
# ==========================================

clean_signal = np.zeros(NUM_SAMPLES)

for distance, amplitude in zip(
    TARGET_DISTANCES,
    TARGET_AMPLITUDES
):

    # Calculate round-trip time delay
    time_delay = (2 * distance) / C

    # Convert delay into sample index
    delay_samples = round(time_delay * FS)

    # Generate target echo
    if delay_samples + PULSE_WIDTH <= NUM_SAMPLES:

        clean_signal[
            delay_samples:delay_samples + PULSE_WIDTH
        ] += amplitude

# ==========================================
# ADD GAUSSIAN NOISE
# ==========================================

noise = rng.normal(
    loc=0,
    scale=NOISE_STD,
    size=NUM_SAMPLES
)

received_signal = clean_signal + noise

# ==========================================
# IMPROVED THRESHOLD DETECTION
# ==========================================

THRESHOLD = 0.4
MIN_CONSECUTIVE = 3

# Find samples above threshold
above_threshold = received_signal >= THRESHOLD

# Pad the Boolean array to identify continuous runs
padded = np.concatenate(
    ([False], above_threshold, [False])
)

# Find run start and end indices
run_starts = np.where(
    (~padded[:-1]) & padded[1:]
)[0]

run_ends = np.where(
    padded[:-1] & (~padded[1:])
)[0]

# Keep runs with enough consecutive samples
candidate_detections = []

for start, end in zip(run_starts, run_ends):

    run_length = end - start

    if run_length >= MIN_CONSECUTIVE:
        candidate_detections.append(start)

# Merge nearby detections
detections = []

MERGE_DISTANCE = 2 * PULSE_WIDTH

for detection in candidate_detections:

    if (
        len(detections) == 0
        or detection - detections[-1] > MERGE_DISTANCE
    ):
        detections.append(detection)

# ==========================================
# CALCULATE TARGET DISTANCES
# ==========================================

detected_distances = []

for sample_index in detections:

    measured_delay = sample_index / FS

    calculated_distance = (
        C * measured_delay
    ) / 2

    detected_distances.append(calculated_distance)

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\nRADARIQ - NOISY RADAR RANGE DETECTION")
print("--------------------------------------")

print(f"Sampling frequency: {FS / 1e6:.2f} MHz")
print(f"Noise standard deviation: {NOISE_STD}")
print(f"Detection threshold: {THRESHOLD}")
print(f"Minimum consecutive samples: {MIN_CONSECUTIVE}")

print("\nACTUAL TARGETS")
print("--------------")

for distance in TARGET_DISTANCES:
    print(f"Target distance: {distance:.2f} m")

print("\nDETECTED TARGETS")
print("----------------")

if len(detected_distances) == 0:

    print("No targets detected.")

else:

    for index, distance in enumerate(detected_distances):

        print(
            f"Detection {index + 1}: "
            f"Distance = {distance:.2f} m"
        )

print(f"\nTotal detections: {len(detections)}")
print(f"Actual targets: {len(TARGET_DISTANCES)}")

# ==========================================
# PLOT SIGNALS
# ==========================================

plt.figure(figsize=(13, 6))

plt.plot(
    time * 1e6,
    received_signal,
    label="Received Signal with Noise",
    linewidth=1
)

plt.plot(
    time * 1e6,
    clean_signal,
    label="Clean Echo Signal",
    linewidth=2
)

plt.axhline(
    THRESHOLD,
    linestyle="--",
    label="Detection Threshold"
)

for sample_index in detections:

    plt.axvline(
        x=time[sample_index] * 1e6,
        linestyle=":",
        alpha=0.6
    )

plt.xlabel("Time (microseconds)")
plt.ylabel("Amplitude")

plt.title("RadarIQ - Improved Noisy Range Detection")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

print("\nNOISY RANGE DETECTION COMPLETE")
