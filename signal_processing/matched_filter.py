
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags, find_peaks

# -----------------------------------------
# RADARIQ - MATCHED FILTERING
# -----------------------------------------

C = 3e8
FS = 10e6
DURATION = 100e-6
NUM_SAMPLES = int(FS * DURATION)

TARGET_DISTANCES = [1500, 3000, 4500]
TARGET_AMPLITUDES = [1.0, 0.7, 0.5]

PULSE_WIDTH = 10
NOISE_STD = 0.25
RANDOM_SEED = 42

# -----------------------------------------
# STEP 1: GENERATE TRANSMITTED PULSE
# -----------------------------------------

transmitted_pulse = np.ones(PULSE_WIDTH)

# -----------------------------------------
# STEP 2: GENERATE RECEIVED SIGNAL
# -----------------------------------------

time = np.arange(NUM_SAMPLES) / FS

received_signal = np.zeros(NUM_SAMPLES)

for distance, amplitude in zip(
    TARGET_DISTANCES, TARGET_AMPLITUDES
):
    delay = (2 * distance) / C
    delay_samples = int(round(delay * FS))

    received_signal[
        delay_samples:delay_samples + PULSE_WIDTH
    ] += amplitude

# Add Gaussian noise
rng = np.random.default_rng(RANDOM_SEED)

noise = rng.normal(
    0,
    NOISE_STD,
    NUM_SAMPLES
)

received_signal += noise

# -----------------------------------------
# STEP 3: APPLY MATCHED FILTER
# -----------------------------------------

correlation = correlate(
    received_signal,
    transmitted_pulse,
    mode="full",
    method="fft"
)

lags = correlation_lags(
    len(received_signal),
    len(transmitted_pulse),
    mode="full"
)

# Keep only non-negative delays
valid = lags >= 0

correlation = correlation[valid]
lags = lags[valid]

# -----------------------------------------
# STEP 4: DETECT TARGETS
# -----------------------------------------

THRESHOLD = 3.0

peaks, properties = find_peaks(
    correlation,
    height=THRESHOLD,
    distance=2 * PULSE_WIDTH
)

detected_distances = (
    lags[peaks] / FS
) * C / 2

# -----------------------------------------
# STEP 5: DISPLAY RESULTS
# -----------------------------------------

print("\nRADARIQ - MATCHED FILTERING")
print("--------------------------------")

print(f"Sampling frequency: {FS / 1e6:.2f} MHz")
print(f"Pulse width: {PULSE_WIDTH} samples")
print(f"Noise standard deviation: {NOISE_STD}")
print(f"Detection threshold: {THRESHOLD:.2f}")

print("\nACTUAL TARGETS")

for distance in TARGET_DISTANCES:
    print(f"Target distance: {distance:.2f} m")

print("\nDETECTED TARGETS")

for i, distance in enumerate(detected_distances):
    print(
        f"Detection {i + 1}: "
        f"Distance = {distance:.2f} m"
    )

print(f"\nTotal detections: {len(detected_distances)}")
print(f"Actual targets: {len(TARGET_DISTANCES)}")

print("\nMATCHED FILTERING COMPLETE")

# -----------------------------------------
# STEP 6: VISUALIZE RESULTS
# -----------------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.plot(
    time * 1e6,
    received_signal,
    label="Received noisy signal"
)

plt.xlabel("Time (microseconds)")
plt.ylabel("Amplitude")
plt.title("Radar Received Signal")
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)

range_axis = (lags / FS) * C / 2

plt.plot(
    range_axis,
    correlation,
    label="Matched filter output"
)

plt.scatter(
    detected_distances,
    correlation[peaks],
    color="red",
    label="Detected targets",
    zorder=3
)

plt.axhline(
    THRESHOLD,
    linestyle="--",
    label="Detection threshold"
)

plt.xlabel("Range (metres)")
plt.ylabel("Correlation")
plt.title("Matched Filter Target Detection")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
