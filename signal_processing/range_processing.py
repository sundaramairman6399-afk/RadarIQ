
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# RADARIQ - MULTIPLE TARGET RANGE DETECTION
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
# TRANSMITTED RADAR SIGNAL
# ==========================================

transmitted_signal = np.zeros(NUM_SAMPLES)

transmitted_signal[:PULSE_WIDTH] = 1.0

# ==========================================
# RECEIVED ECHO SIGNAL
# ==========================================

received_signal = np.zeros(NUM_SAMPLES)

for distance, amplitude in zip(
    TARGET_DISTANCES,
    TARGET_AMPLITUDES
):

    # Calculate round-trip time delay
    time_delay = (2 * distance) / C

    # Convert time delay into sample delay
    delay_samples = round(time_delay * FS)

    # Generate reflected echo
    if delay_samples + PULSE_WIDTH <= NUM_SAMPLES:

        received_signal[
            delay_samples:delay_samples + PULSE_WIDTH
        ] += amplitude

# ==========================================
# TARGET DETECTION - LEADING EDGE
# ==========================================

threshold = 0.1

peaks = np.where(
    (received_signal[1:] >= threshold) &
    (received_signal[:-1] < threshold)
)[0] + 1

# ==========================================
# RANGE CALCULATION
# ==========================================

print("\nRADARIQ - MULTIPLE TARGET RANGE DETECTION")
print("------------------------------------------")

print(f"Sampling frequency: {FS / 1e6:.2f} MHz")
print(f"Number of targets configured: {len(TARGET_DISTANCES)}")

print("\nDETECTED TARGETS")
print("----------------")

for index, peak in enumerate(peaks):

    # Convert sample index to time delay
    measured_delay = peak / FS

    # Calculate target distance
    calculated_distance = (
        C * measured_delay
    ) / 2

    # Read echo amplitude at leading edge
    echo_amplitude = received_signal[peak]

    print(
        f"Target {index + 1}: "
        f"Distance = {calculated_distance:.2f} m | "
        f"Echo amplitude = {echo_amplitude:.2f}"
    )

# ==========================================
# PLOT RADAR SIGNALS
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    time * 1e6,
    transmitted_signal,
    label="Transmitted Pulse",
    linewidth=2
)

plt.plot(
    time * 1e6,
    received_signal,
    label="Received Echoes",
    linewidth=2
)

# Mark detected echo positions
for peak in peaks:

    plt.axvline(
        x=time[peak] * 1e6,
        linestyle="--",
        alpha=0.5
    )

plt.xlabel("Time (microseconds)")
plt.ylabel("Amplitude")

plt.title("RadarIQ - Multiple Target Range Detection")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
