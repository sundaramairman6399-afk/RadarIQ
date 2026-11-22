
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# RADARIQ - NOISE AND WEAK SIGNAL ANALYSIS
# ==========================================

# 1. Configuration
sampling_rate = 1000
duration = 1

time = np.arange(
    0,
    duration,
    1 / sampling_rate
)

N = len(time)

# 2. Generate radar signal components

signal_1 = np.sin(2 * np.pi * 50 * time)

signal_2 = np.sin(2 * np.pi * 120 * time)

# Weak signal: amplitude reduced from 0.5 to 0.05
signal_3 = 0.05 * np.sin(2 * np.pi * 250 * time)

# Combine all signals
clean_signal = signal_1 + signal_2 + signal_3

# 3. Define noise levels
noise_levels = {
    "Low Noise": 0.1,
    "Medium Noise": 0.5,
    "High Noise": 1.0
}

# Reproducible random noise
rng = np.random.default_rng(42)

# Detection threshold
threshold = 0.3

# Actual signal frequencies
actual_frequencies = [50, 120, 250]

# 4. Create visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

print("\nRADARIQ WEAK SIGNAL DETECTION ANALYSIS")
print("--------------------------------------")
print("Actual frequencies: 50, 120, 250 Hz")
print("250 Hz signal amplitude: 0.05")
print(f"Detection threshold: {threshold}")
print("--------------------------------------\n")

# 5. Run experiments
for row, (label, noise_std) in enumerate(noise_levels.items()):

    # Generate random noise
    noise = rng.normal(0, noise_std, N)

    # Add noise to clean signal
    noisy_signal = clean_signal + noise

    # 6. Apply FFT
    fft_result = np.fft.rfft(noisy_signal)

    frequencies = np.fft.rfftfreq(
        N,
        d=1 / sampling_rate
    )

    magnitude = (2 / N) * np.abs(fft_result)

    # 7. Detect significant peaks
    peak_indices = []

    for i in range(1, len(magnitude) - 1):

        if (
            magnitude[i] > magnitude[i - 1]
            and magnitude[i] > magnitude[i + 1]
            and magnitude[i] > threshold
        ):
            peak_indices.append(i)

    detected_frequencies = [
        frequencies[i] for i in peak_indices
    ]

    # 8. Check target detection
    correctly_detected = 0
    missed_targets = []

    for actual in actual_frequencies:

        found = any(
            abs(detected - actual) <= 2
            for detected in detected_frequencies
        )

        if found:
            correctly_detected += 1
        else:
            missed_targets.append(actual)

    # 9. Identify false detections
    false_detections = [
        detected
        for detected in detected_frequencies
        if not any(
            abs(detected - actual) <= 2
            for actual in actual_frequencies
        )
    ]

    # 10. Display results
    print(f"\n{label}")
    print(f"Noise standard deviation: {noise_std}")

    print(
        "Detected frequencies:",
        [round(float(f), 1) for f in detected_frequencies]
    )

    print(
        f"Targets detected: {correctly_detected}/3"
    )

    print(
        "Missed targets:",
        missed_targets if missed_targets else "None"
    )

    print(
        "False detections:",
        [round(float(f), 1) for f in false_detections]
    )

    # 11. Time-domain visualization
    axes[row, 0].plot(time, noisy_signal)

    axes[row, 0].set_title(
        f"{label} - Time Domain"
    )

    axes[row, 0].set_xlabel("Time (seconds)")
    axes[row, 0].set_ylabel("Amplitude")
    axes[row, 0].grid()

    # 12. Frequency-domain visualization
    axes[row, 1].plot(frequencies, magnitude)

    axes[row, 1].axhline(
        threshold,
        color="red",
        linestyle="--",
        label="Detection Threshold"
    )

    axes[row, 1].set_xlim(0, 400)

    axes[row, 1].set_title(
        f"{label} - FFT Spectrum"
    )

    axes[row, 1].set_xlabel("Frequency (Hz)")
    axes[row, 1].set_ylabel("Magnitude")
    axes[row, 1].legend()
    axes[row, 1].grid()

# 13. Final visualization
plt.suptitle(
    "RadarIQ - Weak Signal Detection Under Noise",
    fontsize=16
)

plt.tight_layout()
plt.show()
