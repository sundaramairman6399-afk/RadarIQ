
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# ==========================================
# RADARIQ - FILTER CUTOFF COMPARISON
# ==========================================

# 1. Configuration
sampling_rate = 1000
duration = 1

time = np.arange(0, duration, 1 / sampling_rate)
N = len(time)

# 2. Generate radar signal
signal_1 = np.sin(2 * np.pi * 50 * time)
signal_2 = np.sin(2 * np.pi * 120 * time)
signal_3 = 0.05 * np.sin(2 * np.pi * 250 * time)

clean_signal = signal_1 + signal_2 + signal_3

# 3. Add noise
rng = np.random.default_rng(42)
noise = rng.normal(0, 1.0, N)

noisy_signal = clean_signal + noise

# 4. Calculate SNR
def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    return 10 * np.log10(signal_power / noise_power)

input_snr = calculate_snr(clean_signal, noise)

# 5. Define cutoff frequencies
cutoff_frequencies = [150, 300, 400]

results = {}

print("\nRADARIQ FILTER COMPARISON")
print("-------------------------")
print(f"Input SNR: {input_snr:.2f} dB\n")

# 6. Apply different filters
for cutoff in cutoff_frequencies:

    sos = butter(
        4,
        cutoff,
        btype="lowpass",
        fs=sampling_rate,
        output="sos"
    )

    filtered_clean = sosfiltfilt(sos, clean_signal)
    filtered_noise = sosfiltfilt(sos, noise)

    filtered_signal = filtered_clean + filtered_noise

    output_snr = calculate_snr(
        filtered_clean,
        filtered_noise
    )

    # Measure preservation of 250 Hz component
    t = np.arange(N) / sampling_rate

    reference_250 = np.sin(2 * np.pi * 250 * t)

    amplitude_250 = (
        2 / N
    ) * abs(
        np.dot(filtered_clean, reference_250)
    )

    results[cutoff] = {
        "signal": filtered_signal,
        "snr": output_snr,
        "amplitude": amplitude_250
    }

    print(f"Cutoff frequency: {cutoff} Hz")
    print(f"Output SNR: {output_snr:.2f} dB")
    print(f"250 Hz amplitude: {amplitude_250:.4f}")
    print(f"SNR improvement: {output_snr - input_snr:.2f} dB")
    print("-------------------------")

# 7. Visualization
fig, axes = plt.subplots(3, 1, figsize=(13, 11))

# Original noisy signal
axes[0].plot(time, noisy_signal)
axes[0].set_title("Original Noisy Radar Signal")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Amplitude")
axes[0].grid()

# Compare filtered signals
for cutoff in cutoff_frequencies:

    axes[1].plot(
        time,
        results[cutoff]["signal"],
        label=f"{cutoff} Hz cutoff"
    )

axes[1].set_title("Filtered Signal Comparison")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Amplitude")
axes[1].legend()
axes[1].grid()

# Compare SNR
snr_values = [
    results[c]["snr"] for c in cutoff_frequencies
]

axes[2].bar(
    [str(c) for c in cutoff_frequencies],
    snr_values
)

axes[2].set_title("Output SNR Comparison")
axes[2].set_xlabel("Cutoff Frequency (Hz)")
axes[2].set_ylabel("SNR (dB)")
axes[2].grid(axis="y")

plt.tight_layout()
plt.show()
