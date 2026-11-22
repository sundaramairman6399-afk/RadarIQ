
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt

# ==========================================
# RADARIQ - BUTTERWORTH FILTER ANALYSIS
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
signal_3 = 0.05 * np.sin(2 * np.pi * 250 * time)

clean_signal = signal_1 + signal_2 + signal_3

# 3. Add noise
rng = np.random.default_rng(42)

noise_std = 1.0
noise = rng.normal(0, noise_std, N)

noisy_signal = clean_signal + noise

# 4. Design Butterworth low-pass filter
cutoff_frequency = 300
filter_order = 4

sos = butter(
    filter_order,
    cutoff_frequency,
    btype="lowpass",
    fs=sampling_rate,
    output="sos"
)

# 5. Apply filter
filtered_signal = sosfiltfilt(sos, noisy_signal)

# Filter clean signal and noise separately for SNR comparison
filtered_clean = sosfiltfilt(sos, clean_signal)
filtered_noise = sosfiltfilt(sos, noise)

# 6. Calculate SNR
def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    return 10 * np.log10(signal_power / noise_power)

input_snr = calculate_snr(clean_signal, noise)

output_snr = calculate_snr(
    filtered_clean,
    filtered_noise
)

# 7. Calculate FFT
frequencies = np.fft.rfftfreq(
    N,
    d=1 / sampling_rate
)

noisy_fft = np.fft.rfft(noisy_signal)
filtered_fft = np.fft.rfft(filtered_signal)

noisy_magnitude = (2 / N) * np.abs(noisy_fft)
filtered_magnitude = (2 / N) * np.abs(filtered_fft)

# 8. Display results
print("\nRADARIQ BUTTERWORTH FILTER ANALYSIS")
print("----------------------------------")

print(f"Sampling rate: {sampling_rate} Hz")
print(f"Filter cutoff: {cutoff_frequency} Hz")
print(f"Filter order: {filter_order}")

print(f"\nInput SNR: {input_snr:.2f} dB")
print(f"Output SNR: {output_snr:.2f} dB")

print(
    f"SNR improvement: {output_snr - input_snr:.2f} dB"
)

print("\nExpected frequencies: 50, 120, 250 Hz")

# 9. Visualize results
fig, axes = plt.subplots(3, 1, figsize=(13, 12))

# Original noisy signal
axes[0].plot(time, noisy_signal)
axes[0].set_title("Original Noisy Radar Signal")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Amplitude")
axes[0].grid()

# Filtered signal
axes[1].plot(time, filtered_signal)
axes[1].set_title("Filtered Radar Signal")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Amplitude")
axes[1].grid()

# FFT comparison
axes[2].plot(
    frequencies,
    noisy_magnitude,
    label="Before filtering",
    alpha=0.7
)

axes[2].plot(
    frequencies,
    filtered_magnitude,
    label="After filtering",
    alpha=0.8
)

axes[2].axvline(
    cutoff_frequency,
    color="red",
    linestyle="--",
    label="300 Hz cutoff"
)

axes[2].set_xlim(0, 500)
axes[2].set_title("FFT Comparison: Before vs After Filtering")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Magnitude")
axes[2].legend()
axes[2].grid()

plt.tight_layout()
plt.show()
