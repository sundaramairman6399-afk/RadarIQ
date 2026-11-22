
import numpy as np
import matplotlib.pyplot as plt

# ==================================
# 1. SIGNAL CONFIGURATION
# ==================================

sampling_rate = 1000
duration = 1
frequency = 100
amplitude = 1

# ==================================
# 2. GENERATE TIME SAMPLES
# ==================================

time = np.arange(
    0,
    duration,
    1 / sampling_rate
)

number_of_samples = len(time)

# ==================================
# 3. GENERATE CLEAN SIGNAL
# ==================================

clean_signal = amplitude * np.sin(
    2 * np.pi * frequency * time
)

# ==================================
# 4. ADD RANDOM NOISE
# ==================================

noise = np.random.normal(
    0,
    0.3,
    number_of_samples
)

noisy_signal = clean_signal + noise

# ==================================
# 5. APPLY FFT
# ==================================

fft_result = np.fft.rfft(noisy_signal)

# Calculate corresponding frequencies
frequency_values = np.fft.rfftfreq(
    number_of_samples,
    d=1 / sampling_rate
)

# Calculate magnitude
magnitude = (
    2 / number_of_samples
) * np.abs(fft_result)

# ==================================
# 6. DETECT DOMINANT FREQUENCY
# ==================================

# Ignore the zero-frequency (DC) component
peak_index = np.argmax(magnitude[1:]) + 1

detected_frequency = frequency_values[peak_index]

print("--------------------------------")
print("       RADARIQ FFT ANALYZER")
print("--------------------------------")

print("Sampling Rate:", sampling_rate, "Hz")
print("Total Samples:", number_of_samples)
print("Actual Frequency:", frequency, "Hz")
print("Detected Frequency:", detected_frequency, "Hz")

# ==================================
# 7. VISUALIZE RESULTS
# ==================================

plt.figure(figsize=(12, 8))

# Time domain
plt.subplot(2, 1, 1)

plt.plot(time, noisy_signal, label="Noisy Signal")

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Time Domain Signal")
plt.grid()
plt.legend()

# Frequency domain
plt.subplot(2, 1, 2)

plt.plot(frequency_values, magnitude)

plt.axvline(
    detected_frequency,
    color="red",
    linestyle="--",
    label=f"Detected: {detected_frequency:.1f} Hz"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Frequency Spectrum")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
