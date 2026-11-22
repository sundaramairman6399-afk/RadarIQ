
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuration
sampling_rate = 1000
duration = 1

# 2. Generate time samples
time = np.arange(
    0,
    duration,
    1 / sampling_rate
)

N = len(time)

# 3. Generate multiple frequencies
signal_1 = np.sin(2 * np.pi * 50 * time)

signal_2 = np.sin(2 * np.pi * 120 * time)

signal_3 = 0.5 * np.sin(2 * np.pi * 250 * time)

# 4. Combine signals
combined_signal = signal_1 + signal_2 + signal_3

# 5. Add noise
noise = np.random.normal(0, 0.2, N)

noisy_signal = combined_signal + noise

# 6. Apply FFT
fft_result = np.fft.rfft(noisy_signal)

frequencies = np.fft.rfftfreq(
    N,
    d=1 / sampling_rate
)

magnitude = (2 / N) * np.abs(fft_result)

# 7. Detect significant peaks
threshold = 0.3

peak_indices = []

for i in range(1, len(magnitude) - 1):

    if (
        magnitude[i] > magnitude[i - 1]
        and magnitude[i] > magnitude[i + 1]
        and magnitude[i] > threshold
    ):
        peak_indices.append(i)

# 8. Display results
print("\nRADARIQ MULTI-FREQUENCY ANALYZER")
print("--------------------------------")

for index in peak_indices:
    print(
        f"Detected frequency: {frequencies[index]:.1f} Hz | "
        f"Magnitude: {magnitude[index]:.2f}"
    )

# 9. Visualize
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time, noisy_signal)
plt.title("Combined Noisy Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(frequencies, magnitude)
plt.axhline(
    threshold,
    color="red",
    linestyle="--",
    label="Detection threshold"
)

plt.title("Multi-Frequency FFT Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
