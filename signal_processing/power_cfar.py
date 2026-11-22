
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# RADARIQ - POWER BASED CA-CFAR
# ==========================================

# 1. Radar configuration
sampling_rate = 1000
duration = 1

time = np.arange(0, duration, 1 / sampling_rate)
N = len(time)

# 2. Generate target signals
signal_1 = np.sin(2 * np.pi * 50 * time)
signal_2 = np.sin(2 * np.pi * 120 * time)
signal_3 = 0.05 * np.sin(2 * np.pi * 250 * time)

clean_signal = signal_1 + signal_2 + signal_3

# 3. Add Gaussian noise
rng = np.random.default_rng(42)
noise = rng.normal(0, 1.0, N)

noisy_signal = clean_signal + noise

# 4. FFT and power spectrum
fft_result = np.fft.rfft(noisy_signal)

frequencies = np.fft.rfftfreq(
    N,
    d=1 / sampling_rate
)

magnitude = np.abs(fft_result)
power = magnitude ** 2

# 5. CA-CFAR parameters
training_cells = 10
guard_cells = 2
Pfa = 0.01

# Square-law CA-CFAR multiplier
alpha = training_cells * (
    Pfa ** (-1 / training_cells) - 1
)

threshold = np.zeros(len(power))
detections = []

# 6. Apply power-based CA-CFAR
for i in range(
    training_cells + guard_cells,
    len(power) - training_cells - guard_cells
):

    left_cells = power[
        i - guard_cells - training_cells:
        i - guard_cells
    ]

    right_cells = power[
        i + guard_cells + 1:
        i + guard_cells + training_cells + 1
    ]

    noise_power = (
        np.sum(left_cells) + np.sum(right_cells)
    ) / (2 * training_cells)

    threshold[i] = alpha * noise_power

    if power[i] > threshold[i]:
        if (
            power[i] > power[i - 1]
            and power[i] > power[i + 1]
        ):
            detections.append(i)

# 7. Print results
print("\nRADARIQ POWER BASED CA-CFAR")
print("--------------------------------")

print(f"Training cells: {training_cells}")
print(f"Guard cells: {guard_cells}")
print(f"Pfa: {Pfa}")
print(f"Alpha: {alpha:.3f}")

print("\nDetected targets:")

for index in detections:
    print(
        f"Frequency: {frequencies[index]:.1f} Hz | "
        f"Power: {power[index]:.3f}"
    )

# 8. Plot spectrum and threshold
plt.figure(figsize=(13, 7))

plt.plot(
    frequencies,
    power,
    label="FFT Power"
)

plt.plot(
    frequencies,
    threshold,
    label="CFAR Threshold"
)

plt.scatter(
    frequencies[detections],
    power[detections],
    marker="x",
    s=80,
    label="Detected Targets"
)

plt.xlim(0, 400)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")

plt.title("RadarIQ - Power Based CA-CFAR")

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
