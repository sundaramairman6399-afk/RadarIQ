
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# RADARIQ - CFAR TARGET DETECTION
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

# 4. Calculate FFT
fft_result = np.fft.rfft(noisy_signal)

frequencies = np.fft.rfftfreq(
    N,
    d=1 / sampling_rate
)

magnitude = (2 / N) * np.abs(fft_result)

# 5. CFAR configuration
training_cells = 10
guard_cells = 2
Pfa = 0.01

# CA-CFAR threshold multiplier
alpha = training_cells * (
    Pfa ** (-1 / training_cells) - 1
)

threshold = np.zeros(len(magnitude))

detections = []

# 6. Apply CA-CFAR
for i in range(
    training_cells + guard_cells,
    len(magnitude) - training_cells - guard_cells
):

    # Left training cells
    left_cells = magnitude[
        i - guard_cells - training_cells:
        i - guard_cells
    ]

    # Right training cells
    right_cells = magnitude[
        i + guard_cells + 1:
        i + guard_cells + training_cells + 1
    ]

    # Estimate local noise
    noise_level = (
        np.sum(left_cells) + np.sum(right_cells)
    ) / (2 * training_cells)

    # Calculate adaptive threshold
    threshold[i] = alpha * noise_level

    # Detect target
    if magnitude[i] > threshold[i]:

        # Local peak check
        if (
            magnitude[i] > magnitude[i - 1]
            and magnitude[i] > magnitude[i + 1]
        ):
            detections.append(i)

# 7. Display results
print("\nRADARIQ CFAR TARGET DETECTION")
print("--------------------------------")

print(f"Training cells: {training_cells}")
print(f"Guard cells: {guard_cells}")
print(f"False alarm probability: {Pfa}")
print(f"Threshold multiplier: {alpha:.2f}")

print("\nDetected frequencies:")

for index in detections:

    print(
        f"Frequency: {frequencies[index]:.1f} Hz | "
        f"Magnitude: {magnitude[index]:.3f}"
    )

# 8. Visualize FFT and CFAR threshold
plt.figure(figsize=(13, 7))

plt.plot(
    frequencies,
    magnitude,
    label="FFT Magnitude"
)

plt.plot(
    frequencies,
    threshold,
    label="CFAR Threshold"
)

plt.scatter(
    frequencies[detections],
    magnitude[detections],
    marker="x",
    s=80,
    label="Detected Targets"
)

plt.xlim(0, 400)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.title("RadarIQ - CA-CFAR Target Detection")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
