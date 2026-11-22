
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# RADARIQ - CA-CFAR vs OS-CFAR
# Corrected implementation
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

# 4. Calculate FFT power
fft_result = np.fft.rfft(noisy_signal)

frequencies = np.fft.rfftfreq(
    N,
    d=1 / sampling_rate
)

power = np.abs(fft_result) ** 2

# 5. CFAR configuration
training_cells = 10
guard_cells = 2

total_training = 2 * training_cells

Pfa = 0.01

# OS-CFAR rank (1-based, ascending order)
rank = 15

# 6. CA-CFAR multiplier
alpha_ca = total_training * (
    Pfa ** (-1 / total_training) - 1
)

# 7. Correct OS-CFAR multiplier
def calculate_os_alpha(n, k, pfa):

    low = 0.0
    high = 100.0

    for _ in range(100):

        alpha = (low + high) / 2

        probability = 1.0

        for j in range(k):
            probability *= (
                (n - j) / (n - j + alpha)
            )

        if probability > pfa:
            low = alpha
        else:
            high = alpha

    return (low + high) / 2


alpha_os = calculate_os_alpha(
    total_training,
    rank,
    Pfa
)

# 8. Generic CFAR detector
def run_cfar(method):

    threshold = np.zeros(len(power))
    detections = []

    start = training_cells + guard_cells
    end = len(power) - start

    for i in range(start, end):

        # Left training cells
        left = power[
            i - guard_cells - training_cells:
            i - guard_cells
        ]

        # Right training cells
        right = power[
            i + guard_cells + 1:
            i + guard_cells + training_cells + 1
        ]

        # Combine training cells
        training = np.concatenate((left, right))

        # Estimate noise
        if method == "CA":

            noise_level = np.mean(training)
            threshold[i] = alpha_ca * noise_level

        elif method == "OS":

            sorted_training = np.sort(training)

            noise_level = sorted_training[rank - 1]

            threshold[i] = alpha_os * noise_level

        # Target detection
        if (
            power[i] > threshold[i]
            and power[i] > power[i - 1]
            and power[i] > power[i + 1]
        ):

            detections.append(i)

    return threshold, detections


# 9. Run both detectors
ca_threshold, ca_detections = run_cfar("CA")
os_threshold, os_detections = run_cfar("OS")

# 10. Display results
print("\nRADARIQ - CA-CFAR vs OS-CFAR")
print("--------------------------------")

print(f"Training cells per side: {training_cells}")
print(f"Guard cells per side: {guard_cells}")
print(f"Total training cells: {total_training}")
print(f"False alarm probability: {Pfa}")
print(f"OS rank: {rank}")

print(f"\nCA multiplier: {alpha_ca:.3f}")
print(f"OS multiplier: {alpha_os:.3f}")

print("\nCA-CFAR detections:")

for index in ca_detections:

    print(
        f"{frequencies[index]:.1f} Hz | "
        f"Power: {power[index]:.2f}"
    )

print("\nOS-CFAR detections:")

for index in os_detections:

    print(
        f"{frequencies[index]:.1f} Hz | "
        f"Power: {power[index]:.2f}"
    )

# 11. Plot comparison
plt.figure(figsize=(14, 8))

plt.plot(
    frequencies,
    power,
    label="FFT Power",
    alpha=0.7
)

plt.plot(
    frequencies,
    ca_threshold,
    label="CA-CFAR Threshold"
)

plt.plot(
    frequencies,
    os_threshold,
    label="OS-CFAR Threshold"
)

plt.scatter(
    frequencies[ca_detections],
    power[ca_detections],
    marker="x",
    s=70,
    label="CA detections"
)

plt.scatter(
    frequencies[os_detections],
    power[os_detections],
    marker="o",
    s=45,
    facecolors="none",
    label="OS detections"
)

plt.xlim(0, 400)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")

plt.title("RadarIQ - CA-CFAR vs OS-CFAR")

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
