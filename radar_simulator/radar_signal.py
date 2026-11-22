
import numpy as np
import matplotlib.pyplot as plt

# 1. Signal configuration
sampling_rate = 1000
duration = 1
frequency = 100
amplitude = 1

# 2. Generate time values
time = np.arange(
    0,
    duration,
    1 / sampling_rate
)

# 3. Generate clean signal
signal = amplitude * np.sin(
    2 * np.pi * frequency * time
)

# 4. Generate random noise
noise = np.random.normal(
    0,
    0.3,
    len(time)
)

# 5. Add noise
noisy_signal = signal + noise

# 6. Plot signals
plt.figure(figsize=(12, 5))

plt.plot(time, signal, label="Clean Signal")
plt.plot(time, noisy_signal, label="Noisy Signal", alpha=0.6)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("RadarIQ - Signal Generation")

plt.legend()
plt.grid()

plt.show()
