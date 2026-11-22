
import numpy as np

# ==========================================
# RADARIQ - COHERENT INTEGRATION
# ==========================================

FS = 1000
DURATION = 1
SAMPLES = FS * DURATION

TARGET_FREQUENCY = 250
TARGET_AMPLITUDE = 0.05

NOISE_STD = 1.0
NUM_PULSES = 10
NUM_TRIALS = 500

rng = np.random.default_rng(42)

time = np.arange(SAMPLES) / FS

# ==========================================
# GENERATE CLEAN TARGET SIGNAL
# ==========================================

target_signal = TARGET_AMPLITUDE * np.sin(
    2 * np.pi * TARGET_FREQUENCY * time
)

# FFT of clean target
target_fft = np.fft.rfft(target_signal)

# Target frequency bin
target_bin = TARGET_FREQUENCY

signal_component = target_fft[target_bin]

# ==========================================
# COLLECT MULTIPLE NOISY OBSERVATIONS
# ==========================================

noise_fft_all = np.zeros(
    (NUM_TRIALS, NUM_PULSES),
    dtype=complex
)

for trial in range(NUM_TRIALS):

    for pulse in range(NUM_PULSES):

        noise = rng.normal(0, NOISE_STD, SAMPLES)

        received_signal = target_signal + noise

        received_fft = np.fft.rfft(received_signal)

        noise_component = (
            received_fft[target_bin] - signal_component
        )

        noise_fft_all[trial, pulse] = noise_component


# ==========================================
# SINGLE-PULSE SNR
# ==========================================

single_noise_power = np.mean(
    np.abs(noise_fft_all) ** 2
)

single_signal_power = np.abs(signal_component) ** 2

single_snr = single_signal_power / single_noise_power

single_snr_db = 10 * np.log10(single_snr)


# ==========================================
# COHERENT INTEGRATION
# ==========================================

# Add complex FFT values across pulses.
# Signal phase is aligned across all pulses.

integrated_signal = NUM_PULSES * signal_component

integrated_noise = np.sum(
    noise_fft_all,
    axis=1
)

integrated_signal_power = np.abs(integrated_signal) ** 2

integrated_noise_power = np.mean(
    np.abs(integrated_noise) ** 2
)

integrated_snr = (
    integrated_signal_power / integrated_noise_power
)

integrated_snr_db = 10 * np.log10(integrated_snr)


# ==========================================
# THEORETICAL SNR IMPROVEMENT
# ==========================================

theoretical_gain = 10 * np.log10(NUM_PULSES)

measured_gain = integrated_snr_db - single_snr_db


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\nRADARIQ - COHERENT INTEGRATION")
print("--------------------------------")

print("Target frequency:", TARGET_FREQUENCY, "Hz")
print("Target amplitude:", TARGET_AMPLITUDE)
print("Noise standard deviation:", NOISE_STD)

print("\nIntegration settings")
print("--------------------")
print("Number of pulses:", NUM_PULSES)
print("Number of trials:", NUM_TRIALS)

print("\nSNR RESULTS")
print("-----------")

print(f"Single-pulse SNR: {single_snr_db:.2f} dB")

print(
    f"Integrated SNR: {integrated_snr_db:.2f} dB"
)

print(f"Theoretical SNR gain: {theoretical_gain:.2f} dB")

print(f"Measured SNR improvement: {measured_gain:.2f} dB")

print("\nCOHERENT INTEGRATION COMPLETE")
