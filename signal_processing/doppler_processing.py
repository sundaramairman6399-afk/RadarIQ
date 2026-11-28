"""Pulse-Doppler processing for the RadarIQ simulation.

This module models a coherent pulsed radar. Each target produces a delayed
copy of the transmitted pulse, while its complex phase rotates from pulse
to pulse according to its Doppler frequency. Matched filtering estimates
range; an FFT across pulses estimates Doppler frequency and radial velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

SPEED_OF_LIGHT = 3.0e8


@dataclass(frozen=True)
class RadarConfig:
    """Radar parameters shared by simulation and processing."""

    sampling_rate_hz: float = 10.0e6
    carrier_frequency_hz: float = 10.0e9
    pulse_repetition_interval_s: float = 1.0e-3
    num_pulses: int = 64
    num_fast_time_samples: int = 1000
    pulse_width_samples: int = 10
    noise_std: float = 0.03
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.carrier_frequency_hz <= 0:
            raise ValueError("carrier_frequency_hz must be positive")
        if self.pulse_repetition_interval_s <= 0:
            raise ValueError("pulse_repetition_interval_s must be positive")
        if self.num_pulses < 2:
            raise ValueError("num_pulses must be at least 2")
        if self.num_fast_time_samples <= self.pulse_width_samples:
            raise ValueError("num_fast_time_samples must exceed pulse_width_samples")
        if self.pulse_width_samples < 1:
            raise ValueError("pulse_width_samples must be at least 1")
        if self.noise_std < 0:
            raise ValueError("noise_std cannot be negative")

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT / self.carrier_frequency_hz

    @property
    def range_bin_spacing_m(self) -> float:
        return SPEED_OF_LIGHT / (2.0 * self.sampling_rate_hz)

    @property
    def max_unambiguous_velocity_mps(self) -> float:
        return self.wavelength_m / (4.0 * self.pulse_repetition_interval_s)

    @property
    def velocity_bin_spacing_mps(self) -> float:
        return self.wavelength_m / (
            2.0 * self.num_pulses * self.pulse_repetition_interval_s
        )


@dataclass(frozen=True)
class Target:
    """A point target with constant range, radial velocity and amplitude."""

    range_m: float
    velocity_mps: float
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        if self.range_m < 0:
            raise ValueError("range_m cannot be negative")
        if self.amplitude < 0:
            raise ValueError("amplitude cannot be negative")


@dataclass(frozen=True)
class RangeDopplerResult:
    """Outputs and axes from the range-Doppler processing chain."""

    received_pulses: np.ndarray
    range_profiles: np.ndarray
    range_doppler_map: np.ndarray
    range_axis_m: np.ndarray
    doppler_frequency_axis_hz: np.ndarray
    velocity_axis_mps: np.ndarray


def simulate_received_pulses(
    config: RadarConfig,
    targets: Iterable[Target],
) -> np.ndarray:
    """Create a coherent pulse-by-fast-time complex baseband data cube.

    The target's round-trip delay sets its fast-time sample. Its Doppler
    frequency f_d = 2*v/lambda creates a pulse-to-pulse phase rotation
    exp(j*2*pi*f_d*p*PRI).
    """
    targets = tuple(targets)
    rng = np.random.default_rng(config.random_seed)

    if config.noise_std:
        noise_scale = config.noise_std / np.sqrt(2.0)
        received = noise_scale * (
            rng.standard_normal(
                (config.num_pulses, config.num_fast_time_samples)
            )
            + 1j * rng.standard_normal(
                (config.num_pulses, config.num_fast_time_samples)
            )
        )
    else:
        received = np.zeros(
            (config.num_pulses, config.num_fast_time_samples),
            dtype=np.complex128,
        )

    transmitted_pulse = np.ones(
        config.pulse_width_samples,
        dtype=np.complex128,
    )
    pulse_times = (
        np.arange(config.num_pulses)
        * config.pulse_repetition_interval_s
    )

    for target in targets:
        round_trip_delay_s = 2.0 * target.range_m / SPEED_OF_LIGHT
        delay_samples = int(round(
            round_trip_delay_s * config.sampling_rate_hz
        ))

        if delay_samples + config.pulse_width_samples > config.num_fast_time_samples:
            raise ValueError(
                f"Target at {target.range_m:g} m falls outside the "
                "simulated fast-time window"
            )

        doppler_hz = 2.0 * target.velocity_mps / config.wavelength_m
        pulse_phase = np.exp(1j * 2.0 * np.pi * doppler_hz * pulse_times)

        received[
            :,
            delay_samples:delay_samples + config.pulse_width_samples,
        ] += (
            target.amplitude
            * pulse_phase[:, np.newaxis]
            * transmitted_pulse[np.newaxis, :]
        )

    return received


def matched_filter_range_profiles(
    received_pulses: np.ndarray,
    pulse_width_samples: int,
) -> np.ndarray:
    """Pulse-compress each row and return non-negative-delay range profiles."""
    received_pulses = np.asarray(received_pulses)

    if received_pulses.ndim != 2:
        raise ValueError("received_pulses must be a 2D array")
    if pulse_width_samples < 1 or pulse_width_samples >= received_pulses.shape[1]:
        raise ValueError("pulse_width_samples is outside the fast-time dimension")

    reference = np.ones(pulse_width_samples, dtype=np.complex128)
    matched_filter = np.conjugate(reference[::-1])

    return np.stack(
        [
            np.convolve(pulse, matched_filter, mode="valid")
            for pulse in received_pulses
        ],
        axis=0,
    )


def process_range_doppler(
    received_pulses: np.ndarray,
    config: RadarConfig,
    *,
    apply_window: bool = True,
) -> RangeDopplerResult:
    """Apply matched filtering followed by a slow-time Doppler FFT."""
    received_pulses = np.asarray(received_pulses)

    expected_shape = (
        config.num_pulses,
        config.num_fast_time_samples,
    )
    if received_pulses.shape != expected_shape:
        raise ValueError(
            f"Expected received_pulses shape {expected_shape}, "
            f"got {received_pulses.shape}"
        )

    range_profiles = matched_filter_range_profiles(
        received_pulses,
        config.pulse_width_samples,
    )

    if apply_window:
        slow_time_window = np.hanning(config.num_pulses)
    else:
        slow_time_window = np.ones(config.num_pulses)

    windowed_profiles = range_profiles * slow_time_window[:, np.newaxis]

    # Axis 0 is slow time (pulse number); axis 1 is fast time (range).
    doppler_spectrum = np.fft.fftshift(
        np.fft.fft(windowed_profiles, axis=0),
        axes=0,
    )

    doppler_frequency_axis = np.fft.fftshift(
        np.fft.fftfreq(
            config.num_pulses,
            d=config.pulse_repetition_interval_s,
        )
    )

    velocity_axis = doppler_frequency_axis * config.wavelength_m / 2.0
    range_axis = (
        np.arange(range_profiles.shape[1])
        * config.range_bin_spacing_m
    )

    return RangeDopplerResult(
        received_pulses=received_pulses,
        range_profiles=range_profiles,
        range_doppler_map=doppler_spectrum,
        range_axis_m=range_axis,
        doppler_frequency_axis_hz=doppler_frequency_axis,
        velocity_axis_mps=velocity_axis,
    )


def plot_range_doppler_map(result: RangeDopplerResult) -> None:
    """Display the range-Doppler magnitude in dB."""
    magnitude = np.abs(result.range_doppler_map)
    magnitude_db = 20.0 * np.log10(
        np.maximum(magnitude, np.finfo(float).tiny)
    )
    magnitude_db -= np.max(magnitude_db)

    plt.figure(figsize=(12, 7))
    plt.pcolormesh(
        result.velocity_axis_mps,
        result.range_axis_m / 1000.0,
        magnitude_db.T,
        shading="auto",
        cmap="viridis",
        vmin=-60,
        vmax=0,
    )
    plt.colorbar(label="Relative magnitude (dB)")
    plt.xlabel("Radial velocity (m/s)")
    plt.ylabel("Range (km)")
    plt.title("RadarIQ - Pulse-Doppler Range-Doppler Map")
    plt.tight_layout()
    plt.show()


def main() -> None:
    config = RadarConfig()
    targets = (
        Target(range_m=1500.0, velocity_mps=-1.875, amplitude=1.0),
        Target(range_m=3000.0, velocity_mps=0.9375, amplitude=0.7),
        Target(range_m=4500.0, velocity_mps=2.8125, amplitude=0.5),
    )

    received_pulses = simulate_received_pulses(config, targets)
    result = process_range_doppler(received_pulses, config)

    print("\nRADARIQ - PULSE-DOPPLER PROCESSING")
    print("----------------------------------")
    print(f"Carrier frequency: {config.carrier_frequency_hz / 1e9:.2f} GHz")
    print(f"Wavelength: {config.wavelength_m:.4f} m")
    print(f"Pulse repetition interval: {config.pulse_repetition_interval_s * 1e3:.2f} ms")
    print(f"Number of coherent pulses: {config.num_pulses}")
    print(f"Range-bin spacing: {config.range_bin_spacing_m:.2f} m")
    print(f"Velocity-bin spacing: {config.velocity_bin_spacing_mps:.4f} m/s")
    print(
        "Maximum unambiguous radial speed: "
        f"+/-{config.max_unambiguous_velocity_mps:.3f} m/s"
    )

    print("\nCONFIGURED TARGETS")
    print("------------------")
    for target in targets:
        print(
            f"Range: {target.range_m:7.1f} m | "
            f"Radial velocity: {target.velocity_mps:+.4f} m/s"
        )

    print("\nRANGE-DOPPLER MAP READY")
    plot_range_doppler_map(result)


if __name__ == "__main__":
    main()
