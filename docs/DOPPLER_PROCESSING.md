# Pulse-Doppler Processing

## Objective

This milestone adds a coherent pulse-Doppler processing chain to RadarIQ. It
simulates multiple complex-baseband pulse echoes, estimates range with a
matched filter, and estimates radial velocity with an FFT across pulses.

## Signal model

For pulse index m, a target echo at fast-time delay tau has a pulse-to-pulse
phase progression:

    x_m(t) = A s(t-tau) exp(j 2 pi f_D m T_r) + n_m(t)

where A is target amplitude, s(t) is the transmitted pulse, T_r is the pulse
repetition interval (PRI), f_D is Doppler frequency, and n_m(t) is complex
additive white Gaussian noise.

For a monostatic radar:

    f_D = 2 v_r / lambda
    v_r = lambda f_D / 2
    lambda = c / f_c

The sign convention in this simulator is that positive radial velocity
produces positive Doppler frequency.

## Processing chain

1. Pulse collection: rows are coherent pulses and columns are fast-time samples.
2. Range compression: a matched filter is applied independently to each pulse.
3. Slow-time window: a Hann window reduces Doppler sidelobes. Disable it when
   checking exact FFT-bin behaviour in tests.
4. Doppler FFT: an FFT along the pulse dimension estimates Doppler frequency
   for every range bin.
5. Axis conversion: Doppler frequency is converted to radial velocity.

## Resolution and ambiguity

For the default configuration:

- Carrier frequency: 10 GHz.
- Wavelength: 0.03 m.
- Sampling rate: 10 MHz.
- PRI: 1 ms.
- Number of pulses: 64.
- Range-bin spacing: c/(2 fs) = 15 m.
- Doppler-bin spacing: 1/(N PRI) = 15.625 Hz.
- Velocity-bin spacing: lambda/(2 N PRI) = 0.234375 m/s.
- Maximum unambiguous radial speed: lambda/(4 PRI) = 7.5 m/s.

The unambiguous range for this simple pulsed model is limited by the PRI:
R_unamb = c PRI / 2 = 150 km. The simulated fast-time record is shorter,
so the actual simulated range window is more restrictive.

## Run

From the repository root:

    python -m signal_processing.doppler_processing

Run the unit tests:

    python -m unittest discover -s tests -v

## Scope and assumptions

This is a simplified coherent pulsed-radar baseband model, not an FMCW
chirp simulator. Targets have constant range and radial velocity over the
coherent processing interval. The model does not yet include acceleration,
range migration, multipath, antenna patterns, clutter, transmitter phase
noise, or Doppler ambiguity resolution. Those belong in later milestones.
