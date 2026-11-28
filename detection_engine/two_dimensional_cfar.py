
"""
RadarIQ - Two-Dimensional CA-CFAR Detector

Applies Cell Averaging CFAR to a 2D Range-Doppler power map.

Rows    = Doppler bins
Columns = Range bins
"""

import numpy as np


def ca_cfar_2d(
    power_map: np.ndarray,
    guard_cells: tuple = (2, 2),
    training_cells: tuple = (4, 4),
    pfa: float = 0.01
):
    """
    Apply 2D Cell Averaging CFAR.

    Parameters
    ----------
    power_map : np.ndarray
        Two-dimensional radar power matrix.

    guard_cells : tuple
        (doppler_guard, range_guard)

    training_cells : tuple
        (doppler_training, range_training)

    pfa : float
        Desired probability of false alarm.

    Returns
    -------
    detections : np.ndarray
        Boolean detection matrix.

    threshold_map : np.ndarray
        Detection threshold at each valid CUT.

    noise_map : np.ndarray
        Estimated local noise power.
    """

    # STEP 1: VALIDATE INPUT

    power_map = np.asarray(power_map)

    if power_map.ndim != 2:
        raise ValueError("Power map must be a 2D array.")

    if np.iscomplexobj(power_map):
        raise ValueError("Provide a real-valued power map.")

    if not np.all(np.isfinite(power_map)):
        raise ValueError("Power map must contain only finite values.")

    if np.any(power_map < 0):
        raise ValueError("Power values cannot be negative.")

    if not 0 < pfa < 1:
        raise ValueError("Pfa must be between 0 and 1.")

    if len(guard_cells) != 2 or len(training_cells) != 2:
        raise ValueError("Guard and training cells must contain two values.")

    guard_doppler, guard_range = guard_cells
    train_doppler, train_range = training_cells

    for value in guard_cells:
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("Guard cells must be non-negative integers.")

    for value in training_cells:
        if not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError("Training cells must be positive integers.")

    rows, cols = power_map.shape

    # STEP 2: CALCULATE OUTER WINDOW SIZE

    half_doppler = guard_doppler + train_doppler
    half_range = guard_range + train_range

    outer_rows = 2 * half_doppler + 1
    outer_cols = 2 * half_range + 1

    # STEP 3: CREATE TRAINING CELL MASK

    # Initially, every cell is considered a training cell.
    training_mask = np.ones(
        (outer_rows, outer_cols),
        dtype=bool
    )

    # Find the centre of the window.
    center_row = half_doppler
    center_col = half_range

    # Exclude CUT and guard cells from noise estimation.
    training_mask[
        center_row - guard_doppler:center_row + guard_doppler + 1,
        center_col - guard_range:center_col + guard_range + 1
    ] = False

    # Count the remaining training cells.
    number_of_training_cells = np.count_nonzero(training_mask)

    # STEP 4: CALCULATE CA-CFAR MULTIPLIER

    # Theoretical multiplier for independent exponential
    # noise-power samples.
    alpha = number_of_training_cells * (
        pfa ** (-1 / number_of_training_cells) - 1
    )

    # STEP 5: INITIALIZE OUTPUT MATRICES

    detections = np.zeros((rows, cols), dtype=bool)

    # Boundary cells remain unprocessed.
    threshold_map = np.full((rows, cols), np.inf)

    noise_map = np.full((rows, cols), np.nan)

    # STEP 6: SLIDE CFAR WINDOW OVER THE POWER MAP

    for row in range(half_doppler, rows - half_doppler):

        for col in range(half_range, cols - half_range):

            # Extract the complete outer window around the CUT.
            window = power_map[
                row - half_doppler:row + half_doppler + 1,
                col - half_range:col + half_range + 1
            ]

            # Select only training cells.
            training_data = window[training_mask]

            # Estimate local noise power.
            noise_estimate = np.mean(training_data)

            # Calculate adaptive threshold.
            threshold = alpha * noise_estimate

            # Store estimated noise and threshold.
            noise_map[row, col] = noise_estimate
            threshold_map[row, col] = threshold

            # Compare CUT power against the threshold.
            if power_map[row, col] > threshold:
                detections[row, col] = True

    return detections, threshold_map, noise_map
