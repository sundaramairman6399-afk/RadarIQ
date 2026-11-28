
"""
RadarIQ - Target Extraction and Clustering

Groups neighbouring 2D CFAR detections into target clusters.

For each cluster, the strongest power cell is selected
as the representative target location.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class DetectedTarget:
    """Represents one extracted radar target."""

    range_m: float
    velocity_mps: float
    peak_power: float
    num_detection_cells: int
    doppler_bin: int
    range_bin: int


def extract_targets(
    detections: np.ndarray,
    power_map: np.ndarray,
    range_axis_m: np.ndarray,
    velocity_axis_mps: np.ndarray,
    min_cluster_size: int = 1,
) -> list[DetectedTarget]:
    """
    Group neighbouring detections and extract target estimates.

    Parameters
    ----------
    detections:
        Boolean 2D CFAR detection matrix.

    power_map:
        2D Range-Doppler power matrix.

    range_axis_m:
        Range values corresponding to matrix columns.

    velocity_axis_mps:
        Velocity values corresponding to matrix rows.

    min_cluster_size:
        Minimum number of connected detection cells required.

    Returns
    -------
    list[DetectedTarget]:
        Extracted target estimates.
    """

    detections = np.asarray(detections, dtype=bool)
    power_map = np.asarray(power_map)

    # STEP 1: VALIDATE INPUTS

    if detections.ndim != 2 or power_map.ndim != 2:
        raise ValueError("Detection and power maps must be 2D.")

    if detections.shape != power_map.shape:
        raise ValueError("Detection and power maps must have equal shapes.")

    if not np.all(np.isfinite(power_map)):
        raise ValueError("Power map must contain finite values.")

    if np.iscomplexobj(power_map) or np.any(power_map < 0):
        raise ValueError("Power map must contain non-negative real values.")

    if len(range_axis_m) != power_map.shape[1]:
        raise ValueError("Range axis length must match power-map columns.")

    if len(velocity_axis_mps) != power_map.shape[0]:
        raise ValueError("Velocity axis length must match power-map rows.")

    if not isinstance(min_cluster_size, (int, np.integer)):
        raise ValueError("min_cluster_size must be an integer.")

    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be at least 1.")

    # STEP 2: DEFINE NEIGHBOUR CONNECTIVITY

    # Eight-connected neighbourhood:
    # Horizontal, vertical and diagonal neighbours
    # belong to the same cluster.

    structure = np.ones((3, 3), dtype=int)

    # STEP 3: LABEL CONNECTED DETECTION CELLS

    labels, number_of_clusters = ndimage.label(
        detections,
        structure=structure
    )

    extracted_targets = []

    # STEP 4: PROCESS EACH CLUSTER

    for cluster_id in range(1, number_of_clusters + 1):

        # Find all cells belonging to this cluster.
        cluster_cells = np.argwhere(labels == cluster_id)

        # Ignore clusters smaller than the requested size.
        if len(cluster_cells) < min_cluster_size:
            continue

        # STEP 5: FIND THE STRONGEST CELL

        cluster_powers = power_map[
            cluster_cells[:, 0],
            cluster_cells[:, 1]
        ]

        strongest_index = np.argmax(cluster_powers)

        doppler_bin, range_bin = cluster_cells[strongest_index]

        # STEP 6: CONVERT BIN INDICES INTO PHYSICAL VALUES

        target_range = float(range_axis_m[range_bin])

        target_velocity = float(velocity_axis_mps[doppler_bin])

        peak_power = float(power_map[doppler_bin, range_bin])

        # STEP 7: STORE EXTRACTED TARGET

        extracted_targets.append(
            DetectedTarget(
                range_m=target_range,
                velocity_mps=target_velocity,
                peak_power=peak_power,
                num_detection_cells=len(cluster_cells),
                doppler_bin=int(doppler_bin),
                range_bin=int(range_bin),
            )
        )

    # Sort targets by range for easier interpretation.
    extracted_targets.sort(key=lambda target: target.range_m)

    return extracted_targets
