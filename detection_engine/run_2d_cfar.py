
"""
RadarIQ - Integrated 2D CFAR Target Detection

Complete processing pipeline:
1. Configure radar parameters.
2. Simulate received radar pulses.
3. Generate the Range-Doppler map.
4. Convert the map into power.
5. Apply 2D CA-CFAR detection.
6. Extract connected detection clusters.
7. Rank candidates by peak power.
8. Compare detected candidates with configured targets.
"""

import numpy as np

from signal_processing.doppler_processing import (
    RadarConfig,
    Target,
    simulate_received_pulses,
    process_range_doppler,
)

from detection_engine.two_dimensional_cfar import ca_cfar_2d

from detection_engine.target_extraction import extract_targets


def main():

    # ==========================================================
    # STEP 1: CONFIGURE RADAR
    # ==========================================================

    config = RadarConfig()

    # Define three simulated targets.
    targets = [
        Target(
            range_m=1500,
            velocity_mps=-1.875,
            amplitude=1.0
        ),
        Target(
            range_m=3000,
            velocity_mps=0.9375,
            amplitude=0.7
        ),
        Target(
            range_m=4500,
            velocity_mps=2.8125,
            amplitude=0.5
        ),
    ]

    print("\nRADARIQ - INTEGRATED TARGET DETECTION")
    print("======================================")

    # ==========================================================
    # STEP 2: SIMULATE RECEIVED RADAR SIGNAL
    # ==========================================================

    received_pulses = simulate_received_pulses(
        config,
        targets
    )

    # ==========================================================
    # STEP 3: GENERATE RANGE-DOPPLER MAP
    # ==========================================================

    result = process_range_doppler(
        received_pulses,
        config
    )

    # The Range-Doppler map contains complex-valued samples.
    # Squared magnitude converts it into a real-valued power map.
    range_doppler_power = (
        np.abs(result.range_doppler_map) ** 2
    )

    # ==========================================================
    # STEP 4: APPLY TWO-DIMENSIONAL CA-CFAR
    # ==========================================================

    detections, threshold_map, noise_map = ca_cfar_2d(
        power_map=range_doppler_power,
        guard_cells=(2, 2),
        training_cells=(4, 4),
        pfa=0.01
    )

    # ==========================================================
    # STEP 5: EXTRACT TARGET CANDIDATES
    # ==========================================================

    extracted_targets = extract_targets(
        detections=detections,
        power_map=range_doppler_power,
        range_axis_m=result.range_axis_m,
        velocity_axis_mps=result.velocity_axis_mps,
        min_cluster_size=2
    )

    # ==========================================================
    # STEP 6: DISPLAY PROCESSING STATISTICS
    # ==========================================================

    valid_cut_count = np.count_nonzero(
        np.isfinite(threshold_map)
    )

    detection_cell_count = np.count_nonzero(detections)

    print(f"\nRange-Doppler map shape: {range_doppler_power.shape}")
    print("Desired Pfa: 0.01")
    print(f"Total valid CUTs: {valid_cut_count}")
    print(f"Total CFAR detection cells: {detection_cell_count}")
    print(f"Extracted detection clusters: {len(extracted_targets)}")

    # ==========================================================
    # STEP 7: RANK CANDIDATES BY PEAK POWER
    # ==========================================================

    # Stronger radar echoes generally produce higher peak power.
    # This ranking helps us inspect the strongest candidates first.
    strongest_targets = sorted(
        extracted_targets,
        key=lambda target: target.peak_power,
        reverse=True
    )

    print("\nTOP 15 CANDIDATES BY PEAK POWER")
    print("================================")

    for index, target in enumerate(
        strongest_targets[:15],
        start=1
    ):

        print(f"\nCandidate {index}")
        print("----------------")

        print(f"Range: {target.range_m:.2f} m")

        print(
            f"Velocity: {target.velocity_mps:+.4f} m/s"
        )

        print(
            f"Peak power: {target.peak_power:.4f}"
        )

        print(
            f"Detection cells: {target.num_detection_cells}"
        )

    # ==========================================================
    # STEP 8: DISPLAY CONFIGURED TARGETS
    # ==========================================================

    print("\nCONFIGURED TARGETS")
    print("==================")

    for index, target in enumerate(targets, start=1):

        print(f"\nExpected Target {index}")

        print(f"Range: {target.range_m:.2f} m")

        print(
            f"Velocity: {target.velocity_mps:+.4f} m/s"
        )

        print(f"Amplitude: {target.amplitude:.2f}")

    # ==========================================================
    # STEP 9: FINISH
    # ==========================================================

    print("\nPROCESSING COMPLETE")


if __name__ == "__main__":
    main()
