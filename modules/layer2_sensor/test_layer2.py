"""
test_layer2.py
Quick sanity/scenario test suite for Layer 2 (Synthetic Sensor Stream Detection).

Loads real frames from data/real/ and data/synthetic/ (the same dataset
used for training) rather than generating standalone dummy frames - this
ensures the test reflects the actual distribution the model was trained
and validated on. If you've replaced data/ with your own CARLA capture,
this test will automatically use that instead.

Run from inside modules/layer2_sensor/ (or adjust the import path) with:
    python test_layer2.py
"""

import os
import time
import glob
import cv2

from detector import Layer2Detector

DATA_DIR = "data"


def load_sample(directory: str, index: int = 0):
    """Loads the Nth image file from a directory (sorted, deterministic)."""
    files = sorted(glob.glob(os.path.join(directory, "*.png")) +
                    glob.glob(os.path.join(directory, "*.jpg")))
    if not files:
        raise FileNotFoundError(
            f"No image files found in '{directory}'. "
            f"Run generate_camera_dataset.py first, or point DATA_DIR at your own data."
        )
    if index >= len(files):
        index = index % len(files)
    return cv2.imread(files[index])


def load_sample_by_substring(directory: str, substring: str, index: int = 0):
    """Loads the Nth file in a directory whose name contains `substring`
    (e.g. 'denoised', 'smoothed') - useful for picking a specific attack type."""
    files = sorted(f for f in glob.glob(os.path.join(directory, "*.png")) if substring in f)
    if not files:
        raise FileNotFoundError(f"No files containing '{substring}' found in '{directory}'.")
    if index >= len(files):
        index = index % len(files)
    return cv2.imread(files[index])


def run_scenario(name, detector, frame, regions=None):
    start = time.perf_counter()
    result = detector.detect(frame, regions=regions)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"\n[{name}]")
    print(f"  result: {result}")
    print(f"  inference time: {elapsed_ms:.2f} ms")
    return result, elapsed_ms


def main():
    detector = Layer2Detector(model_path="model.pkl")

    real_dir = os.path.join(DATA_DIR, "real")
    synth_dir = os.path.join(DATA_DIR, "synthetic")

    print("=" * 60)
    print("Layer 2 scenario tests")
    print("=" * 60)

    # Scenario 1: genuine frame, no region data -> expect low alert_score
    real_frame = load_sample(real_dir, index=1)
    run_scenario("1. Genuine frame, no LiDAR data", detector, real_frame)

    # Scenario 2: synthetic/attack frame (denoised) -> expect high alert_score
    synth_frame = load_sample_by_substring(synth_dir, "denoised", index=0)
    run_scenario("2. Synthetic (denoised, too-clean) frame", detector, synth_frame)

    # Scenario 2b: synthetic/attack frame (smoothed) -> expect high alert_score
    smoothed_frame = load_sample_by_substring(synth_dir, "smoothed", index=0)
    run_scenario("2b. Synthetic (smoothed) frame", detector, smoothed_frame)

    # Scenario 3: genuine frame + consistent regions -> expect very low alert_score
    consistent_regions = [
        {"region_id": "front_left", "camera_object_present": False, "lidar_object_present": False},
        {"region_id": "front_center", "camera_object_present": True, "lidar_object_present": True},
        {"region_id": "front_right", "camera_object_present": False, "lidar_object_present": False},
    ]
    run_scenario(
        "3. Genuine frame, camera/LiDAR agree",
        detector,
        load_sample(real_dir, index=2),
        regions=consistent_regions,
    )

    # Scenario 4: genuine frame BUT camera/LiDAR disagree (blanked pedestrian attack)
    mismatched_regions = [
        {"region_id": "front_left", "camera_object_present": False, "lidar_object_present": False},
        {"region_id": "front_center", "camera_object_present": False, "lidar_object_present": True},  # mismatch!
        {"region_id": "front_right", "camera_object_present": False, "lidar_object_present": False},
    ]
    run_scenario(
        "4. Genuine frame, camera/LiDAR DISAGREE (attack)",
        detector,
        load_sample(real_dir, index=3),
        regions=mismatched_regions,
    )

    # Scenario 5: replay attack - feed the SAME frame twice in a row
    replay_frame = load_sample(real_dir, index=4)
    run_scenario("5a. Genuine frame (first time seen)", detector, replay_frame)
    run_scenario("5b. Same frame replayed (attack)", detector, replay_frame)

    print("\n" + "=" * 60)
    print("Expected: scenarios 1, 3, 5a -> low alert_score")
    print("          scenarios 2, 2b    -> high alert_score, synthetic_flag=True")
    print("          scenario 4         -> elevated alert_score, cross_modal_mismatch=True")
    print("          scenario 5b        -> elevated alert_score, replay_flag=True")
    print("=" * 60)


if __name__ == "__main__":
    main()