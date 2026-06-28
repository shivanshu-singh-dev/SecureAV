"""
generate_lidar_dataset.py
Layer 2 - Synthetic Sensor Stream Detection (cross-modal consistency data)

Generates region-level camera-vs-LiDAR observation samples for testing/
validating the cross_modal.py consistency check, and for tuning the
mismatch-weighting in detector.py.

Each sample = one frame's worth of region observations:
    {
        "frame_id": int,
        "regions": [
            {"region_id": str, "camera_object_present": bool,
             "lidar_object_present": bool, "lidar_point_density": float},
            ...
        ],
        "label": "consistent" | "attack"
    }

Two scenario classes generated (label-balanced):
    consistent  - camera and LiDAR agree on every region (normal driving)
    attack      - exactly one region is deliberately flipped, simulating
                  one of two real attack patterns:
                    - "camera_blind_spot": LiDAR sees an object, camera was
                      made to report nothing there (object hidden from vision)
                    - "phantom_camera_object": camera reports an object,
                      LiDAR sees nothing there (visual spoof / sticker attack)

Saved as a single JSON file (easy to load, easy to inspect, no image
encoding needed since this is structured/tabular data, not pixels).

Usage:
    python generate_lidar_dataset.py --samples 400
"""

import os
import json
import argparse
import numpy as np

REGION_LAYOUT = ["front_left", "front_center", "front_right", "rear_left", "rear_center", "rear_right"]


def make_consistent_sample(frame_id, rng):
    regions = []
    for region_id in REGION_LAYOUT:
        present = bool(rng.random() < 0.35)  # ~35% chance an object is in this region
        density = float(rng.uniform(8, 40)) if present else float(rng.uniform(0, 1.5))
        regions.append({
            "region_id": region_id,
            "camera_object_present": present,
            "lidar_object_present": present,  # agreement by construction
            "lidar_point_density": round(density, 2),
        })
    return {"frame_id": frame_id, "regions": regions, "label": "consistent"}


def make_attack_sample(frame_id, rng):
    # start from a consistent frame, then corrupt exactly one region
    sample = make_consistent_sample(frame_id, rng)
    sample["label"] = "attack"

    attack_region_idx = rng.randint(0, len(REGION_LAYOUT))
    attack_type = rng.choice(["camera_blind_spot", "phantom_camera_object"])

    region = sample["regions"][attack_region_idx]
    if attack_type == "camera_blind_spot":
        # LiDAR genuinely sees something solid; camera was tampered to hide it
        region["lidar_object_present"] = True
        region["lidar_point_density"] = float(rng.uniform(15, 40))
        region["camera_object_present"] = False
    else:
        # camera reports an object; LiDAR sees nothing there (spoofed visual)
        region["camera_object_present"] = True
        region["lidar_object_present"] = False
        region["lidar_point_density"] = float(rng.uniform(0, 1.0))

    region["lidar_point_density"] = round(region["lidar_point_density"], 2)
    sample["attack_type"] = attack_type
    sample["attack_region"] = region["region_id"]
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=400, help="samples per class (consistent / attack)")
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--seed", type=int, default=456)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    print(f"Generating {args.samples} consistent samples and {args.samples} attack samples...")

    dataset = []
    frame_id = 0
    for i in range(args.samples):
        dataset.append(make_consistent_sample(frame_id, rng))
        frame_id += 1
        dataset.append(make_attack_sample(frame_id, rng))
        frame_id += 1

    rng.shuffle(dataset)

    out_path = os.path.join(args.out, "lidar_cross_modal_dataset.json")
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)

    n_consistent = sum(1 for d in dataset if d["label"] == "consistent")
    n_attack = sum(1 for d in dataset if d["label"] == "attack")

    print(f"\nDone.")
    print(f"  Total samples: {len(dataset)} (consistent={n_consistent}, attack={n_attack})")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
