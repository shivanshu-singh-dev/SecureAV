"""
generate_camera_dataset.py
Layer 2 - Synthetic Sensor Stream Detection

Generates a training-ready camera dataset:
    data/real/        - genuine frames (varied scenes + realistic sensor noise)
    data/synthetic/    - attack frames (3 attack types, see below)

Why generate rather than scrape: a real CARLA capture is the "production"
data source (see README), but this script gives you an instant, large,
class-balanced dataset to validate and pre-train the pipeline right now,
with full control over variety - which matters more for classifier
performance than raw photorealism, since the model is learning noise
statistics, not scene content.

Variety dimensions (so the model learns the real signal, not one scene):
    - scene layout: gradient direction, simulated "road" band, simulated
      "object" blobs (stand-ins for cars/pedestrians/signs)
    - lighting: brightness and contrast jitter per sample
    - sensor noise profile: per-sample-varied std/type, mimicking different
      cameras / lighting conditions a real fleet would see

Attack types generated for "synthetic" (label=1):
    1. denoised   - real frame with noise stripped out (too clean)
    2. replayed   - a real frame reused verbatim as a "different" frame
                    (zero new noise vs. its true previous frame)
    3. smoothed   - aggressive blur, simulating a low-effort AI upscale/regen

Usage:
    python generate_camera_dataset.py --samples 400
"""

import os
import argparse
import numpy as np
import cv2


def make_scene(h, w, rng, scene_seed):
    """
    Builds one synthetic 'road scene' base image (no sensor noise yet):
    a gradient (stand-in for sky-to-road perspective) plus a road band and
    a few object blobs (stand-ins for vehicles/pedestrians/signs), with
    randomized brightness/contrast so frames aren't all visually identical.
    """
    srng = np.random.RandomState(scene_seed)

    # base sky-to-ground gradient, direction varies per scene
    direction = srng.choice(["vertical", "horizontal", "diagonal"])
    if direction == "vertical":
        base = np.tile(np.linspace(40, 220, h).reshape(h, 1), (1, w))
    elif direction == "horizontal":
        base = np.tile(np.linspace(40, 220, w), (h, 1))
    else:
        yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
        base = 40 + 180 * (0.5 * yy + 0.5 * xx)

    scene = base.astype(np.float32)

    # simulated road band (darker horizontal strip in lower half)
    road_top = int(h * srng.uniform(0.55, 0.75))
    scene[road_top:, :] *= srng.uniform(0.5, 0.7)

    # a few object blobs at random positions/sizes (stand-ins for cars/peds/signs)
    n_objects = srng.randint(1, 4)
    for _ in range(n_objects):
        cy = srng.randint(int(h * 0.2), int(h * 0.9))
        cx = srng.randint(0, w)
        radius = srng.randint(8, 28)
        brightness = srng.uniform(0.3, 1.6)
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        scene[mask] *= brightness

    # brightness/contrast jitter so lighting conditions vary across samples
    brightness_shift = srng.uniform(-25, 25)
    contrast_scale = srng.uniform(0.85, 1.15)
    scene = (scene - 128) * contrast_scale + 128 + brightness_shift

    return np.clip(scene, 0, 255).astype(np.float32)


def add_sensor_noise(scene, rng):
    """
    Adds physically-motivated sensor noise:
    - Gaussian thermal noise (std varies per sample -> different camera units)
    - mild salt-and-pepper component (sensor hot pixels), sparse
    This is what real_world frames have and synthetic ones lack/get wrong.
    """
    noise_std = rng.uniform(5, 12)
    noisy = scene + rng.normal(0, noise_std, scene.shape)

    # sparse hot-pixel noise
    hot_pixel_mask = rng.random(scene.shape) < 0.0015
    noisy[hot_pixel_mask] = 255

    return np.clip(noisy, 0, 255).astype(np.uint8)


def make_denoised_attack(real_frame, rng):
    """Attack type 1: strip noise via blur -> 'too clean' synthetic stream."""
    return cv2.GaussianBlur(real_frame, (5, 5), 0)


def make_smoothed_attack(real_frame, rng):
    """Attack type 2: aggressive smoothing, simulating low-effort AI regen."""
    small = cv2.resize(real_frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (real_frame.shape[1], real_frame.shape[0]), interpolation=cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=400, help="samples per class (real / synthetic)")
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    real_dir = os.path.join(args.out, "real")
    synth_dir = os.path.join(args.out, "synthetic")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(synth_dir, exist_ok=True)

    master_rng = np.random.RandomState(args.seed)

    print(f"Generating {args.samples} real frames and {args.samples} synthetic (attack) frames...")
    print(f"Frame size: {args.height}x{args.width}")

    attack_types = ["denoised", "smoothed", "replayed"]
    prev_real_frame = None

    for i in range(args.samples):
        scene_seed = master_rng.randint(0, 10_000_000)
        frame_rng = np.random.RandomState(master_rng.randint(0, 10_000_000))

        scene = make_scene(args.height, args.width, frame_rng, scene_seed)
        real_frame = add_sensor_noise(scene, frame_rng)

        cv2.imwrite(os.path.join(real_dir, f"real_{i:04d}.png"), real_frame)

        # rotate through attack types for class balance and variety
        attack_type = attack_types[i % len(attack_types)]
        if attack_type == "denoised":
            attack_frame = make_denoised_attack(real_frame, frame_rng)
        elif attack_type == "smoothed":
            attack_frame = make_smoothed_attack(real_frame, frame_rng)
        else:  # replayed - reuse a previous real frame verbatim as if it were "live"
            attack_frame = prev_real_frame if prev_real_frame is not None else real_frame

        cv2.imwrite(os.path.join(synth_dir, f"synthetic_{i:04d}_{attack_type}.png"), attack_frame)

        prev_real_frame = real_frame

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{args.samples} pairs generated")

    print(f"\nDone.")
    print(f"  Real frames:      {real_dir}  ({args.samples} files)")
    print(f"  Synthetic frames: {synth_dir}  ({args.samples} files, mix of denoised/smoothed/replayed)")


if __name__ == "__main__":
    main()
