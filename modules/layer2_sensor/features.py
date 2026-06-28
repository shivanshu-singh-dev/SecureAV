"""
features.py
Layer 2 - Synthetic Sensor Stream Detection

Extracts a small "noise fingerprint" feature vector from a single camera frame.
Real camera sensors produce noise in specific, physically-determined patterns
(thermal noise, lens aberrations, temporal correlation between frames).
Synthetic / replayed / AI-generated frames tend to be statistically "too clean"
or have the wrong kind of noise pattern.

This module turns a raw frame (and, optionally, the previous frame) into a
short numeric feature vector that a simple classifier can use to tell the
difference.

No deep learning here on purpose - these are handcrafted, physically motivated
features. That keeps this fast (edge-friendly), interpretable (defensible to
judges), and trainable on a small dataset.
"""

import numpy as np
import cv2


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Ensure frame is single-channel grayscale, float32."""
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.astype(np.float32)


def residual_noise(frame: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    """
    High-frequency residual = frame - blurred(frame).
    This isolates the fine-grained noise/texture that a blur would smooth away.
    Real sensor noise lives here. Over-smoothed synthetic frames have very
    little energy here.
    """
    gray = _to_gray(frame)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return gray - blurred


def fft_high_freq_energy(residual: np.ndarray) -> float:
    """
    Energy in the high-frequency band of the residual's 2D FFT.
    Real noise has a roughly flat high-frequency spectrum.
    Smoothed / generated content drops off sharply at high frequencies.
    """
    f = np.fft.fft2(residual)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    # "high frequency" = outside the central 25% region
    mask = np.ones((h, w), dtype=bool)
    ry, rx = int(h * 0.25), int(w * 0.25)
    mask[cy - ry:cy + ry, cx - rx:cx + rx] = False

    high_freq_energy = magnitude[mask].mean()
    total_energy = magnitude.mean() + 1e-8
    return float(high_freq_energy / total_energy)


def temporal_correlation(residual_t: np.ndarray, residual_t_minus_1: np.ndarray) -> float:
    """
    Normalized cross-correlation between this frame's noise residual and the
    previous frame's noise residual.

    Real camera noise has a specific (often low but non-zero, and physically
    consistent) correlation pattern between consecutive frames caused by real
    sensor readout characteristics. Replayed/looped or AI-generated streams
    tend to have an unnaturally high (identical frames) or unnaturally low
    (independently generated noise) correlation.

    Returns a value in roughly [-1, 1].
    """
    a = residual_t.flatten()
    b = residual_t_minus_1.flatten()

    a = a - a.mean()
    b = b - b.mean()

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def extract_features(frame: np.ndarray, prev_frame: np.ndarray = None) -> dict:
    """
    Main entry point. Extracts the full feature vector for one frame.

    Args:
        frame: current camera frame (BGR or grayscale numpy array)
        prev_frame: previous camera frame, same format. Optional - if not
                    provided, temporal_corr is returned as 0.0 (neutral).

    Returns:
        dict of named features. Keep this a dict (not a bare array) so the
        feature order is self-documenting and won't silently break if a
        feature is added later.
    """
    residual = residual_noise(frame)

    noise_std = float(residual.std())
    noise_mean_abs = float(np.abs(residual).mean())
    hf_energy_ratio = fft_high_freq_energy(residual)

    if prev_frame is not None:
        prev_residual = residual_noise(prev_frame)
        temporal_corr = temporal_correlation(residual, prev_residual)
    else:
        temporal_corr = 0.0

    return {
        "noise_std": noise_std,
        "noise_mean_abs": noise_mean_abs,
        "hf_energy_ratio": hf_energy_ratio,
        "temporal_corr": temporal_corr,
    }


def features_to_vector(features: dict) -> np.ndarray:
    """
    Converts the feature dict into a fixed-order numpy vector for the model.
    Keeping this as a separate explicit function (rather than relying on
    dict ordering) so training and inference can never silently mismatch.
    """
    order = ["noise_std", "noise_mean_abs", "hf_energy_ratio", "temporal_corr"]
    return np.array([features[k] for k in order], dtype=np.float32)


def frame_hash(frame: np.ndarray, hash_size: int = 16) -> str:
    """
    Cheap perceptual hash: downscale to hash_size x hash_size grayscale,
    flatten to a string. Used to catch REPLAY attacks specifically.

    Why this is a separate mechanism from the noise classifier:
    a replayed frame is a genuine real frame reused verbatim. Its noise
    statistics are, by definition, identical to real sensor noise - there
    is no statistical feature that can distinguish "this real-looking frame
    is live" from "this real-looking frame is a recording of a past moment."
    The only way to catch a replay is to notice the frame (or its near-exact
    noise pattern) has already been seen before. That is a duplicate/identity
    check, not a classification problem.
    """
    gray = _to_gray(frame)
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    quantized = (small / 8).astype(np.uint8)  # coarse quantization, tolerant to minor re-encoding noise
    return quantized.tobytes().hex()
