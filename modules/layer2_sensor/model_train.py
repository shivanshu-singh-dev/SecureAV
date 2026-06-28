"""
model_train.py
Layer 2 - Synthetic Sensor Stream Detection

Trains the hardware-fingerprint classifier: given a frame's noise feature
vector, predict whether it came from a genuine sensor (0) or a
synthetic/replayed/tampered stream (1).

Expected data layout (adjust paths below to match your actual dataset):

    data/
        real/        <- genuine camera frames (.png/.jpg)
        synthetic/   <- your generated attack frames (blurred/denoised/replayed)

Usage:
    python model_train.py
Produces:
    model.pkl   - trained classifier, loaded by detector.py at runtime
"""

import os
import pickle
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from features import extract_features, features_to_vector

DATA_DIR = "data"
REAL_DIR = os.path.join(DATA_DIR, "real")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")
MODEL_OUT = "model.pkl"


def load_frames_from_dir(directory: str, exclude_substring: str = None) -> list[np.ndarray]:
    frames = []
    if not os.path.isdir(directory):
        print(f"  [!] Warning: directory not found: {directory}")
        return frames

    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        if exclude_substring and exclude_substring in fname:
            continue
        path = os.path.join(directory, fname)
        frame = cv2.imread(path)
        if frame is not None:
            frames.append(frame)
    return frames


def build_dataset() -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading real frames from {REAL_DIR} ...")
    real_frames = load_frames_from_dir(REAL_DIR)
    print(f"  -> {len(real_frames)} real frames loaded")

    print(f"Loading synthetic/attack frames from {SYNTHETIC_DIR} ...")
    # NOTE: "replayed" frames are deliberately excluded from this classifier's
    # training set. A replayed frame is a genuine real frame reused verbatim -
    # its noise statistics are indistinguishable from real data by definition,
    # so including it here only adds label noise (a "1" example with "0"-like
    # features) and would actively hurt the model. Replay attacks are caught
    # separately in detector.py via frame-hash history, not noise statistics.
    synthetic_frames = load_frames_from_dir(SYNTHETIC_DIR, exclude_substring="replayed")
    print(f"  -> {len(synthetic_frames)} synthetic frames loaded (replayed frames excluded - see note)")

    if len(real_frames) == 0 or len(synthetic_frames) == 0:
        raise RuntimeError(
            "Need frames in both data/real/ and data/synthetic/ to train. "
            "See the README for how to generate synthetic/attack frames "
            "from your real footage."
        )

    X, y = [], []

    # label 0 = real/genuine
    prev = None
    for frame in real_frames:
        feats = extract_features(frame, prev_frame=prev)
        X.append(features_to_vector(feats))
        y.append(0)
        prev = frame

    # label 1 = synthetic/attack
    prev = None
    for frame in synthetic_frames:
        feats = extract_features(frame, prev_frame=prev)
        X.append(features_to_vector(feats))
        y.append(1)
        prev = frame

    return np.array(X), np.array(y)


def train_and_evaluate(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {acc * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["real", "synthetic"]))

    return clf, scaler, acc


def main():
    X, y = build_dataset()
    print(f"\nTotal samples: {len(X)} (real={sum(y == 0)}, synthetic={sum(y == 1)})")

    clf, scaler, acc = train_and_evaluate(X, y)

    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"classifier": clf, "scaler": scaler, "accuracy": acc}, f)

    print(f"\nSaved trained model -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
