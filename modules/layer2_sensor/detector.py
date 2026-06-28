"""
detector.py
Layer 2 - Synthetic Sensor Stream Detection

Public integration point for the rest of the team. Mirrors the Layer 3
pattern:

    from modules.layer2_sensor.detector import Layer2Detector
    detector = Layer2Detector()                  # call once at startup
    result = detector.detect(camera_frame, regions)   # call every cycle

See README.md for the full input/output contract.
"""

import pickle
from collections import deque
import numpy as np

from features import extract_features, features_to_vector, frame_hash
from cross_modal import RegionObservation, check_frame_consistency

MODEL_PATH = "model.pkl"

# Weights for combining the three sub-scores into one alert_score.
# Fingerprint check currently weighted higher since it runs every frame;
# cross-modal check is weighted lower since it depends on region data being
# available. Tune these once you have real validation numbers.
FINGERPRINT_WEIGHT = 0.5
CROSS_MODAL_WEIGHT = 0.3
REPLAY_WEIGHT = 0.2

# How many recent frame hashes to remember for replay detection.
# At ~30fps, 300 frames = ~10 seconds of history. A genuine live feed should
# never produce two near-identical frames 10 seconds apart at highway speed.
REPLAY_HISTORY_SIZE = 300


class Layer2Detector:
    """
    NOTE on why there are two distinct mechanisms for "is this frame fake":

    1. Hardware fingerprint classifier - catches frames whose NOISE
       STATISTICS don't match real sensor physics (denoised, smoothed,
       AI-regenerated frames). This is a classification problem.

    2. Replay/duplicate detector - catches frames that are genuine, real,
       correctly-noised frames... that have simply been SEEN BEFORE and are
       being fed back in as if live. A replayed frame's noise statistics are
       indistinguishable from real data by definition, so the fingerprint
       classifier structurally cannot catch this attack type. Catching it
       requires checking frame identity/history, not frame statistics.

    Both run every cycle; their scores are combined in detect().
    """

    def __init__(self, model_path: str = MODEL_PATH):
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        self.classifier = saved["classifier"]
        self.scaler = saved["scaler"]
        self.training_accuracy = saved.get("accuracy")

        self._prev_frame = None  # for temporal correlation between calls
        self._recent_hashes = deque(maxlen=REPLAY_HISTORY_SIZE)

    def _fingerprint_score(self, camera_frame: np.ndarray) -> float:
        """
        Returns probability that this frame is synthetic/tampered, in [0, 1].
        """
        feats = extract_features(camera_frame, prev_frame=self._prev_frame)
        vec = features_to_vector(feats).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)

        # predict_proba -> [P(real), P(synthetic)]
        proba = self.classifier.predict_proba(vec_scaled)[0]
        synthetic_proba = float(proba[1])

        self._prev_frame = camera_frame
        return synthetic_proba

    def _replay_score(self, camera_frame: np.ndarray) -> float:
        """
        Returns 1.0 if this exact frame (within hash tolerance) has appeared
        in recent history, 0.0 otherwise. A live feed should never repeat.
        """
        h = frame_hash(camera_frame)
        is_replay = h in self._recent_hashes
        self._recent_hashes.append(h)
        return 1.0 if is_replay else 0.0

    def detect(self, camera_frame: np.ndarray, regions: list = None) -> dict:
        """
        Args:
            camera_frame: current camera frame as a numpy array (BGR or grayscale)
            regions: optional list of RegionObservation (or equivalent dicts)
                     describing camera vs LiDAR object presence per region.
                     If omitted, the cross-modal check is skipped for this cycle.

        Returns:
            {
                "synthetic_flag": bool,        # True = fingerprint suggests fake stream
                "fingerprint_score": float,    # 0.0 (genuine) - 1.0 (synthetic)
                "replay_flag": bool,           # True = this exact frame seen before recently
                "cross_modal_mismatch": bool,  # True = camera/LiDAR disagree
                "consistency_score": float,    # 0.0 (full disagreement) - 1.0 (full agreement)
                "alert_score": float,          # 0.0-1.0, combined - feed into Dempster-Shafer
            }
        """
        fingerprint_score = self._fingerprint_score(camera_frame)
        synthetic_flag = fingerprint_score > 0.5

        replay_score = self._replay_score(camera_frame)
        replay_flag = replay_score > 0.5

        if regions:
            region_objs = [
                r if isinstance(r, RegionObservation) else RegionObservation(**r)
                for r in regions
            ]
            cross_modal_result = check_frame_consistency(region_objs)
            cross_modal_mismatch = cross_modal_result["mismatch_flag"]
            consistency_score = cross_modal_result["consistency_score"]
            cross_modal_alert = 1.0 - consistency_score
        else:
            cross_modal_mismatch = False
            consistency_score = 1.0
            cross_modal_alert = 0.0

        if regions:
            alert_score = (
                FINGERPRINT_WEIGHT * fingerprint_score
                + CROSS_MODAL_WEIGHT * cross_modal_alert
                + REPLAY_WEIGHT * replay_score
            )
        else:
            # renormalize across just fingerprint + replay when no region data this cycle
            alert_score = (
                (FINGERPRINT_WEIGHT / (FINGERPRINT_WEIGHT + REPLAY_WEIGHT)) * fingerprint_score
                + (REPLAY_WEIGHT / (FINGERPRINT_WEIGHT + REPLAY_WEIGHT)) * replay_score
            )

        return {
            "synthetic_flag": synthetic_flag,
            "fingerprint_score": round(fingerprint_score, 4),
            "replay_flag": replay_flag,
            "cross_modal_mismatch": cross_modal_mismatch,
            "consistency_score": round(consistency_score, 4),
            "alert_score": round(float(alert_score), 4),
        }
