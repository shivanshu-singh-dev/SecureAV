"""
cross_modal.py
Layer 2 - Synthetic Sensor Stream Detection

Checks whether the camera and LiDAR agree on what physically exists in front
of the vehicle. Two real sensors looking at the same scene cannot legitimately
disagree about whether a solid object is present in a given region. If they
do disagree, one of the streams has likely been tampered with or replaced.

This is intentionally rule-based, not a trained model - the underlying logic
("can two sensors describe contradictory physical realities") doesn't need
to be learned from data, it just needs to be checked.
"""

from dataclasses import dataclass


@dataclass
class RegionObservation:
    """
    A simplified region-level observation. In a real pipeline this would come
    from the camera's object detector (bounding box -> region) and from
    clustering LiDAR points into the same region grid.
    """
    region_id: str
    camera_object_present: bool
    lidar_object_present: bool
    lidar_point_density: float = 0.0  # points per m^2 in this region, optional


def check_region_agreement(observation: RegionObservation) -> dict:
    """
    Compares one region's camera vs LiDAR reading.

    Returns a dict describing whether they agree, and if not, which kind of
    mismatch it is (useful for the dashboard / forensic log later).
    """
    cam = observation.camera_object_present
    lidar = observation.lidar_object_present

    if cam == lidar:
        return {
            "region_id": observation.region_id,
            "agree": True,
            "mismatch_type": None,
        }

    if lidar and not cam:
        mismatch_type = "camera_blind_spot"   # LiDAR sees something, camera reports nothing - classic "blanked" attack
    else:
        mismatch_type = "phantom_camera_object"  # camera sees something, LiDAR reports nothing - possible visual spoof

    return {
        "region_id": observation.region_id,
        "agree": False,
        "mismatch_type": mismatch_type,
    }


def check_frame_consistency(observations: list[RegionObservation]) -> dict:
    """
    Checks consistency across all regions in a single frame.

    Returns:
        {
            "mismatch_flag": bool,         # True if ANY region disagrees
            "mismatch_count": int,
            "mismatched_regions": [...],   # list of region-level results that disagreed
            "consistency_score": float,    # 0.0 (total disagreement) to 1.0 (full agreement)
        }
    """
    if not observations:
        # No regions to check - treat as consistent by default, nothing to flag
        return {
            "mismatch_flag": False,
            "mismatch_count": 0,
            "mismatched_regions": [],
            "consistency_score": 1.0,
        }

    results = [check_region_agreement(obs) for obs in observations]
    mismatched = [r for r in results if not r["agree"]]

    consistency_score = 1.0 - (len(mismatched) / len(results))

    return {
        "mismatch_flag": len(mismatched) > 0,
        "mismatch_count": len(mismatched),
        "mismatched_regions": mismatched,
        "consistency_score": float(consistency_score),
    }
