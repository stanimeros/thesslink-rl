"""
ThessLink-RL Inference Script for Near4all Research Project
Uses the trained policy to suggest optimal Meeting Points between two users.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from env import THESSALONIKI_BBOX

try:
    from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
    from sb3_contrib import TRPO
except ImportError as e:
    raise ImportError(
        "stable-baselines3 and sb3-contrib are required. Install with: pip install stable-baselines3 sb3-contrib"
    ) from e

# Map algo name (from path) to loader
_MODEL_LOADERS = {
    "dqn": DQN.load,
    "ppo": PPO.load,
    "a2c": A2C.load,
    "trpo": TRPO.load,
    "sac": SAC.load,
    "td3": TD3.load,
    "ddpg": DDPG.load,
}
CONTINUOUS_ALGOS = {"sac", "td3", "ddpg"}

DEFAULT_MODEL_PATH = Path(__file__).parent / "policies" / "PPO" / "best_model.zip"
DEFAULT_POIS_PATH = Path(__file__).parent / "thesslink_pois.csv"


class MeetingPointSuggestion(NamedTuple):
    """Result of a meeting point suggestion."""

    lat: float
    lon: float
    poi_type: str
    privacy_score: float
    poi_index: int
    distance_a_km: float | None
    distance_b_km: float | None
    total_distance_km: float | None


def _normalize_coords(
    lat: float,
    lon: float,
    lat_min: float = THESSALONIKI_BBOX["south"],
    lat_max: float = THESSALONIKI_BBOX["north"],
    lon_min: float = THESSALONIKI_BBOX["west"],
    lon_max: float = THESSALONIKI_BBOX["east"],
) -> tuple[float, float]:
    def norm(v: float, lo: float, hi: float) -> float:
        return (v - lo) / (hi - lo) if hi != lo else 0.5

    return (norm(lat, lat_min, lat_max), norm(lon, lon_min, lon_max))


def suggest_meeting_point(
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    pois_path: str | Path = DEFAULT_POIS_PATH,
) -> MeetingPointSuggestion:
    """
    Suggest the best Meeting Point (POI) between two users in Thessaloniki
    using the trained PPO policy.

    Args:
        lat_a: Latitude of User A.
        lon_a: Longitude of User A.
        lat_b: Latitude of User B.
        lon_b: Longitude of User B.
        model_path: Path to the saved policy (thesslink_policy.zip).
        pois_path: Path to the POI CSV (thesslink_pois.csv).

    Returns:
        MeetingPointSuggestion with the recommended POI and metadata.

    Raises:
        FileNotFoundError: If model or POI file is missing.
    """
    model_path = Path(model_path)
    pois_path = Path(pois_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train with: python train_all_algos.py"
        )
    if not pois_path.exists():
        raise FileNotFoundError(
            f"POI file not found at {pois_path}. Train with: python train_all_algos.py"
        )

    # Infer algorithm from path (e.g. policies/DQN/best_model.zip → DQN)
    path_lower = model_path.as_posix().lower()
    loader = PPO.load  # default
    for name, load_fn in _MODEL_LOADERS.items():
        if name in path_lower:
            loader = load_fn
            break
    model = loader(str(model_path))
    pois = pd.read_csv(pois_path)

    if len(pois) == 0:
        raise ValueError("POI file is empty.")

    # Normalize user coordinates (same as env)
    na_lat, na_lon = _normalize_coords(lat_a, lon_a)
    nb_lat, nb_lon = _normalize_coords(lat_b, lon_b)
    obs = np.array([na_lat, na_lon, nb_lat, nb_lon], dtype=np.float32).reshape(1, -1)

    action, _ = model.predict(obs, deterministic=True)
    path_lower = model_path.as_posix().lower()
    is_continuous = any(a in path_lower for a in CONTINUOUS_ALGOS)

    if is_continuous and action.shape and action.size >= 2:
        # Action is (lat_norm, lon_norm); find nearest POI
        lat_norm, lon_norm = float(action[0]), float(action[1])
        lat_min, lat_max = THESSALONIKI_BBOX["south"], THESSALONIKI_BBOX["north"]
        lon_min, lon_max = THESSALONIKI_BBOX["west"], THESSALONIKI_BBOX["east"]
        target_lat = lat_min + np.clip(lat_norm, 0, 1) * (lat_max - lat_min)
        target_lon = lon_min + np.clip(lon_norm, 0, 1) * (lon_max - lon_min)
        from geopy.distance import geodesic
        dists = pois.apply(
            lambda r: geodesic((target_lat, target_lon), (r["lat"], r["lon"])).km, axis=1
        )
        poi_idx = int(dists.argmin())
    else:
        poi_idx = int(action[0])

    if poi_idx < 0 or poi_idx >= len(pois):
        poi_idx = min(max(0, poi_idx), len(pois) - 1)

    row = pois.iloc[poi_idx]
    poi_lat = float(row["lat"])
    poi_lon = float(row["lon"])

    try:
        from geopy.distance import geodesic

        dist_a = geodesic((lat_a, lon_a), (poi_lat, poi_lon)).km
        dist_b = geodesic((lat_b, lon_b), (poi_lat, poi_lon)).km
        total_dist = dist_a + dist_b
    except ImportError:
        dist_a = dist_b = total_dist = None

    return MeetingPointSuggestion(
        lat=poi_lat,
        lon=poi_lon,
        poi_type=str(row.get("poi_type", "unknown")),
        privacy_score=float(row.get("privacy_score", 0.5)),
        poi_index=poi_idx,
        distance_a_km=dist_a,
        distance_b_km=dist_b,
        total_distance_km=total_dist,
    )


def main() -> int:
    """CLI demo: suggest meeting point for two locations."""
    import argparse
    parser = argparse.ArgumentParser(description="Suggest meeting point using trained policy")
    parser.add_argument("coords", nargs="*", type=float, help="lat_a lon_a lat_b lon_b")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to policy")
    parser.add_argument("--pois-path", type=Path, default=DEFAULT_POIS_PATH, help="Path to POI CSV")
    args = parser.parse_args()

    if len(args.coords) >= 4:
        lat_a, lon_a, lat_b, lon_b = args.coords[0], args.coords[1], args.coords[2], args.coords[3]
    else:
        lat_a, lon_a = 40.6293, 22.9597  # near Aristotle University
        lat_b, lon_b = 40.6261, 22.9484  # near White Tower

    try:
        result = suggest_meeting_point(
            lat_a, lon_a, lat_b, lon_b,
            model_path=args.model_path, pois_path=args.pois_path,
        )
        print("Suggested Meeting Point:")
        print(f"  Location: ({result.lat:.6f}, {result.lon:.6f})")
        print(f"  Type: {result.poi_type}")
        print(f"  Privacy Score: {result.privacy_score:.2f}")
        if result.total_distance_km is not None:
            print(f"  Distance A→POI: {result.distance_a_km:.3f} km")
            print(f"  Distance B→POI: {result.distance_b_km:.3f} km")
            print(f"  Total travel: {result.total_distance_km:.3f} km")
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
