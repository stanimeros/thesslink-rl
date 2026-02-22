"""
ThessLink-RL Environment for Near4all Research Project
Gymnasium environment for RL-based Meeting Point (POI) suggestion in Thessaloniki.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

try:
    import osmnx as ox
except ImportError:
    raise ImportError("osmnx is required. Install with: pip install osmnx")

# Use project-local OSM cache to avoid repeated Overpass API calls (OSMnx 2.x)
_OSM_CACHE_DIR = Path(__file__).resolve().parent / "cache"
ox.settings.use_cache = True
ox.settings.cache_folder = str(_OSM_CACHE_DIR)

try:
    from geopy.distance import geodesic
except ImportError:
    raise ImportError("geopy is required. Install with: pip install geopy")

# Thessaloniki bounding box (~6km × 6km city center, stays under Overpass query limits)
THESSALONIKI_BBOX = {
    "north": 40.655,
    "south": 40.598,
    "east": 22.985,
    "west": 22.910,
}

# Privacy scores by POI type (higher = more private)
PRIVACY_SCORES = {
    "park": 0.9,
    "leisure": 0.9,  # fallback for parks
    "cafe": 0.5,
    "restaurant": 0.3,
    "default": 0.4,
}

logger = logging.getLogger(__name__)


def _get_poi_type(row: pd.Series) -> str:
    """Infer POI type from OSM tags for privacy scoring."""
    amenity = row.get("amenity", "")
    leisure = row.get("leisure", "")
    if leisure == "park":
        return "park"
    if amenity == "cafe":
        return "cafe"
    if amenity == "restaurant":
        return "restaurant"
    return "default"


def _get_poi_coords(geometry) -> tuple[float, float]:
    """Extract (lat, lon) from a GeoDataFrame geometry."""
    if geometry is None:
        return (np.nan, np.nan)
    if hasattr(geometry, "x") and hasattr(geometry, "y"):
        return (geometry.y, geometry.x)  # GeoPandas uses (x=lon, y=lat)
    centroid = geometry.centroid if hasattr(geometry, "centroid") else geometry
    return (centroid.y, centroid.x)


def fetch_thessaloniki_pois(
    top_n: int = 50,
    place_name: str = "Thessaloniki, Greece",
    use_bbox_fallback: bool = True,
) -> pd.DataFrame:
    """
    Fetch POIs (cafe, restaurant, park) in Thessaloniki using OSMnx.

    Args:
        top_n: Maximum number of POIs to return (by proximity to city center).
        place_name: Place to search (used if bbox fetch fails).
        use_bbox_fallback: If True, fall back to bbox when place geocoding fails.

    Returns:
        DataFrame with columns: lat, lon, poi_type, privacy_score, geometry.

    Raises:
        ValueError: If no POIs are found after all attempts.
    """
    all_pois = []
    center = (40.6264, 22.9484)  # Thessaloniki center (White Tower area)
    dist_m = 2500  # 2.5 km radius → ~5 km box, avoids Overpass subdivision

    def _fetch(tags: dict, label: str) -> pd.DataFrame:
        try:
            logger.info("Fetching %s from OSM (~%d m radius)...", label, dist_m)
            return ox.features_from_point(center, tags=tags, dist=dist_m)
        except Exception as e:
            logger.warning("Point fetch failed: %s. Trying place...", e)
            if use_bbox_fallback:
                return ox.features_from_place(place_name, tags=tags)
            raise

    try:
        all_pois.append(_fetch({"amenity": ["cafe", "restaurant"]}, "cafes & restaurants"))
    except Exception as e:
        logger.warning("Could not fetch amenities: %s", e)
    try:
        all_pois.append(_fetch({"leisure": "park"}, "parks"))
    except Exception as e:
        logger.warning("Could not fetch parks: %s", e)

    if not all_pois:
        raise ValueError(
            "No POIs found in Thessaloniki. Check network connection and OSMnx/Overpass availability."
        )

    combined = pd.concat([gdf for gdf in all_pois if gdf is not None and len(gdf) > 0])
    combined = combined[~combined.geometry.isna()].copy()

    if len(combined) == 0:
        raise ValueError("No valid POI geometries found in Thessaloniki.")

    # Extract coordinates and metadata
    records = []
    center = (40.6264, 22.9484)  # Thessaloniki center
    for idx, row in combined.iterrows():
        try:
            lat, lon = _get_poi_coords(row.geometry)
            if np.isnan(lat) or np.isnan(lon):
                continue
            poi_type = _get_poi_type(row)
            privacy = PRIVACY_SCORES.get(poi_type, PRIVACY_SCORES["default"])
            dist_to_center = geodesic(center, (lat, lon)).km
            records.append({"lat": lat, "lon": lon, "poi_type": poi_type, "privacy_score": privacy, "dist_to_center": dist_to_center})
        except Exception as e:
            logger.debug("Skipping POI %s: %s", idx, e)
            continue

    if not records:
        raise ValueError("No POIs with valid coordinates could be extracted.")

    df = pd.DataFrame(records)
    df = df.sort_values("dist_to_center").head(top_n).reset_index(drop=True)

    return df


class ThessLinkEnv(gym.Env):
    """
    Gymnasium environment for learning optimal Meeting Point (POI) selection
    between two users in Thessaloniki.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        pois: pd.DataFrame | None = None,
        top_n: int = 50,
        weight_distance: float = 0.5,
        weight_privacy: float = 0.5,
        distance_scale_km: float = 10.0,
        max_steps_per_episode: int = 1,
        seed: int | None = None,
    ):
        """
        Args:
            pois: Pre-fetched POI DataFrame. If None, fetches from OSM.
            top_n: Number of POIs in action space (used when pois is None).
            weight_distance: Weight for minimizing travel distance (Energy).
            weight_privacy: Weight for maximizing privacy score.
            distance_scale_km: Scale factor for distance-based reward component.
            max_steps_per_episode: Steps before truncation (1 = single-step decision).
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self.weight_distance = weight_distance
        self.weight_privacy = weight_privacy
        self.distance_scale_km = distance_scale_km
        self.max_steps_per_episode = max_steps_per_episode

        if pois is not None and len(pois) > 0:
            self.pois = pois.copy()
        else:
            self.pois = fetch_thessaloniki_pois(top_n=top_n)

        self.n_pois = len(self.pois)
        if self.n_pois == 0:
            raise ValueError("Cannot create environment with 0 POIs.")

        # Bounds for normalizing observations
        self._lat_min = THESSALONIKI_BBOX["south"]
        self._lat_max = THESSALONIKI_BBOX["north"]
        self._lon_min = THESSALONIKI_BBOX["west"]
        self._lon_max = THESSALONIKI_BBOX["east"]

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(self.n_pois)
        self._step_count = 0

    def _sample_user_position(self) -> tuple[float, float]:
        lat = self._rng.uniform(self._lat_min, self._lat_max)
        lon = self._rng.uniform(self._lon_min, self._lon_max)
        return (lat, lon)

    def _normalize_obs(self, lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> np.ndarray:
        def norm(v, lo, hi):
            return (v - lo) / (hi - lo) if hi != lo else 0.5

        return np.array(
            [
                norm(lat_a, self._lat_min, self._lat_max),
                norm(lon_a, self._lon_min, self._lon_max),
                norm(lat_b, self._lat_min, self._lat_max),
                norm(lon_b, self._lon_min, self._lon_max),
            ],
            dtype=np.float32,
        )

    def _compute_reward(
        self,
        lat_a: float, lon_a: float,
        lat_b: float, lon_b: float,
        action: int,
    ) -> tuple[float, float, float]:
        """Returns (total_reward, total_distance_km, privacy_score)."""
        row = self.pois.iloc[action]
        poi_lat, poi_lon = row["lat"], row["lon"]
        privacy = row["privacy_score"]

        dist_a = geodesic((lat_a, lon_a), (poi_lat, poi_lon)).km
        dist_b = geodesic((lat_b, lon_b), (poi_lat, poi_lon)).km
        total_dist = dist_a + dist_b

        # Reward: minimize distance (penalize), maximize privacy (reward)
        # Normalize distance by scale so reward is roughly in [-1, 1]
        dist_component = -total_dist / self.distance_scale_km
        reward = (
            self.weight_distance * dist_component +
            self.weight_privacy * privacy
        )
        return reward, total_dist, privacy

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0

        lat_a, lon_a = self._sample_user_position()
        lat_b, lon_b = self._sample_user_position()

        self._current_lat_a, self._current_lon_a = lat_a, lon_a
        self._current_lat_b, self._current_lon_b = lat_b, lon_b

        obs = self._normalize_obs(lat_a, lon_a, lat_b, lon_b)
        return obs, {"lat_a": lat_a, "lon_a": lon_a, "lat_b": lat_b, "lon_b": lon_b}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1

        lat_a, lon_a = self._current_lat_a, self._current_lon_a
        lat_b, lon_b = self._current_lat_b, self._current_lon_b

        reward, total_dist, privacy = self._compute_reward(lat_a, lon_a, lat_b, lon_b, action)
        obs = self._normalize_obs(lat_a, lon_a, lat_b, lon_b)

        terminated = True
        truncated = self._step_count >= self.max_steps_per_episode

        info = {
            "total_distance_km": total_dist,
            "privacy_score": privacy,
            "selected_poi_index": int(action),
            "lat_a": lat_a,
            "lon_a": lon_a,
            "lat_b": lat_b,
            "lon_b": lon_b,
        }

        return obs, float(reward), terminated, truncated, info

    def get_pois(self) -> pd.DataFrame:
        """Return the current POI DataFrame (for inference)."""
        return self.pois.copy()
