"""
ThessLink-RL Map Visualization Script for Near4all Research Project
Plots User A, User B, and the suggested Meeting Point on an interactive Folium map.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import folium
except ImportError:
    raise ImportError("folium is required. Install with: pip install folium")

from inference import suggest_meeting_point, MeetingPointSuggestion

DEFAULT_OUTPUT = Path(__file__).parent / "meeting_point_map.html"


def plot_meeting_point(
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
    result: MeetingPointSuggestion | None = None,
    output_path: str | Path = DEFAULT_OUTPUT,
    zoom_start: int = 13,
) -> Path:
    """
    Plot User A, User B, and the suggested Meeting Point on an interactive map.

    Args:
        lat_a: Latitude of User A.
        lon_a: Longitude of User A.
        lat_b: Latitude of User B.
        lon_b: Longitude of User B.
        result: Pre-computed suggestion, or None to run inference.
        output_path: Path to save the HTML map.
        zoom_start: Initial zoom level (default: 13).

    Returns:
        Path to the saved HTML file.
    """
    if result is None:
        result = suggest_meeting_point(lat_a, lon_a, lat_b, lon_b)

    # Center map on midpoint of the three locations
    center_lat = (lat_a + lat_b + result.lat) / 3
    center_lon = (lon_a + lon_b + result.lon) / 3

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="OpenStreetMap")

    # User A - blue
    folium.Marker(
        [lat_a, lon_a],
        popup=f"<b>User A</b><br>({lat_a:.6f}, {lon_a:.6f})",
        tooltip="User A",
        icon=folium.Icon(color="blue", icon="user"),
    ).add_to(m)

    # User B - green
    folium.Marker(
        [lat_b, lon_b],
        popup=f"<b>User B</b><br>({lat_b:.6f}, {lon_b:.6f})",
        tooltip="User B",
        icon=folium.Icon(color="green", icon="user"),
    ).add_to(m)

    # Suggested meeting point - red
    dist_info = ""
    if result.total_distance_km is not None:
        dist_info = f"<br>Total travel: {result.total_distance_km:.2f} km"
    folium.Marker(
        [result.lat, result.lon],
        popup=f"<b>Suggested Meeting Point</b><br>{result.poi_type}<br>Privacy: {result.privacy_score:.2f}{dist_info}",
        tooltip=f"Meet here ({result.poi_type})",
        icon=folium.Icon(color="red", icon="heart"),
    ).add_to(m)

    # Polylines from users to meeting point
    folium.PolyLine(
        [(lat_a, lon_a), (result.lat, result.lon)],
        color="blue",
        weight=3,
        opacity=0.6,
        popup=f"A → POI: {result.distance_a_km:.2f} km" if result.distance_a_km else "A → POI",
    ).add_to(m)
    folium.PolyLine(
        [(lat_b, lon_b), (result.lat, result.lon)],
        color="green",
        weight=3,
        opacity=0.6,
        popup=f"B → POI: {result.distance_b_km:.2f} km" if result.distance_b_km else "B → POI",
    ).add_to(m)


    output_path = Path(output_path)
    m.save(str(output_path))
    return output_path


def main() -> int:
    """CLI: plot meeting point map for two user locations."""
    lat_a, lon_a = 40.6293, 22.9597  # Aristotle University
    lat_b, lon_b = 40.6261, 22.9484  # White Tower

    if len(sys.argv) >= 5:
        try:
            lat_a, lon_a = float(sys.argv[1]), float(sys.argv[2])
            lat_b, lon_b = float(sys.argv[3]), float(sys.argv[4])
        except ValueError:
            print("Usage: python plot_result.py [lat_a lon_a lat_b lon_b] [output.html]")
            return 1

    output = DEFAULT_OUTPUT
    if len(sys.argv) >= 6:
        output = Path(sys.argv[5])

    try:
        path = plot_meeting_point(lat_a, lon_a, lat_b, lon_b, output_path=output)
        print(f"Map saved to {path}")
        print("Open in a browser to view.")
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
