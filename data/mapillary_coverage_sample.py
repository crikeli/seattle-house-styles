"""
Phase 0 step 2: sample Mapillary street-level image coverage for a
geographically diverse set of Seattle single-family addresses.

Mapillary's v4 API only supports spatial search via vector tiles (no bbox
query on the Entity API - confirmed empirically, see conversation notes), so
this fetches the z14 tile covering each sampled address, decodes it with
mapbox_vector_tile, converts image-point pixel coordinates to lon/lat, and
finds the nearest image within COVERAGE_RADIUS_M meters.

Output: prints per-region and overall coverage %, and writes
data/mapillary_coverage_sample.parquet with per-address results for
inspection.
"""

import math
import os
import time

import mapbox_vector_tile
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PARCELS_PATH = os.path.join(DATA_DIR, "parcels_master.parquet")
OUT_PATH = os.path.join(DATA_DIR, "mapillary_coverage_sample.parquet")

TOKEN = os.environ["MAPILLARY_TOKEN"]
TILE_ZOOM = 14
TILE_EXTENT = 4096  # default MVT extent
COVERAGE_RADIUS_M = 40  # "usable" = an image within this distance of the address
N_REGIONS = 5
SAMPLES_PER_REGION = 60

_tile_cache: dict[tuple[int, int, int], list[dict]] = {}


def deg2num(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_pixel_to_lonlat(x: int, y: int, zoom: int, px: float, py: float, extent: int) -> tuple[float, float]:
    n = 2.0**zoom
    lon = (x + px / extent) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + py / extent) / n)))
    return lon, math.degrees(lat_rad)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_tile_images(x: int, y: int, zoom: int) -> list[dict]:
    key = (zoom, x, y)
    if key in _tile_cache:
        return _tile_cache[key]

    url = f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/{zoom}/{x}/{y}"
    resp = requests.get(url, params={"access_token": TOKEN}, timeout=30)
    resp.raise_for_status()
    tile = mapbox_vector_tile.decode(resp.content)

    points = []
    for feature in tile.get("image", {}).get("features", []):
        px, py = feature["geometry"]["coordinates"]
        lon, lat = tile_pixel_to_lonlat(x, y, zoom, px, py, TILE_EXTENT)
        props = feature["properties"]
        points.append(
            {
                "lon": lon,
                "lat": lat,
                "compass_angle": props.get("compass_angle"),
                "is_pano": props.get("is_pano"),
                "image_id": props.get("id"),
            }
        )
    _tile_cache[key] = points
    time.sleep(0.05)
    return points


def nearest_image(lat: float, lon: float) -> dict:
    x, y = deg2num(lat, lon, TILE_ZOOM)
    points = fetch_tile_images(x, y, TILE_ZOOM)
    if not points:
        return {"found": False, "distance_m": None, "is_pano": None, "image_id": None}

    dists = [haversine_m(lat, lon, p["lat"], p["lon"]) for p in points]
    idx = int(np.argmin(dists))
    best_dist = dists[idx]
    if best_dist > COVERAGE_RADIUS_M:
        return {"found": False, "distance_m": best_dist, "is_pano": None, "image_id": None}
    return {
        "found": True,
        "distance_m": best_dist,
        "is_pano": points[idx]["is_pano"],
        "image_id": points[idx]["image_id"],
    }


def pick_diverse_regions(df: pd.DataFrame, n_regions: int) -> list[pd.DataFrame]:
    """Split Seattle's bounding box into an n_regions x 1 grid along the
    longer axis (north-south, matching Seattle's shape) so sampled regions
    span the city rather than clustering downtown."""
    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    edges = np.linspace(lat_min, lat_max, n_regions + 1)
    regions = []
    for i in range(n_regions):
        mask = (df["lat"] >= edges[i]) & (df["lat"] < edges[i + 1])
        regions.append(df[mask])
    return regions


def main() -> None:
    parcels = pd.read_parquet(PARCELS_PATH)
    regions = pick_diverse_regions(parcels, N_REGIONS)

    all_results = []
    for i, region_df in enumerate(regions):
        sample = region_df.sample(n=min(SAMPLES_PER_REGION, len(region_df)), random_state=42 + i)
        print(f"\nRegion {i + 1}/{N_REGIONS} (lat {region_df['lat'].min():.4f}-{region_df['lat'].max():.4f}, n={len(sample)}):")

        region_results = []
        for _, row in sample.iterrows():
            result = nearest_image(row["lat"], row["lon"])
            result["pin"] = row["pin"]
            result["address"] = row["address"]
            result["lat"] = row["lat"]
            result["lon"] = row["lon"]
            result["region"] = i + 1
            region_results.append(result)

        region_df_out = pd.DataFrame(region_results)
        coverage_pct = region_df_out["found"].mean() * 100
        print(f"  Coverage: {coverage_pct:.1f}% ({region_df_out['found'].sum()}/{len(region_df_out)})")
        all_results.append(region_df_out)

    results = pd.concat(all_results, ignore_index=True)
    overall_pct = results["found"].mean() * 100
    print(f"\n{'=' * 50}")
    print(f"OVERALL COVERAGE: {overall_pct:.1f}% ({results['found'].sum()}/{len(results)})")
    print(f"{'=' * 50}")
    print("\nBy region:")
    print(results.groupby("region")["found"].agg(["mean", "count"]))

    found = results[results["found"]]
    if len(found):
        print(f"\nDistance to nearest image (found only): mean={found['distance_m'].mean():.1f}m, "
              f"median={found['distance_m'].median():.1f}m")
        print(f"Panorama share among found: {found['is_pano'].mean() * 100:.1f}%")

    results.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
