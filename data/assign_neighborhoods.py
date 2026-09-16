"""
One-time, fully local step: assign each single-family parcel a neighborhood
name by spatially joining parcels_master.parquet against Seattle's official
Community Reporting Areas (CRA) boundaries. No King County/Mapillary/Anthropic
traffic involved - this only enables batching the King County photo fetch by
neighborhood (see fetch_kc_photos.py).

Source: Seattle GIS open data, "Community Reporting Areas" (85 polygon parts,
53 unique neighborhoods after excluding water-body cutout parts).

Output: data/parcels_master.parquet gains a `neighborhood` column (added
in place). Parcels that don't fall inside any CRA polygon (e.g. right at the
water's edge) get neighborhood=None and are excluded from the fetch batches.
"""

import os

import geopandas as gpd
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PARCELS_PATH = os.path.join(DATA_DIR, "parcels_master.parquet")
CRA_PATH = os.path.join(DATA_DIR, "seattle_community_reporting_areas.geojson")

PROJECTED_CRS = "EPSG:32610"  # UTM zone 10N, meters - matches the crime-hotspots project


def main() -> None:
    parcels = pd.read_parquet(PARCELS_PATH)
    points = gpd.GeoDataFrame(
        parcels,
        geometry=gpd.points_from_xy(parcels["lon"], parcels["lat"]),
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)

    cra = gpd.read_file(CRA_PATH)
    cra = cra[cra["WATER"] == 0][["GEN_ALIAS", "geometry"]].rename(columns={"GEN_ALIAS": "neighborhood"})
    cra = cra.to_crs(PROJECTED_CRS)

    joined = gpd.sjoin(points, cra, how="left", predicate="within").drop(columns=["index_right"])
    # A handful of parcels may straddle two polygon parts of the same named
    # neighborhood (sjoin can duplicate rows in that case) - keep one.
    joined = joined.drop_duplicates(subset="pin")

    unmatched = joined["neighborhood"].isna().sum()
    print(f"Parcels: {len(joined)}, unmatched (no neighborhood): {unmatched} "
          f"({unmatched / len(joined) * 100:.1f}%)")
    print(f"\nHomes per neighborhood (top 10):")
    print(joined["neighborhood"].value_counts().head(10))
    print(f"\nHomes per neighborhood (bottom 10):")
    print(joined["neighborhood"].value_counts().tail(10))
    print(f"\nTotal neighborhoods with at least one home: {joined['neighborhood'].nunique()}")

    out = joined.drop(columns="geometry")
    out.to_parquet(PARCELS_PATH, index=False)
    print(f"\nWrote {PARCELS_PATH} (added 'neighborhood' column)")


if __name__ == "__main__":
    main()
