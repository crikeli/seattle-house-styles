"""
Fetch single-family residential parcels in Seattle from King County GIS's
"Parcels with Address, Property and Ownership Information (Public)" dataset
(ArcGIS FeatureServer, no auth required).

This one dataset provides address, lat/lon, and present-use code per parcel,
so no join against a separate Assessor extract is needed for this project's
scope (image-only style classification - we only need an address point per
home to look up street-level imagery, not structural attributes).

Filter: CTYNAME = 'SEATTLE' AND PREUSE_CODE = 2 ("Single Family (Res Use/Zone)").

Output: data/parcels_master.parquet (PIN, address, lat, lon, lot_sqft).
"""

import os
import time

import pandas as pd
import requests

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parcels_master.parquet")

FEATURE_SERVER = (
    "https://services.arcgis.com/Ej0PsM5Aw677QF1W/arcgis/rest/services/"
    "PARCEL_ADDRESS_PUB_AREA_3069/FeatureServer/0/query"
)
WHERE = "CTYNAME='SEATTLE' AND PREUSE_CODE=2 AND LAT IS NOT NULL AND LON IS NOT NULL"
OUT_FIELDS = "PIN,ADDR_FULL,LAT,LON,LOTSQFT"
PAGE_SIZE = 1000


def fetch_all() -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        params = {
            "where": WHERE,
            "outFields": OUT_FIELDS,
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
            "orderByFields": "PIN",
        }
        resp = requests.get(FEATURE_SERVER, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"])

        features = data.get("features", [])
        if not features:
            break
        rows.extend(f["attributes"] for f in features)
        offset += len(features)
        print(f"Fetched {offset} parcels so far...")
        if len(features) < PAGE_SIZE:
            break
        time.sleep(0.1)  # be polite to the free public service

    return pd.DataFrame(rows)


def main() -> None:
    df = fetch_all()
    df = df.rename(
        columns={"PIN": "pin", "ADDR_FULL": "address", "LAT": "lat", "LON": "lon", "LOTSQFT": "lot_sqft"}
    )
    df["address"] = df["address"].str.strip()
    df = df.drop_duplicates(subset="pin").reset_index(drop=True)

    print(f"\nTotal single-family Seattle parcels: {len(df)}")
    print(df.head())

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
