"""
Fetch King County Assessor property photos (eRealProperty MediaHandler), one
neighborhood at a time, politely throttled.

STATUS: NOT for production use yet. King County's site-wide terms of use
prohibit publishing/displaying/distributing this content without prior
written permission - a request has been sent but not yet answered (see
project notes). This script exists so the fetch mechanics are built and
tested against a tiny sample now, ready to run for real the moment
permission is confirmed. Do not run this at full neighborhood/city scale
until then.

Mechanics:
- For each PIN in the target neighborhood: GET Dashboard.aspx?ParcelNbr=<PIN>,
  parse out the MediaHandler.aspx?Media=<id> reference, then GET that image.
- REQUEST_DELAY_SECONDS between every request (two per home: the dashboard
  page, then the image) keeps this to roughly one request/second - about
  40 minutes of traffic for an average ~2,340-home neighborhood, once a day.
- Checkpointed: writes one row per PIN to data/kc_photos_status.parquet as it
  goes (found/not-found/error + local file path), so a run can be stopped
  and resumed without re-fetching PINs already recorded.

Usage:
    python fetch_kc_photos.py --neighborhood "Ballard" [--limit 5]
"""

import argparse
import os
import re
import time

import pandas as pd
import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PARCELS_PATH = os.path.join(DATA_DIR, "parcels_master.parquet")
STATUS_PATH = os.path.join(DATA_DIR, "kc_photos_status.parquet")
PHOTOS_DIR = os.path.join(DATA_DIR, "kc_photos")

BASE_URL = "https://blue.kingcounty.com/Assessor/eRealProperty"
HEADERS = {"User-Agent": "Mozilla/5.0"}
REQUEST_DELAY_SECONDS = 1.0
MEDIA_ID_PATTERN = re.compile(r"MediaHandler\.aspx\?Media=(\d+)")


def load_status() -> pd.DataFrame:
    if os.path.exists(STATUS_PATH):
        return pd.read_parquet(STATUS_PATH)
    return pd.DataFrame(columns=["pin", "status", "media_id", "photo_path"])


def save_status(df: pd.DataFrame) -> None:
    df.to_parquet(STATUS_PATH, index=False)


def fetch_one(pin: str) -> dict:
    dashboard_url = f"{BASE_URL}/Dashboard.aspx"
    resp = requests.get(dashboard_url, headers=HEADERS, params={"ParcelNbr": pin}, timeout=30)
    time.sleep(REQUEST_DELAY_SECONDS)
    if resp.status_code != 200:
        return {"pin": pin, "status": f"dashboard_http_{resp.status_code}", "media_id": None, "photo_path": None}

    match = MEDIA_ID_PATTERN.search(resp.text)
    if not match:
        return {"pin": pin, "status": "no_photo_found", "media_id": None, "photo_path": None}
    media_id = match.group(1)

    photo_resp = requests.get(f"{BASE_URL}/MediaHandler.aspx", headers=HEADERS, params={"Media": media_id}, timeout=30)
    time.sleep(REQUEST_DELAY_SECONDS)
    if photo_resp.status_code != 200 or not photo_resp.content:
        return {"pin": pin, "status": f"photo_http_{photo_resp.status_code}", "media_id": media_id, "photo_path": None}

    os.makedirs(PHOTOS_DIR, exist_ok=True)
    abs_photo_path = os.path.join(PHOTOS_DIR, f"{pin}.jpg")
    with open(abs_photo_path, "wb") as f:
        f.write(photo_resp.content)

    # Stored relative to the project root (not absolute) so this file stays
    # portable across machines - this parquet is committed to the repo (it
    # has no image bytes, just fetch bookkeeping), unlike the photos
    # themselves.
    relative_photo_path = os.path.join("data", "kc_photos", f"{pin}.jpg")
    return {"pin": pin, "status": "ok", "media_id": media_id, "photo_path": relative_photo_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neighborhood", required=True, help="Exact neighborhood name, e.g. 'Ballard'")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of homes fetched (for testing)")
    args = parser.parse_args()

    parcels = pd.read_parquet(PARCELS_PATH)
    target = parcels[parcels["neighborhood"] == args.neighborhood]
    if target.empty:
        available = sorted(parcels["neighborhood"].dropna().unique())
        raise SystemExit(f"No homes found for neighborhood '{args.neighborhood}'. Available: {available}")

    status = load_status()
    already_done = set(status["pin"])
    already_done_in_target = already_done & set(target["pin"])
    todo = target[~target["pin"].isin(already_done)]
    if args.limit:
        todo = todo.head(args.limit)

    print(f"Neighborhood: {args.neighborhood}")
    print(f"Total homes: {len(target)}, already fetched: {len(already_done_in_target)}, "
          f"to fetch this run: {len(todo)}")
    if todo.empty:
        print("Nothing to do.")
        return

    new_rows = []
    for i, (_, row) in enumerate(todo.iterrows()):
        result = fetch_one(row["pin"])
        new_rows.append(result)
        print(f"  [{i + 1}/{len(todo)}] {row['address']} (PIN {row['pin']}): {result['status']}")

        if (i + 1) % 20 == 0:
            status = pd.concat([status, pd.DataFrame(new_rows)], ignore_index=True)
            save_status(status)
            new_rows = []

    if new_rows:
        status = pd.concat([status, pd.DataFrame(new_rows)], ignore_index=True)
        save_status(status)

    print(f"\nWrote status to {STATUS_PATH}")
    ok_count = (status[status["pin"].isin(todo["pin"])]["status"] == "ok").sum()
    print(f"This run: {ok_count}/{len(todo)} photos fetched successfully.")


if __name__ == "__main__":
    main()
