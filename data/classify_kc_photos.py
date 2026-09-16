"""
Classify architectural style for downloaded King County Assessor photos
(data/kc_photos/<pin>.jpg), using the same taxonomy, Pydantic schema, and
zero-shot prompt validated in the Phase 0 VLM pilot (vlm_pilot.py).

Supports both Haiku and Sonnet via --model, so results can be compared
directly (same style as the Phase 0 pilot's Mapillary comparison). The
earlier Mapillary pilot found Haiku hallucinated confidently on noisy
street-level photos (apartment buildings, empty alleys, underexposed
shots misread as houses); King County's Assessor photos are purpose-taken
shots of the actual property, a cleaner input, so it's worth re-testing
Haiku here rather than assuming the earlier disqualification carries over.

Output: data/kc_photos_classified.parquet (pin, model, style, confidence,
rationale, cost_usd, address, lat, lon, neighborhood). Appends to existing
results rather than overwriting - already-classified (pin, model) pairs are
skipped, so this is safe to re-run as more photos/models are added.

Usage:
    python classify_kc_photos.py --model haiku [--neighborhood Fremont]
"""

import argparse
import base64
import os
from enum import Enum

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_PATH = os.path.join(DATA_DIR, "kc_photos_status.parquet")
PARCELS_PATH = os.path.join(DATA_DIR, "parcels_master.parquet")
OUT_PATH = os.path.join(DATA_DIR, "kc_photos_classified.parquet")

MODELS = {
    "haiku": {"id": "claude-haiku-4-5", "in_price": 1.00, "out_price": 5.00},
    "sonnet": {"id": "claude-sonnet-5", "in_price": 2.00, "out_price": 10.00},
}

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


class HouseStyle(str, Enum):
    CRAFTSMAN = "Craftsman/Bungalow"
    FOURSQUARE = "American Foursquare"
    VICTORIAN = "Victorian/Queen Anne"
    TUDOR = "Tudor Revival"
    COLONIAL = "Colonial Revival"
    MINIMAL_TRADITIONAL = "Minimal Traditional"
    CAPE_COD = "Cape Cod"
    MID_CENTURY = "Mid-Century Modern"
    RANCH = "Ranch"
    SPLIT_LEVEL = "Split-Level"
    NW_CONTEMPORARY = "Northwest Contemporary"
    CONTEMPORARY = "Contemporary/Modern"
    SPANISH = "Spanish/Mediterranean Revival"
    UNCERTAIN = "Uncertain/Mixed"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class StyleClassification(BaseModel):
    house_visible: bool
    style: HouseStyle
    confidence: Confidence
    rationale: str


PROMPT = """You are looking at a King County Assessor's exterior photo of a \
single-family property in Seattle. Classify the architectural style of the \
house using ONLY the categories provided, if the house is clearly visible.

If the photo does not clearly show a house facade (e.g. obstructed by \
trees/vehicles, too distant, or a different type of structure), set \
house_visible to false and use style "Uncertain/Mixed".

Be honest about uncertainty - use confidence "low" rather than guessing \
confidently when the style is ambiguous, heavily remodeled, or obscured."""


def classify(image_path: str, model_id: str) -> tuple[StyleClassification, dict]:
    with open(image_path, "rb") as f:
        image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.parse(
        model=model_id,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        output_format=StyleClassification,
    )
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    return response.parsed_output, usage


def load_existing() -> pd.DataFrame:
    if os.path.exists(OUT_PATH):
        return pd.read_parquet(OUT_PATH)
    return pd.DataFrame(columns=["pin", "model"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default="sonnet")
    parser.add_argument("--neighborhood", default=None, help="Restrict to one neighborhood (optional)")
    args = parser.parse_args()
    model_cfg = MODELS[args.model]

    status = pd.read_parquet(STATUS_PATH)
    ok = status[status["status"] == "ok"]
    parcels = pd.read_parquet(PARCELS_PATH)
    ok = ok.merge(parcels[["pin", "address", "lat", "lon", "neighborhood"]], on="pin", how="left")

    if args.neighborhood:
        ok = ok[ok["neighborhood"] == args.neighborhood]

    existing = load_existing()
    already_done = set(zip(existing["pin"], existing.get("model", [])))
    todo = ok[~ok.apply(lambda r: (r["pin"], args.model) in already_done, axis=1)]

    print(f"Model: {model_cfg['id']} | Neighborhood filter: {args.neighborhood or 'none'}")
    print(f"Photos available: {len(ok)}, already classified with this model: {len(ok) - len(todo)}, "
          f"to classify this run: {len(todo)}")
    if todo.empty:
        print("Nothing to do.")
        return

    results = []
    for _, row in todo.iterrows():
        parsed, usage = classify(row["photo_path"], model_cfg["id"])
        cost = (usage["input_tokens"] * model_cfg["in_price"] + usage["output_tokens"] * model_cfg["out_price"]) / 1e6
        results.append(
            {
                "pin": row["pin"],
                "model": args.model,
                "house_visible": parsed.house_visible,
                "style": parsed.style.value,
                "confidence": parsed.confidence.value,
                "rationale": parsed.rationale,
                "cost_usd": cost,
                "address": row["address"],
                "lat": row["lat"],
                "lon": row["lon"],
                "neighborhood": row["neighborhood"],
            }
        )
        print(f"  {row['pin']} ({row['address']}): {parsed.style.value} ({parsed.confidence.value}) - ${cost:.5f}")

    new_df = pd.DataFrame(results)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_parquet(OUT_PATH, index=False)

    print(f"\nThis run cost: ${new_df['cost_usd'].sum():.5f}")
    print(f"Wrote {OUT_PATH} ({len(combined)} total classified rows)")


if __name__ == "__main__":
    main()
