"""
Phase 0 step 3: zero-shot vision-LLM pilot for house style classification.

Takes a sample of addresses confirmed to have nearby Mapillary imagery
(from mapillary_coverage_sample.py), fetches each image's URL from
Mapillary's Entity API, and classifies it with two Claude models (Haiku for
cost, Sonnet for quality) against a fixed style taxonomy - so we can compare
cost and (via manual review) plausibility before committing to a full run.

No images are downloaded/stored except a small sample explicitly saved to
data/pilot_images/ for manual visual review (gitignored).

Output: data/vlm_pilot_results.parquet + printed cost/style summary.
"""

import base64
import os
import time
from enum import Enum

import pandas as pd
import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
COVERAGE_PATH = os.path.join(DATA_DIR, "mapillary_coverage_sample.parquet")
OUT_PATH = os.path.join(DATA_DIR, "vlm_pilot_results.parquet")
IMAGES_DIR = os.path.join(DATA_DIR, "pilot_images")

MAPILLARY_TOKEN = os.environ["MAPILLARY_TOKEN"]
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

N_SAMPLE = 40
N_SAVE_LOCALLY = 12  # small subset saved as files for manual visual review
MODELS = {
    "haiku": {"id": "claude-haiku-4-5", "in_price": 1.00, "out_price": 5.00},
    "sonnet": {"id": "claude-sonnet-5", "in_price": 2.00, "out_price": 10.00},
}


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


PROMPT = """You are looking at a single street-level photo that may or may not \
show the front of a single-family house in Seattle. Classify the architectural \
style of the house using ONLY the categories provided, if a house is clearly \
visible.

If the photo does not clearly show a house facade (e.g. it's a street view \
with no house, an obstructed/distant view, or a different type of building), \
set house_visible to false and use style "Uncertain/Mixed".

Be honest about uncertainty - use confidence "low" rather than guessing \
confidently when the style is ambiguous, heavily remodeled, or obscured by \
trees/cars/angle."""


def get_thumb_url(image_id) -> str | None:
    resp = requests.get(
        f"https://graph.mapillary.com/{image_id}",
        headers={"Authorization": f"OAuth {MAPILLARY_TOKEN}"},
        params={"fields": "thumb_1024_url"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("thumb_1024_url")


def classify(image_b64: str, model_id: str) -> tuple[StyleClassification | None, dict]:
    response = client.messages.parse(
        model=model_id,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        output_format=StyleClassification,
    )
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    return response.parsed_output, usage


def main() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    coverage = pd.read_parquet(COVERAGE_PATH)
    found = coverage[coverage["found"]].dropna(subset=["image_id"])
    sample = found.sample(n=min(N_SAMPLE, len(found)), random_state=7)
    print(f"Piloting on {len(sample)} addresses with confirmed nearby imagery.\n")

    results = []
    for i, (_, row) in enumerate(sample.iterrows()):
        image_id = int(row["image_id"])
        try:
            thumb_url = get_thumb_url(image_id)
        except Exception as e:
            print(f"  [{i}] image {image_id}: failed to fetch URL ({e})")
            continue
        if not thumb_url:
            continue

        try:
            img_bytes = requests.get(thumb_url, timeout=30).content
        except Exception as e:
            print(f"  [{i}] image {image_id}: failed to download ({e})")
            continue
        image_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

        if i < N_SAVE_LOCALLY:
            with open(os.path.join(IMAGES_DIR, f"{image_id}.jpg"), "wb") as f:
                f.write(img_bytes)

        for model_key, model_cfg in MODELS.items():
            try:
                parsed, usage = classify(image_b64, model_cfg["id"])
            except Exception as e:
                print(f"  [{i}] {model_key} failed: {e}")
                continue
            cost = (
                usage["input_tokens"] * model_cfg["in_price"]
                + usage["output_tokens"] * model_cfg["out_price"]
            ) / 1e6
            results.append(
                {
                    "address": row["address"],
                    "image_id": image_id,
                    "is_pano": row["is_pano"],
                    "model": model_key,
                    "house_visible": parsed.house_visible,
                    "style": parsed.style.value,
                    "confidence": parsed.confidence.value,
                    "rationale": parsed.rationale,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "cost_usd": cost,
                }
            )
        print(f"  [{i + 1}/{len(sample)}] {row['address']}: "
              f"{[r['style'] for r in results if r['image_id'] == image_id]}")
        time.sleep(0.1)

    df = pd.DataFrame(results)
    if df.empty:
        print("\nNo successful classifications - nothing to summarize.")
        return
    df.to_parquet(OUT_PATH, index=False)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for model_key in MODELS:
        sub = df[df["model"] == model_key]
        total_cost = sub["cost_usd"].sum()
        avg_cost = sub["cost_usd"].mean()
        print(f"\n{model_key} ({MODELS[model_key]['id']}):")
        print(f"  Images classified: {len(sub)}")
        print(f"  house_visible=True rate: {sub['house_visible'].mean() * 100:.1f}%")
        print(f"  Avg cost/image: ${avg_cost:.5f}  |  Total pilot cost: ${total_cost:.4f}")
        print(f"  Extrapolated cost @ 72,500 covered homes: ${avg_cost * 72500:.2f}")
        print(f"  Confidence distribution:\n{sub['confidence'].value_counts()}")
        print(f"  Style distribution:\n{sub['style'].value_counts()}")

    pivot = df.pivot(index="image_id", columns="model", values="style")
    agreement = (pivot["haiku"] == pivot["sonnet"]).mean()
    print(f"\nHaiku/Sonnet exact style agreement: {agreement * 100:.1f}%")

    print(f"\nWrote {OUT_PATH}")
    print(f"Saved {min(N_SAVE_LOCALLY, len(sample))} sample images to {IMAGES_DIR} for manual review")


if __name__ == "__main__":
    main()
