# Seattle House Styles

Classifies the architectural style of single-family homes in Seattle
(Craftsman, Tudor Revival, Colonial Revival, Mid-Century Modern, and similar
categories) from exterior property photos, using a zero-shot Claude vision
model, then maps the results.

## Status: local prototype, not yet published

**This project is blocked on a pending permission request to King County**
(see [`kc_permission_request_email.md`](kc_permission_request_email.md)).
King County's Assessor's Office photographs every property directly as part
of its own records, and those photos are the best imagery source found for
this project - but the county's site terms require written permission
before that content can be published or redistributed. Until that's
resolved:

- No King County photos are committed to this repo (`data/kc_photos/` is
  gitignored).
- The local prototype map (`kc_prototype_map.html`, which embeds photos
  directly as base64) is also gitignored and never published.
- `data/fetch_kc_photos.py` is deliberately not run at scale - see its
  docstring.

Everything else here - the pipeline code, the parcel/neighborhood data, and
the classification results (addresses + style labels, no photos) - is
public so the approach is documented and ready to scale up the moment
permission is granted.

## How it works

**Finding the homes.** Home locations, addresses, and parcel IDs come from
King County's own GIS open data - the
[Parcels with Address, Property and Ownership Information](https://gis-kingcounty.opendata.arcgis.com/datasets/kingcounty::parcels-with-address-property-and-ownership-information-public)
feature service, filtered to Seattle's ~128,700 single-family parcels
(`data/fetch_parcels.py`). Each home's neighborhood is assigned by spatially
joining its coordinates against Seattle's official
[Community Reporting Area](https://data-seattlecitygis.opendata.arcgis.com/datasets/SeattleCityGIS::community-reporting-areas-3/about)
boundaries (`data/assign_neighborhoods.py`).

**Why King County Assessor photos, not Google Street View.** Google's
Street View terms of service explicitly prohibit bulk caching of imagery,
displaying it outside Google's own map viewer, and using it as input to a
machine learning model - all three of which this project needs. Mapillary
(a crowdsourced, openly-licensed alternative) was tested first
(`data/mapillary_coverage_sample.py`, `data/vlm_pilot.py`), but real usable
coverage - a photo that actually shows the house, not an obstructed or
oblique street frame - came out to roughly a quarter to a third of homes.

King County's [eRealProperty](https://blue.kingcounty.com/Assessor/eRealProperty/default.aspx)
photos are cleaner (purpose-shot, front-facing) and, in testing so far,
available for nearly every parcel checked - but require the permission
described above.

**Classifying each photo.** Each photo is sent to a Claude vision model
(`data/classify_kc_photos.py`) with a fixed prompt and a 14-category style
taxonomy (Craftsman/Bungalow, American Foursquare, Victorian/Queen Anne,
Tudor Revival, Colonial Revival, Minimal Traditional, Cape Cod, Mid-Century
Modern, Ranch, Split-Level, Northwest Contemporary, Contemporary/Modern,
Spanish/Mediterranean Revival, plus "Uncertain/Mixed" for photos where no
style can be confidently determined). The model returns a structured
result - style, confidence (high/medium/low), a short rationale, and
whether a house was actually visible - and is explicitly instructed to
decline rather than guess when the photo is unclear or doesn't show a
house.

**Why Haiku, not Sonnet.** An early pilot on Mapillary's noisier
street-level photos found the cheaper Haiku model would confidently
mislabel non-house scenes - calling an apartment building "Craftsman," an
empty alley "Minimal Traditional." Sonnet made no such errors in that same
test but costs about 3x more per image. Because King County's Assessor
photos are purpose-shot images of the actual house rather than noisy street
frames, Haiku was re-tested on this cleaner source specifically - and
performed reliably, with no scene-level hallucinations found in manual
spot-checks. Haiku is used going forward for cost efficiency; Sonnet
remains the fallback if quality issues turn up at larger scale.

**Other approaches considered:**
- *Satellite/aerial imagery* - rejected as a primary source. Architectural
  style is mostly a facade property (porch columns, window trim, siding,
  roofline detail) that isn't visible from directly overhead.
- *A custom-trained image classifier* - rejected in favor of zero-shot
  classification with an off-the-shelf vision model, since no usable
  labeled training dataset exists for ordinary houses (published academic
  datasets focus on landmark/monumental buildings, not tract housing).

## Known limitations

- Confidence levels are the model's own self-assessment, not verified
  against ground truth (e.g. historic preservation survey records).
- Classification is based on a single exterior photo - occlusion, lighting,
  remodeling, or genuinely blended styles can all affect the result.
- This is a small validation sample (300 homes across 6 neighborhoods as of
  writing), gathered to test the pipeline before any larger run.

## Repo layout

```
data/
  fetch_parcels.py            # pulls all Seattle single-family parcels (King County GIS)
  assign_neighborhoods.py     # spatial join -> neighborhood column (one-time, local)
  fetch_kc_photos.py          # throttled, resumable KC Assessor photo fetch (not run at scale yet)
  classify_kc_photos.py       # zero-shot Claude classification
  build_prototype_map.py      # builds the local prototype (gitignored output)
  mapillary_coverage_sample.py, vlm_pilot.py, generate_review_pdf.py  # Phase 0 (Mapillary) - superseded, kept for history
  parcels_master.parquet      # gitignored, regenerate via fetch_parcels.py
  kc_photos_status.parquet    # fetch status per parcel (no image bytes)
  kc_photos_classified.parquet # style/confidence/rationale per parcel (no image bytes)
kc_permission_request_email.md  # the exact request sent to King County GIS Center
environment.yml
```

## Setup

```bash
conda env create -f environment.yml
conda activate seattle-house-styles
cp .env.example .env  # fill in MAPILLARY_TOKEN, ANTHROPIC_API_KEY, MAPTILER_KEY
```

```bash
python data/fetch_parcels.py
python data/assign_neighborhoods.py
python data/classify_kc_photos.py --model haiku --neighborhood "Fremont"
python data/build_prototype_map.py  # local-only output, see Status above
```

`fetch_kc_photos.py` requires King County's written permission before
running beyond a small validation sample - see the Status section.

## Stack

pandas, GeoPandas, Anthropic API (Claude vision), matplotlib, Leaflet +
MapTiler for the local prototype map.
