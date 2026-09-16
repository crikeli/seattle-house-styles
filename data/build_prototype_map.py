"""
Build a small, self-contained, LOCAL-ONLY prototype of the King County photo
classification pipeline, covering whatever homes have been fetched/classified
so far (see kc_photos_classified.parquet). Three tabs: Map (clickable markers
with photo + classification popups), Stats by Neighborhood (style tally bar
charts), About (methodology write-up). Images are embedded as base64 data
URIs so the output is a single file that opens directly in a browser (no
server, no external file references) - and, just as importantly, nothing
here gets published or pushed anywhere. This is a proof-of-concept for
reviewing locally while King County's permission request is pending, not a
production artifact.

Output: seattle-house-styles/kc_prototype_map.html (project root, not docs/,
so it's obviously separate from anything GitHub Pages would serve).
"""

import base64
import io
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFIED_PATH = os.path.join(PROJECT_ROOT, "data", "kc_photos_classified.parquet")
OUT_PATH = os.path.join(PROJECT_ROOT, "kc_prototype_map.html")
MAPTILER_KEY = os.environ["MAPTILER_KEY"]

# Palette (dataviz skill reference palette, same as used in the crime-hotspots project)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
BLUE = "#2a78d6"

# Fixed 8-hue categorical order (dataviz skill reference palette) - validated
# for adjacent-pair CVD/normal-vision separation. A map is effectively an
# "all-pairs" context (any two marker colors can end up adjacent as you pan/
# zoom), where the skill's guidance is to cap at 3 hues or fold the rest into
# "Other" - we go to 8 instead, justified by the secondary encoding already
# in place (every marker's exact style is always available via its popup,
# not color alone).
CATEGORICAL_HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_STYLE_COLOR = "#898781"   # a real style, just not common enough for its own hue
UNCLASSIFIED_COLOR = "#c3c2b7"  # not a style at all - the model declined to guess


def to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def assign_style_colors(df: pd.DataFrame) -> tuple[dict, list]:
    """Map each style to a marker color: the CATEGORICAL_HUES fixed order,
    assigned to the most frequent named styles in the current data (most
    common gets slot 1/blue, etc.) - "Uncertain/Mixed" is never assigned a
    hue slot since it isn't a style. Anything past the 8 hues falls back to
    OTHER_STYLE_COLOR. Returns (style -> color dict, list of styles folded
    into "Other")."""
    counts = df[df["style"] != "Uncertain/Mixed"]["style"].value_counts()
    top_styles = counts.head(len(CATEGORICAL_HUES)).index.tolist()
    other_styles = counts.index[len(CATEGORICAL_HUES):].tolist()

    colors = {style: hue for style, hue in zip(top_styles, CATEGORICAL_HUES)}
    for style in other_styles:
        colors[style] = OTHER_STYLE_COLOR
    colors["Uncertain/Mixed"] = UNCLASSIFIED_COLOR
    return colors, other_styles


def make_style_tally_chart(df: pd.DataFrame) -> str:
    """One horizontal bar panel per neighborhood, each with its own x-scale
    (sample sizes vary a lot - First Hill n=3 vs. 50 elsewhere - so a shared
    scale would make the small neighborhoods unreadable). A single hue is
    used throughout since bar position/label already carries style identity;
    no categorical color-coding is needed here."""
    neighborhoods = sorted(df["neighborhood"].dropna().unique())
    fig, axes = plt.subplots(
        len(neighborhoods), 1, figsize=(7, 1.9 * len(neighborhoods)), facecolor=SURFACE
    )
    if len(neighborhoods) == 1:
        axes = [axes]

    for ax, hood in zip(axes, neighborhoods):
        counts = df[df["neighborhood"] == hood]["style"].value_counts().sort_values()
        ax.barh(counts.index, counts.values, color=BLUE, height=0.6, zorder=3)
        ax.set_facecolor(SURFACE)
        ax.set_title(f"{hood} (n={counts.sum()})", loc="left", fontsize=10, color=INK_PRIMARY, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8, colors=INK_PRIMARY, length=0)
        ax.tick_params(axis="x", labelsize=7, colors=INK_MUTED)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(INK_MUTED)
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        max_count = counts.max()
        for y, v in enumerate(counts.values):
            ax.text(v + max_count * 0.02, y, str(v), va="center", fontsize=7.5, color=INK_PRIMARY)
        ax.set_xlim(0, max_count * 1.15)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def main() -> None:
    df = pd.read_parquet(CLASSIFIED_PATH)
    status = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "kc_photos_status.parquet"))
    df = df.merge(status[["pin", "photo_path"]], on="pin", how="left")

    # A few homes (the original 3-home pilot) have both a sonnet and a haiku
    # classification. The full audit trail stays in kc_photos_classified.parquet -
    # this only affects what's displayed, to avoid two overlapping markers at
    # the same address and to keep neighborhood counts at a clean n=50. Haiku
    # is preferred since it's the model the project has standardized on.
    dupes = df[df.duplicated(subset="pin", keep=False)]
    if not dupes.empty:
        print(f"Note: {dupes['pin'].nunique()} homes have classifications from multiple "
              f"models; displaying the haiku result for each (sonnet result still in "
              f"{os.path.basename(CLASSIFIED_PATH)}).")
    df["_model_rank"] = df["model"].map({"haiku": 0, "sonnet": 1}).fillna(2)
    df = df.sort_values("_model_rank").drop_duplicates(subset="pin", keep="first").drop(columns="_model_rank")

    style_colors, other_styles = assign_style_colors(df)
    n_own_hue = len(style_colors) - len(other_styles) - 1  # exclude "Other" bucket + Uncertain/Mixed
    print(f"Marker colors: {n_own_hue} styles get their own hue; "
          f"{len(other_styles)} minor styles folded into 'Other': {other_styles}")

    features_js = []
    for _, row in df.iterrows():
        img_uri = to_data_uri(os.path.join(PROJECT_ROOT, row["photo_path"]))
        features_js.append(
            f"""{{
  lat: {row['lat']}, lon: {row['lon']},
  address: {row['address']!r}, style: {row['style']!r}, model: {row['model']!r},
  confidence: {row['confidence']!r}, rationale: {row['rationale']!r},
  image: {img_uri!r}
}}"""
        )
    features_block = ",\n".join(features_js)

    center_lat, center_lon = df["lat"].mean(), df["lon"].mean()
    print(f"Embedding {len(df)} homes ({df['neighborhood'].nunique()} neighborhoods): "
          f"{df.groupby('neighborhood').size().to_dict()}")

    chart_uri = make_style_tally_chart(df)

    model_breakdown = ', '.join(f"{k}={v}" for k, v in df['model'].value_counts().items())
    style_colors_json = json.dumps(style_colors)
    other_styles_json = json.dumps(other_styles)
    other_style_color = OTHER_STYLE_COLOR
    unclassified_color = UNCLASSIFIED_COLOR

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>KC Photo Classification Prototype (LOCAL ONLY - not for publishing)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
  html, body {{ height: 100%; margin: 0; }}
  body {{
    display: flex; flex-direction: column;
    font-family: -apple-system, sans-serif; background: {SURFACE}; color: {INK_PRIMARY};
  }}
  #banner {{
    flex: 0 0 auto;
    background: #7a1f1f; color: white; padding: 10px 16px;
    font-size: 0.85rem; font-weight: bold;
  }}
  #tabs {{
    flex: 0 0 auto;
    display: flex; gap: 4px; padding: 10px 16px 0; border-bottom: 1px solid #e1e0d9;
    background: {SURFACE};
  }}
  .tab-btn {{
    border: none; background: none; padding: 10px 16px; font-size: 0.9rem;
    font-weight: 600; color: {INK_MUTED}; cursor: pointer; border-bottom: 3px solid transparent;
  }}
  .tab-btn.active {{ color: {INK_PRIMARY}; border-bottom-color: {BLUE}; }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  /* Only the Map tab fills the remaining viewport height (flex:1) - Stats
     and About keep natural content height with normal page scrolling. */
  #tab-map.active {{ flex: 1 1 auto; min-height: 0; }}
  #map {{ height: 100%; width: 100%; }}
  .popup-img {{ width: 220px; display: block; margin-bottom: 6px; border-radius: 4px; }}
  .popup-style {{ font-weight: bold; font-size: 1rem; }}
  .popup-meta {{ color: #666; font-size: 0.8rem; margin: 2px 0 6px; }}
  .popup-rationale {{ font-size: 0.78rem; color: #333; max-width: 220px; }}
  #charts {{ max-width: 800px; margin: 32px auto 60px; padding: 0 16px; }}
  #charts h2 {{ font-size: 1.1rem; margin-bottom: 4px; }}
  #charts p {{ color: {INK_MUTED}; font-size: 0.85rem; margin-top: 0; }}
  #charts img {{ width: 100%; height: auto; }}
  #about {{ max-width: 720px; margin: 0 auto; padding: 32px 16px 60px; line-height: 1.55; }}
  #about h2 {{ font-size: 1.3rem; margin: 0 0 12px; }}
  #about h3 {{ font-size: 1rem; margin: 28px 0 6px; }}
  #about p, #about li {{ font-size: 0.92rem; color: #2a2a28; }}
  #about ul, #about ol {{ padding-left: 20px; margin: 6px 0; }}
  #about code {{ background: #f0efe9; padding: 1px 5px; border-radius: 3px; font-size: 0.85em; }}
  #about a {{ color: {BLUE}; }}
  #about .callout {{
    background: #fff8e8; border-left: 3px solid #eda100; padding: 10px 14px;
    font-size: 0.85rem; margin: 16px 0; border-radius: 2px;
  }}
</style>
</head>
<body>
<div id="banner">LOCAL PROTOTYPE ONLY - not published, pending King County permission (see kc_permission_request_email.md)</div>
<div id="tabs">
  <button class="tab-btn active" data-tab="tab-map">Map</button>
  <button class="tab-btn" data-tab="tab-stats">Stats by Neighborhood</button>
  <button class="tab-btn" data-tab="tab-about">About</button>
</div>

<div id="tab-map" class="tab-panel active">
  <div id="map"></div>
</div>

<div id="tab-stats" class="tab-panel">
  <div id="charts">
    <h2>Architectural style tally by neighborhood</h2>
    <p>Zero-shot classification via Claude on King County Assessor photos. Model used per home: {model_breakdown}.</p>
    <img src="{chart_uri}" alt="Style tally bar charts by neighborhood" />
  </div>
</div>

<div id="tab-about" class="tab-panel">
  <div id="about">
    <h2>About this project</h2>
    <p>This app classifies the architectural style of single-family homes in Seattle
    (Craftsman, Tudor Revival, Colonial Revival, Mid-Century Modern, and similar
    categories) by running each home's exterior photo through a vision-capable
    Claude model, then plotting the results on a map. It's built with pandas,
    GeoPandas, and the Anthropic API, published as a static site with no backend.</p>

    <div class="callout">
      <strong>Current status:</strong> local prototype only, {len(df)} homes across
      {df['neighborhood'].nunique()} neighborhoods. Not yet published - see "Why King
      County photos" below for why.
    </div>

    <h3>Finding the homes</h3>
    <p>Home locations, addresses, and parcel IDs come from King County's own GIS
    open data - the
    <a href="https://gis-kingcounty.opendata.arcgis.com/datasets/kingcounty::parcels-with-address-property-and-ownership-information-public" target="_blank">Parcels with Address, Property and Ownership Information</a>
    feature service, filtered to Seattle's ~128,700 single-family parcels. Each
    home's neighborhood is assigned by spatially joining its coordinates against
    Seattle's official
    <a href="https://data-seattlecitygis.opendata.arcgis.com/datasets/SeattleCityGIS::community-reporting-areas-3/about" target="_blank">Community Reporting Area</a>
    boundaries.</p>

    <h3>Why King County Assessor photos, not Google Street View</h3>
    <p>Google's Street View terms of service explicitly prohibit bulk caching
    of imagery, displaying it outside Google's own map viewer, and using it as
    input to a machine learning model - all three of which this project needs.
    Mapillary (a crowdsourced, openly-licensed alternative) was tested first, but
    real usable coverage - a photo that actually shows the house, not an
    obstructed or oblique street frame - came out to roughly a quarter to a third
    of homes.</p>
    <p>King County's Assessor's Office photographs each property directly as
    part of its own records
    (<a href="https://blue.kingcounty.com/Assessor/eRealProperty/default.aspx" target="_blank">eRealProperty</a>).
    These photos are cleaner (purpose-shot, front-facing) and, in testing so
    far, available for nearly every parcel checked. However,
    <a href="https://www.kingcounty.gov/about/website/termsofuse" target="_blank">King County's site terms</a>
    require written permission before content can be published or
    redistributed - a request has been sent to the
    <a href="https://kingcounty.gov/en/dept/kcit/data-information-services/gis-center" target="_blank">King County GIS Center</a>
    and this app stays local-only until that's resolved.</p>

    <h3>Classifying each photo</h3>
    <p>Each photo is sent to a Claude vision model with a fixed prompt and a
    14-category style taxonomy (Craftsman/Bungalow, American Foursquare,
    Victorian/Queen Anne, Tudor Revival, Colonial Revival, Minimal Traditional,
    Cape Cod, Mid-Century Modern, Ranch, Split-Level, Northwest Contemporary,
    Contemporary/Modern, Spanish/Mediterranean Revival, plus "Uncertain/Mixed" for
    photos where no style can be confidently determined). The model returns a
    structured result: the style, a confidence level (high/medium/low), a short
    rationale, and whether a house was actually visible in the photo at all - it's
    explicitly instructed to decline rather than guess when the photo is unclear,
    obstructed, or doesn't show a house.</p>

    <h3>Why Haiku, not Sonnet</h3>
    <p>An early pilot on Mapillary's noisier street-level photos found the
    cheaper Haiku model would confidently mislabel non-house scenes - calling an
    apartment building "Craftsman," an empty alley "Minimal Traditional." Sonnet
    made no such errors in that same test but costs about 3x more per image.
    Because King County's Assessor photos are purpose-shot images of the actual
    house rather than noisy street frames, Haiku was re-tested on this cleaner
    source specifically - and performed reliably, with no scene-level
    hallucinations found in manual spot-checks. Haiku is used going forward for
    cost efficiency; Sonnet remains the fallback if quality issues turn up at
    larger scale.</p>

    <h3>Other approaches considered</h3>
    <ul>
      <li><strong>Satellite/aerial imagery</strong> - rejected as a primary
      source. Architectural style is mostly a facade property (porch columns,
      window trim, siding, roofline detail) that isn't visible from directly
      overhead.</li>
      <li><strong>A custom-trained image classifier</strong> - rejected in favor
      of zero-shot classification with an off-the-shelf vision model, since no
      usable labeled training dataset exists for ordinary houses (published
      academic datasets focus on landmark/monumental buildings, not tract
      housing).</li>
    </ul>

    <h3>Known limitations</h3>
    <ul>
      <li>Confidence levels are the model's own self-assessment, not verified
      against ground truth (e.g. historic preservation survey records).</li>
      <li>Classification is based on a single exterior photo - occlusion,
      lighting, remodeling, or genuinely blended styles can all affect the
      result.</li>
      <li>This is a small sample so far, gathered to validate the pipeline
      before any larger run.</li>
    </ul>

    <h3>Sources &amp; links</h3>
    <ul>
      <li><a href="https://gis-kingcounty.opendata.arcgis.com/datasets/kingcounty::parcels-with-address-property-and-ownership-information-public" target="_blank">King County parcel/address dataset</a> - home locations and IDs</li>
      <li><a href="https://data-seattlecitygis.opendata.arcgis.com/datasets/SeattleCityGIS::community-reporting-areas-3/about" target="_blank">Seattle Community Reporting Areas</a> - neighborhood boundaries</li>
      <li><a href="https://blue.kingcounty.com/Assessor/eRealProperty/default.aspx" target="_blank">King County eRealProperty</a> - property photo source</li>
      <li><a href="https://www.kingcounty.gov/about/website/termsofuse" target="_blank">King County website terms of use</a></li>
      <li><a href="https://kingcounty.gov/en/dept/kcit/data-information-services/gis-center" target="_blank">King County GIS Center</a> - contact for the pending permission request</li>
      <li><a href="https://www.anthropic.com/" target="_blank">Anthropic</a> - Claude vision models used for classification</li>
    </ul>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  const homes = [
{features_block}
  ];

  const map = L.map('map').setView([{center_lat}, {center_lon}], 12);
  L.tileLayer('https://api.maptiler.com/maps/streets-v4/256/{{z}}/{{x}}/{{y}}.png?key={MAPTILER_KEY}', {{
    attribution: '<a href="https://www.maptiler.com/copyright/" target="_blank">&copy; MapTiler</a> <a href="https://www.openstreetmap.org/copyright" target="_blank">&copy; OpenStreetMap contributors</a>',
    minZoom: 1,
    crossOrigin: true,
  }}).addTo(map);

  const STYLE_COLORS = {style_colors_json};
  const OTHER_STYLES = {other_styles_json};

  const markers = homes.map(h => {{
    const marker = L.circleMarker([h.lat, h.lon], {{
      radius: 8, fillColor: STYLE_COLORS[h.style] || '#999', color: '#fff', weight: 1.5, fillOpacity: 0.9,
    }}).addTo(map);
    marker.bindPopup(`
      <img class="popup-img" src="${{h.image}}" />
      <div class="popup-style">${{h.style}}</div>
      <div class="popup-meta">${{h.address}} &middot; ${{h.model}} &middot; confidence: ${{h.confidence}}</div>
      <div class="popup-rationale">${{h.rationale}}</div>
    `, {{ maxWidth: 260 }});
    return marker;
  }});

  if (markers.length > 1) {{
    map.fitBounds(L.featureGroup(markers).getBounds(), {{ padding: [30, 30] }});
  }}

  const legend = L.control({{ position: 'bottomright' }});
  legend.onAdd = function () {{
    const div = L.DomUtil.create('div');
    div.style.background = 'white';
    div.style.padding = '8px 10px';
    div.style.borderRadius = '4px';
    div.style.fontSize = '0.75rem';
    div.style.maxWidth = '190px';
    div.style.boxShadow = '0 1px 4px rgba(0,0,0,0.3)';
    div.style.maxHeight = '70vh';
    div.style.overflowY = 'auto';

    let rows = '<div style="font-weight:600;margin-bottom:4px;">Architectural style</div>';
    Object.entries(STYLE_COLORS).forEach(([style, color]) => {{
      if (style === 'Uncertain/Mixed' || OTHER_STYLES.includes(style)) return;
      rows += `<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">
        <span style="width:10px;height:10px;border-radius:50%;background:${{color}};flex:none;display:inline-block;"></span>
        <span>${{style}}</span>
      </div>`;
    }});
    if (OTHER_STYLES.length) {{
      rows += `<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">
        <span style="width:10px;height:10px;border-radius:50%;background:{other_style_color};flex:none;display:inline-block;"></span>
        <span>Other (${{OTHER_STYLES.join(', ')}})</span>
      </div>`;
    }}
    rows += `<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">
      <span style="width:10px;height:10px;border-radius:50%;background:{unclassified_color};flex:none;display:inline-block;"></span>
      <span>Uncertain / not classified</span>
    </div>`;
    div.innerHTML = rows;
    return div;
  }};
  legend.addTo(map);

  // Tab switching. The map is initialized above while its tab is visible, so
  // it gets a correct size on first paint - but Leaflet doesn't know if its
  // container is resized/hidden later, so invalidateSize() is called every
  // time we switch back to the Map tab (cheap, safe to call repeatedly).
  document.querySelectorAll('.tab-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
      if (btn.dataset.tab === 'tab-map') {{
        setTimeout(() => map.invalidateSize(), 0);
      }}
    }});
  }});
</script>
</body>
</html>
"""

    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"Wrote {OUT_PATH}")
    print("Open this file directly in a browser - it's fully self-contained (images embedded).")


if __name__ == "__main__":
    main()
