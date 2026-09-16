"""
One-off Phase 0 review artifact: an 8-example PDF (4 plausible, 4 implausible
classifications) selected by manually comparing the pilot's saved sample
images against what each model claimed. Not part of the production
pipeline - just a visual aid for the Phase 0 decision gate.
"""

import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(DATA_DIR, "pilot_images")
OUT_PATH = os.path.join(DATA_DIR, "phase0_classification_review.pdf")

# (image_id, address, verdict, note)
CORRECT = [
    (
        "960533818528487", "4421 2ND AVE NW",
        "Both models: 'Uncertain/Mixed', house_visible=False",
        "Image is a distorted 360 panorama of a roadway - correctly declined rather than guessing.",
    ),
    (
        "169961345129897", "506 W GALER ST",
        "Both models: 'Uncertain/Mixed', house_visible=False",
        "Image shows multi-story condo/apartment buildings, not single-family homes - correctly declined.",
    ),
    (
        "488096605722586", "1805 NW BLUE RIDGE DR",
        "Haiku: 'Mid-Century Modern' (med) | Sonnet: 'Ranch' (low)",
        "A real brick, low-slung house IS visible on the right - both labels are plausible for that facade.",
    ),
    (
        "1040407689816330", "2134 N 88TH ST",
        "Both models: 'Uncertain/Mixed', house_visible=False",
        "Extreme close-up, blurry road shot - no house in frame at all. Correctly declined.",
    ),
]

INCORRECT = [
    (
        "1297117030844822", "5911 59TH AVE NE",
        "Haiku: 'Craftsman/Bungalow' (medium confidence)",
        "WRONG: image shows multi-story apartment/condo buildings, not a single-family Craftsman house.",
    ),
    (
        "303954357987647", "1712 26TH AVE",
        "Haiku: 'Minimal Traditional' (low confidence)",
        "WRONG: image is a back alley with wooden fences - no house facade is visible in the frame at all.",
    ),
    (
        "807927703432267", "7025 4TH AVE NW",
        "Haiku: 'Minimal Traditional' (medium confidence)",
        "WRONG: image is dark/underexposed - no architectural detail is actually discernible.",
    ),
    (
        "167833455160361", "1154 18TH AVE E",
        "Haiku: 'Northwest Contemporary' (medium confidence)",
        "WRONG STYLE: a real stone house IS visible, but it reads as a steep-roofed Tudor Revival, not "
        "Northwest Contemporary. Sonnet said 'Tudor Revival' on the same image.",
    ),
]


def render_page(pdf: PdfPages, title: str, examples: list[tuple[str, str, str, str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 12))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    for ax, (image_id, address, verdict, note) in zip(axes.flat, examples):
        img_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(address, fontsize=11, fontweight="bold", loc="left")
        caption = f"{verdict}\n\n{note}"
        ax.text(
            0.5, -0.05, caption, transform=ax.transAxes,
            fontsize=8.5, ha="center", va="top", wrap=True,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    with PdfPages(OUT_PATH) as pdf:
        render_page(pdf, "Phase 0 Pilot Review: Plausible Classifications (4 of 4)", CORRECT)
        render_page(pdf, "Phase 0 Pilot Review: Implausible Classifications (4 of 4, all Haiku)", INCORRECT)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
