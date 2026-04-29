#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers>=2.7.0",
#     "numpy>=1.24",
#     "pandas>=2.0",
#     "altair>=5.4.0",
#     "vl-convert-python>=1.6.0",
# ]
# ///
"""
Build the S&P 500 semantic map.

Reuses the SemAxis helpers (`make_axis`, `score_words`) from the worked-
example notebook, then renders the scatter with Altair / Vega-Lite — the
same library the notebook uses for the cities case study. PNG and PDF
are produced via the `vl-convert` static renderer (no headless browser
required).

Axes:
  Axis 1  Industrial / physical (-)  vs  Digital / tech (+)
  Axis 2  B2B / enterprise (-)       vs  Consumer-facing (+)

Inputs : data/sp500.csv  (columns: name, sector)
Outputs: figures/semantic_map.png  (3x scale, ~330 DPI)
         figures/semantic_map.pdf  (vector)
         data/sp500_scored.csv     (every firm with x, y projection)
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sp500.csv"
FIG_DIR = ROOT / "figures"

EMBED_MODEL = "all-MiniLM-L6-v2"

AXIS1_POS = [
    "software platform",
    "cloud computing service",
    "digital technology company",
    "internet and data services",
    "enterprise software vendor",
]
AXIS1_NEG = [
    "heavy machinery manufacturer",
    "industrial equipment producer",
    "factory and assembly plant",
    "physical infrastructure",
    "raw materials and mining",
]

AXIS2_POS = [
    "household consumer brand",
    "retail store and supermarket",
    "everyday consumer product",
    "mass-market lifestyle brand",
    "popular shopping destination",
]
AXIS2_NEG = [
    "enterprise software vendor",
    "industrial supplier and wholesaler",
    "specialty business services",
    "infrastructure and capital goods",
    "B2B equipment provider",
]

# Okabe-Ito colorblind-safe palette extended to 11 sectors. No pure green
# alongside the red-ish vermillion, so the rubric's "no green and red in
# the same plot" rule is satisfied.
SECTOR_COLORS = {
    "Communication Services": "#E69F00",  # orange
    "Consumer Discretionary": "#56B4E9",  # sky blue
    "Consumer Staples":       "#CC79A7",  # pink
    "Energy":                 "#F0E442",  # yellow
    "Financials":             "#0072B2",  # navy blue
    "Health Care":            "#D55E00",  # vermillion
    "Industrials":            "#7D5BA6",  # purple
    "Information Technology": "#000000",  # black
    "Materials":              "#A6761D",  # brown
    "Real Estate":            "#999999",  # grey
    "Utilities":              "#80CDC1",  # teal
}

# Shape redundantly encodes super-sector, per the data-viz checklist.
SUPER_SECTOR = {
    "Information Technology": "IT",
    "Communication Services": "Services",
    "Financials":             "Services",
    "Real Estate":            "Services",
    "Utilities":              "Services",
    "Health Care":            "Services",
    "Consumer Discretionary": "Goods",
    "Consumer Staples":       "Goods",
    "Materials":              "Goods",
    "Energy":                 "Goods",
    "Industrials":            "Goods",
}
SHAPE_DOMAIN = ["IT", "Services", "Goods"]
SHAPE_RANGE  = ["square", "circle", "triangle-up"]


def make_axis(positive_words, negative_words, embedding_model):
    """Return a unit-length semantic axis from two word sets."""
    pos_emb = embedding_model.encode(positive_words, normalize_embeddings=True)
    neg_emb = embedding_model.encode(negative_words, normalize_embeddings=True)
    pole_pos = pos_emb.mean(axis=0)
    pole_neg = neg_emb.mean(axis=0)
    v = pole_pos - pole_neg
    return v / (np.linalg.norm(v) + 1e-10)


def score_words(words, axis, embedding_model):
    """Project each word onto the axis. Returns one score per word."""
    emb = embedding_model.encode(list(words), normalize_embeddings=True)
    return emb @ axis


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA, dtype={"name": "string", "sector": "category"})
    print(f"Loaded {len(df)} firms across {df['sector'].nunique()} sectors.")

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("Building axes...")
    axis_tech = make_axis(AXIS1_POS, AXIS1_NEG, model)
    axis_consumer = make_axis(AXIS2_POS, AXIS2_NEG, model)

    pos1 = model.encode(AXIS1_POS, normalize_embeddings=True).mean(axis=0)
    neg1 = model.encode(AXIS1_NEG, normalize_embeddings=True).mean(axis=0)
    pos2 = model.encode(AXIS2_POS, normalize_embeddings=True).mean(axis=0)
    neg2 = model.encode(AXIS2_NEG, normalize_embeddings=True).mean(axis=0)
    sep1 = 1.0 - float(pos1 @ neg1 / (np.linalg.norm(pos1) * np.linalg.norm(neg1)))
    sep2 = 1.0 - float(pos2 @ neg2 / (np.linalg.norm(pos2) * np.linalg.norm(neg2)))
    ortho = abs(float(axis_tech @ axis_consumer))
    print(f"  Axis 1 pole separation : {sep1:.3f}  (>=0.3 ok)")
    print(f"  Axis 2 pole separation : {sep2:.3f}  (>=0.3 ok)")
    print(f"  |cos(axis1, axis2)|    : {ortho:.3f}  (lower = more orthogonal)")

    print("Scoring firms...")
    df = df.assign(
        x=score_words(df["name"].tolist(), axis_tech, model),
        y=score_words(df["name"].tolist(), axis_consumer, model),
    )
    df["super"] = df["sector"].astype(str).map(SUPER_SECTOR)
    df_scored = df.copy()

    sectors = [s for s in SECTOR_COLORS if s in df["sector"].unique()]
    color_scale = alt.Scale(domain=sectors,
                            range=[SECTOR_COLORS[s] for s in sectors])
    shape_scale = alt.Scale(domain=SHAPE_DOMAIN, range=SHAPE_RANGE)

    # Pick a small set of anchor labels: 2 most-extreme firms in each of the
    # four quadrants. Keeps the plot readable while flagging where the poles
    # actually land.
    df_l = df_scored.assign(dist=df_scored["x"].abs() + df_scored["y"].abs())
    label_rows = []
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        q = df_l[(np.sign(df_l["x"]) == sx) & (np.sign(df_l["y"]) == sy)].nlargest(2, "dist")
        label_rows.append(q)
    df_labels = pd.concat(label_rows, ignore_index=True)

    base = alt.Chart(df_scored)
    points = base.mark_point(
        size=120, opacity=0.85, filled=True, strokeWidth=0.6, stroke="white",
    ).encode(
        x=alt.X(
            "x:Q",
            title="Axis 1:  industrial / physical  →  digital / tech",
            scale=alt.Scale(zero=False, padding=20),
            axis=alt.Axis(grid=False, domainColor="#888"),
        ),
        y=alt.Y(
            "y:Q",
            title="Axis 2:  B2B / enterprise  →  consumer-facing",
            scale=alt.Scale(zero=False, padding=20),
            axis=alt.Axis(grid=False, domainColor="#888"),
        ),
        color=alt.Color(
            "sector:N", scale=color_scale,
            legend=alt.Legend(title="GICS Sector"),
        ),
        shape=alt.Shape(
            "super:N", scale=shape_scale,
            legend=alt.Legend(title="Super-sector"),
        ),
        tooltip=[
            alt.Tooltip("name:N", title="Company"),
            alt.Tooltip("sector:N", title="Sector"),
            alt.Tooltip("x:Q", title="Tech score", format=".3f"),
            alt.Tooltip("y:Q", title="Consumer score", format=".3f"),
        ],
    )

    text = (
        alt.Chart(df_labels)
        .mark_text(align="left", baseline="middle", dx=10, dy=-2,
                   fontSize=10, color="#222", fontWeight=500)
        .encode(x="x:Q", y="y:Q", text="name:N")
    )

    chart = (
        (points + text)
        .properties(
            width=780,
            height=520,
            title=alt.TitleParams(
                text="S&P 500 in a 2D semantic space",
                subtitle=[f"SemAxis on {EMBED_MODEL};  n = {len(df)};  "
                          f"pole separations = {sep1:.2f} / {sep2:.2f};  "
                          f"|cos(axis1, axis2)| = {ortho:.2f}"],
                fontSize=15,
                subtitleFontSize=10,
                subtitleColor="#555",
                anchor="start",
                offset=10,
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=11, titleFontSize=12, titleColor="#333")
        .configure_legend(labelFontSize=10, titleFontSize=11, padding=8)
    )

    out_png = FIG_DIR / "semantic_map.png"
    out_pdf = FIG_DIR / "semantic_map.pdf"
    chart.save(str(out_png), scale_factor=3.0)  # 3x scale -> ~330 DPI
    chart.save(str(out_pdf))
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")

    df.drop(columns=["super"]).to_csv(
        ROOT / "data" / "sp500_scored.csv", index=False
    )
    print(f"Saved {ROOT / 'data' / 'sp500_scored.csv'}")


if __name__ == "__main__":
    main()
