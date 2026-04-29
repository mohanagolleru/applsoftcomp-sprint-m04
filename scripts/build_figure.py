#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers>=2.7.0",
#     "numpy>=1.24",
#     "pandas>=2.0",
#     "matplotlib>=3.7",
#     "adjustText>=1.0",
# ]
# ///
"""
Build the S&P 500 semantic map.

Two semantic axes (SemAxis: An, Kwak, Ahn 2018):
  Axis 1  Digital/Tech (+)  vs  Physical/Industrial (-)
  Axis 2  Consumer-facing (+)  vs  B2B/Enterprise (-)

Inputs : data/sp500.csv  (columns: name, sector)
Outputs: figures/semantic_map.png  (300 DPI)
         figures/semantic_map.pdf  (vector)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sp500.csv"
FIG_DIR = ROOT / "figures"

EMBED_MODEL = "all-MiniLM-L6-v2"  # 90 MB, fast, deterministic, good enough for SemAxis

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

# Okabe-Ito 8-color colorblind-safe palette (no green/red collision)
OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#999999",  # grey
    "#000000",  # black (extension)
    "#7B3F00",  # brown (extension)
    "#A6CEE3",  # light blue (extension)
]

# Coarse super-sector mapping for redundant shape encoding
SUPER_SECTOR = {
    "Information Technology": "IT",
    "Communication Services": "Services",
    "Financials": "Services",
    "Real Estate": "Services",
    "Utilities": "Services",
    "Health Care": "Services",
    "Consumer Discretionary": "Goods",
    "Consumer Staples": "Goods",
    "Materials": "Goods",
    "Energy": "Goods",
    "Industrials": "Goods",
}
SUPER_SHAPE = {"IT": "s", "Services": "o", "Goods": "^"}


def make_axis(positive_words, negative_words, model):
    """Unit-length semantic axis = normalized centroid difference (SemAxis)."""
    pos = model.encode(positive_words, normalize_embeddings=True)
    neg = model.encode(negative_words, normalize_embeddings=True)
    v = pos.mean(axis=0) - neg.mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-10)


def score_words(words, axis, model):
    """Project each word onto the axis (dot product)."""
    emb = model.encode(list(words), normalize_embeddings=True)
    return emb @ axis


def pole_separation(pos_words, neg_words, model):
    """Cosine distance between pole centroids (rule-of-thumb >= 0.3 = well-separated)."""
    p = model.encode(pos_words, normalize_embeddings=True).mean(axis=0)
    n = model.encode(neg_words, normalize_embeddings=True).mean(axis=0)
    p /= np.linalg.norm(p) + 1e-10
    n /= np.linalg.norm(n) + 1e-10
    return 1.0 - float(p @ n)


def axis_orthogonality(a1, a2):
    """|cos| between two axis vectors. 0 = perfectly orthogonal, 1 = parallel."""
    return abs(float(a1 @ a2))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA, dtype={"name": "string", "sector": "category"})
    print(f"Loaded {len(df)} firms across {df['sector'].nunique()} sectors.")

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("Building axes...")
    axis_tech = make_axis(AXIS1_POS, AXIS1_NEG, model)
    axis_consumer = make_axis(AXIS2_POS, AXIS2_NEG, model)

    sep1 = pole_separation(AXIS1_POS, AXIS1_NEG, model)
    sep2 = pole_separation(AXIS2_POS, AXIS2_NEG, model)
    ortho = axis_orthogonality(axis_tech, axis_consumer)
    print(f"  Axis 1 pole separation : {sep1:.3f}  (>=0.3 ok)")
    print(f"  Axis 2 pole separation : {sep2:.3f}  (>=0.3 ok)")
    print(f"  |cos(axis1, axis2)|    : {ortho:.3f}  (lower = more orthogonal)")

    print("Scoring firms...")
    x = score_words(df["name"].tolist(), axis_tech, model)
    y = score_words(df["name"].tolist(), axis_consumer, model)
    df = df.assign(x=x, y=y)
    df["super"] = df["sector"].astype(str).map(SUPER_SECTOR).fillna("Services")

    sectors = sorted(df["sector"].dropna().unique().tolist())
    color_for = {s: OKABE_ITO[i % len(OKABE_ITO)] for i, s in enumerate(sectors)}

    fig, ax = plt.subplots(figsize=(15, 9.5), dpi=300)

    ax.axhline(0, color="#888", lw=0.6, zorder=0)
    ax.axvline(0, color="#888", lw=0.6, zorder=0)

    for sector in sectors:
        sub = df[df["sector"] == sector]
        for super_label, marker in SUPER_SHAPE.items():
            sub2 = sub[sub["super"] == super_label]
            if len(sub2) == 0:
                continue
            ax.scatter(
                sub2["x"], sub2["y"],
                c=color_for[sector], marker=marker, s=55,
                edgecolor="white", linewidth=0.7, alpha=0.88,
                label=sector if super_label == sub["super"].iloc[0] else None,
                zorder=2,
            )

    # Label the 4 most-extreme outliers per quadrant by L1 distance from origin
    df["dist"] = np.abs(df["x"]) + np.abs(df["y"])
    labels = []
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        q = df[(np.sign(df["x"]) == sx) & (np.sign(df["y"]) == sy)].nlargest(4, "dist")
        for _, r in q.iterrows():
            labels.append(
                ax.text(r["x"], r["y"], r["name"],
                        fontsize=8.5, fontweight="medium", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.18",
                                  facecolor="white", edgecolor="none", alpha=0.78))
            )
    adjust_text(labels, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#444", lw=0.5),
                expand_points=(1.4, 1.6), expand_text=(1.2, 1.4))

    xpad = 0.03 * (df["x"].max() - df["x"].min())
    ypad = 0.03 * (df["y"].max() - df["y"].min())
    ax.set_xlim(df["x"].min() - xpad, df["x"].max() + xpad)
    ax.set_ylim(df["y"].min() - ypad, df["y"].max() + ypad)

    quad_props = dict(fontsize=10, color="#444", style="italic",
                      fontweight="bold", alpha=0.55, zorder=1)
    ax.text(0.985, 0.985, "TECH\n+ CONSUMER",       ha="right", va="top",    transform=ax.transAxes, **quad_props)
    ax.text(0.985, 0.015, "TECH\n+ ENTERPRISE",     ha="right", va="bottom", transform=ax.transAxes, **quad_props)
    ax.text(0.015, 0.985, "INDUSTRIAL\n+ CONSUMER", ha="left",  va="top",    transform=ax.transAxes, **quad_props)
    ax.text(0.015, 0.015, "INDUSTRIAL\n+ ENTERPRISE", ha="left", va="bottom", transform=ax.transAxes, **quad_props)

    ax.set_xlabel("Industrial / Physical    ←    Axis 1    →    Digital / Tech",
                  fontsize=11)
    ax.set_ylabel("B2B / Enterprise    ←    Axis 2    →    Consumer-facing",
                  fontsize=11)
    ax.set_title("S&P 500: a semantic map of company names\n"
                 f"(SemAxis on {EMBED_MODEL}; n={len(df)})",
                 fontsize=13, pad=12)

    from matplotlib.lines import Line2D
    handles, lbls = ax.get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, lbls) if not (l in seen or seen.add(l))]

    sector_legend = ax.legend(
        [h for h, _ in uniq], [l for _, l in uniq],
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=9, frameon=False,
        title="GICS Sector  (color)", title_fontsize=10,
    )
    ax.add_artist(sector_legend)

    shape_handles = [
        Line2D([0], [0], marker="s", color="w", label="IT",
               markerfacecolor="#666", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Services",
               markerfacecolor="#666", markersize=9),
        Line2D([0], [0], marker="^", color="w", label="Goods",
               markerfacecolor="#666", markersize=9),
    ]
    ax.legend(
        handles=shape_handles, loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        fontsize=9, frameon=False,
        title="Super-sector  (shape)", title_fontsize=10,
    )

    fig.subplots_adjust(left=0.05, right=0.72, top=0.93, bottom=0.07)
    out_png = FIG_DIR / "semantic_map.png"
    out_pdf = FIG_DIR / "semantic_map.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")

    df.drop(columns=["dist"]).to_csv(ROOT / "data" / "sp500_scored.csv", index=False)
    print(f"Saved {ROOT / 'data' / 'sp500_scored.csv'}")


if __name__ == "__main__":
    main()
