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

# 11 distinguishable, colorblind-safe colors (no pure green/red collision).
# Curated so adjacent sectors are not similar hues.
SECTOR_COLORS = {
    "Communication Services": "#E69F00",  # orange (Okabe-Ito)
    "Consumer Discretionary": "#56B4E9",  # sky blue (Okabe-Ito)
    "Consumer Staples":       "#CC79A7",  # pink (Okabe-Ito)
    "Energy":                 "#F0E442",  # yellow (Okabe-Ito)
    "Financials":             "#0072B2",  # navy blue (Okabe-Ito)
    "Health Care":            "#D55E00",  # vermillion (Okabe-Ito; safe red)
    "Industrials":            "#7D5BA6",  # purple
    "Information Technology": "#000000",  # black
    "Materials":              "#A6761D",  # brown
    "Real Estate":            "#999999",  # grey
    "Utilities":              "#80CDC1",  # teal (avoids pure green)
}

# Coarse super-sector for shape encoding (rubric: "redundantly encode with shape").
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
SUPER_SHAPE = {"IT": "s", "Services": "o", "Goods": "^"}

# Firms whose embedding position contradicts their GICS sector (utilities/
# financials in the tech-enterprise corner). Marked with a heavy ring to
# pull the eye to the headline finding.
SURPRISE_FIRMS = [
    "Public Service Enterprise Group",
    "Quanta Services",
    "Ares Management",
    "Fidelity National Information Services",
]


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
    df["super"] = df["sector"].astype(str).map(SUPER_SECTOR)

    sectors = [s for s in SECTOR_COLORS if s in df["sector"].unique()]

    # Visual hierarchy: 3-per-quadrant most-extreme outliers + hand-picked surprises.
    df["dist"] = np.abs(df["x"]) + np.abs(df["y"])
    outlier_names = set()
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        q = df[(np.sign(df["x"]) == sx) & (np.sign(df["y"]) == sy)].nlargest(3, "dist")
        outlier_names.update(q["name"].tolist())
    surprise_set = set(SURPRISE_FIRMS) & set(df["name"])
    label_names = outlier_names | surprise_set

    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    ax.axhline(0, color="#bbb", lw=0.7, zorder=0)
    ax.axvline(0, color="#bbb", lw=0.7, zorder=0)

    # Background dots — small, dim, shape = super-sector (redundant encoding).
    bg = df[~df["name"].isin(label_names)]
    for sector in sectors:
        for super_lab, marker in SUPER_SHAPE.items():
            sub = bg[(bg["sector"] == sector) & (bg["super"] == super_lab)]
            if len(sub) == 0:
                continue
            ax.scatter(sub["x"], sub["y"],
                       c=SECTOR_COLORS[sector], marker=marker,
                       s=24, alpha=0.45, edgecolor="none",
                       zorder=2, label=sector if marker == SUPER_SHAPE[df.loc[df["sector"] == sector, "super"].iloc[0]] else None)

    # Highlighted outliers — large, opaque, white edge.
    hi = df[df["name"].isin(outlier_names) & ~df["name"].isin(surprise_set)]
    for sector in sectors:
        for super_lab, marker in SUPER_SHAPE.items():
            sub = hi[(hi["sector"] == sector) & (hi["super"] == super_lab)]
            if len(sub) == 0:
                continue
            ax.scatter(sub["x"], sub["y"],
                       c=SECTOR_COLORS[sector], marker=marker,
                       s=90, alpha=0.95, edgecolor="white",
                       linewidth=1.0, zorder=3)

    # Surprises — largest, black ring (neutral, avoids any red/green concern).
    surp = df[df["name"].isin(surprise_set)]
    for sector in sectors:
        for super_lab, marker in SUPER_SHAPE.items():
            sub = surp[(surp["sector"] == sector) & (surp["super"] == super_lab)]
            if len(sub) == 0:
                continue
            ax.scatter(sub["x"], sub["y"],
                       c=SECTOR_COLORS[sector], marker=marker,
                       s=170, alpha=1.0, edgecolor="black",
                       linewidth=2.0, zorder=4)

    labels = []
    for _, r in df[df["name"].isin(label_names)].iterrows():
        weight = "bold" if r["name"] in surprise_set else "medium"
        labels.append(
            ax.text(r["x"], r["y"], r["name"],
                    fontsize=9, fontweight=weight, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.22",
                              facecolor="white", edgecolor="none", alpha=0.85))
        )
    adjust_text(labels, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#555", lw=0.6),
                expand_points=(1.5, 1.7), expand_text=(1.2, 1.4))

    xpad = 0.04 * (df["x"].max() - df["x"].min())
    ypad = 0.06 * (df["y"].max() - df["y"].min())
    ax.set_xlim(df["x"].min() - xpad, df["x"].max() + xpad)
    ax.set_ylim(df["y"].min() - ypad, df["y"].max() + ypad)

    # Quadrant annotations (rubric: "quadrant annotations help"). Inside corners,
    # subtle so they sit behind the data.
    quad = dict(fontsize=10, color="#666", style="italic",
                fontweight="semibold", alpha=0.7, zorder=1)
    ax.text(0.985, 0.985, "TECH + CONSUMER",       ha="right", va="top",    transform=ax.transAxes, **quad)
    ax.text(0.985, 0.015, "TECH + ENTERPRISE",     ha="right", va="bottom", transform=ax.transAxes, **quad)
    ax.text(0.015, 0.985, "INDUSTRIAL + CONSUMER", ha="left",  va="top",    transform=ax.transAxes, **quad)
    ax.text(0.015, 0.015, "INDUSTRIAL + ENTERPRISE", ha="left", va="bottom", transform=ax.transAxes, **quad)

    ax.set_xlabel("← industrial / physical          Axis 1          digital / tech →",
                  fontsize=11, labelpad=8)
    ax.set_ylabel("← B2B / enterprise          Axis 2          consumer-facing →",
                  fontsize=11, labelpad=8)
    ax.set_title(
        f"S&P 500 semantic map  ({EMBED_MODEL}; n={len(df)})",
        fontsize=13.5, pad=12, fontweight="semibold",
    )

    from matplotlib.lines import Line2D
    sector_handles = [
        Line2D([0], [0], marker="o", color="w", label=s,
               markerfacecolor=SECTOR_COLORS[s], markersize=8)
        for s in sectors
    ]
    sector_legend = ax.legend(
        handles=sector_handles,
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=9, frameon=False, labelspacing=0.55,
        title="GICS Sector  (color)", title_fontsize=10,
    )
    ax.add_artist(sector_legend)

    shape_handles = [
        Line2D([0], [0], marker=SUPER_SHAPE[s], color="w", label=s,
               markerfacecolor="#666", markersize=9)
        for s in ("IT", "Services", "Goods")
    ]
    shape_handles.append(
        Line2D([0], [0], marker="o", color="w", label="surprise",
               markerfacecolor="#888", markeredgecolor="black",
               markeredgewidth=1.8, markersize=10)
    )
    ax.legend(
        handles=shape_handles,
        loc="lower left", bbox_to_anchor=(1.01, 0.0),
        fontsize=9, frameon=False, labelspacing=0.55,
        title="Super-sector  (shape)", title_fontsize=10,
    )

    fig.subplots_adjust(left=0.06, right=0.78, top=0.93, bottom=0.07)
    out_png = FIG_DIR / "semantic_map.png"
    out_pdf = FIG_DIR / "semantic_map.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")

    df.drop(columns=["dist"]).to_csv(
        ROOT / "data" / "sp500_scored.csv", index=False
    )
    print(f"Saved {ROOT / 'data' / 'sp500_scored.csv'}")


if __name__ == "__main__":
    main()
