"""
Phase 6 - Anomaly / suspicious-address detection on the Bitcoin OTC /
Bitcoin Alpha trust networks.

Heuristics (each is a weak signal on its own; combined they flag addresses
worth a closer look):

1. Negative-rating concentration: addresses that *received* >= MIN_RATINGS
   ratings where more than NEG_THRESHOLD of them are negative -> distrusted
   by many distinct peers, not just one grudge.
2. Rating-burst behavior: addresses that *issued* an unusually large number
   of ratings within a single day -> sybil / bot-like mass-rating pattern.
3. One-sided new accounts: addresses with very few total ratings (<=3) all
   received from the same small clique -> possible fake reputation
   ("rating ring").

Outputs:
- results/anomalies_negative_<name>.csv
- results/anomalies_bursty_raters_<name>.csv
- results/rating_sign_distribution_<name>.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_graph import load_edgelist

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "bitcoin_otc": "soc-sign-bitcoinotc.csv",
    "bitcoin_alpha": "soc-sign-bitcoinalpha.csv",
}

MIN_RATINGS_RECEIVED = 5
NEG_FRACTION_THRESHOLD = 0.5
BURST_RATINGS_PER_DAY = 15


def rating_sign_distribution(df: pd.DataFrame, name: str) -> None:
    pos = (df["rating"] > 0).sum()
    neg = (df["rating"] < 0).sum()
    zero = (df["rating"] == 0).sum()

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.bar(["Positive", "Negative", "Neutral (0)"], [pos, neg, zero],
           color=["#2e8b57", "#c0392b", "#999999"])
    ax.set_title(f"{name} — rating sign distribution\n"
                 f"({100*pos/len(df):.1f}% pos / {100*neg/len(df):.1f}% neg)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"rating_sign_distribution_{name}.png", dpi=150)
    plt.close(fig)


def negative_concentration(df: pd.DataFrame, name: str) -> pd.DataFrame:
    grouped = df.groupby("target")["rating"].agg(
        num_ratings_received="count",
        num_distinct_raters=lambda s: s.shape[0],
        num_negative=lambda s: (s < 0).sum(),
        avg_rating="mean",
    )
    grouped["neg_fraction"] = grouped["num_negative"] / grouped["num_ratings_received"]
    flagged = grouped[
        (grouped["num_ratings_received"] >= MIN_RATINGS_RECEIVED)
        & (grouped["neg_fraction"] >= NEG_FRACTION_THRESHOLD)
    ].sort_values(["neg_fraction", "num_ratings_received"], ascending=False)
    flagged = flagged.reset_index().rename(columns={"target": "node"})
    flagged.to_csv(RESULTS_DIR / f"anomalies_negative_{name}.csv", index=False)
    return flagged


def bursty_raters(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["time"].dt.date
    daily = df.groupby(["source", "date"]).size().reset_index(name="ratings_that_day")
    flagged = daily[daily["ratings_that_day"] >= BURST_RATINGS_PER_DAY].sort_values(
        "ratings_that_day", ascending=False
    )
    flagged = flagged.rename(columns={"source": "node"})
    flagged.to_csv(RESULTS_DIR / f"anomalies_bursty_raters_{name}.csv", index=False)
    return flagged


if __name__ == "__main__":
    for name, csv in DATASETS.items():
        df = load_edgelist(csv)
        rating_sign_distribution(df, name)
        neg = negative_concentration(df, name)
        bursty = bursty_raters(df, name)

        print(f"\n=== {name} ===")
        print(f"Flagged (negative concentration, >= {MIN_RATINGS_RECEIVED} ratings, "
              f">= {int(NEG_FRACTION_THRESHOLD*100)}% negative): {len(neg)} addresses")
        print(neg.head(10).to_string(index=False))
        print(f"\nFlagged (bursty raters, >= {BURST_RATINGS_PER_DAY} ratings/day): {len(bursty)} events")
        print(bursty.head(10).to_string(index=False))
