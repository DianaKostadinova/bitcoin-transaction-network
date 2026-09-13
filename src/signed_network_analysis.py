"""
Phase 8 - Signed-network-specific analysis: Fairness & Goodness scoring and
Structural Balance Theory (triad analysis).

These techniques are specific to *signed* networks (edges carry +/- weight)
and are the methods actually used in the research literature that produced
these exact datasets (Kumar et al., Leskovec et al.) - deeper than generic
(unsigned) centrality/community analysis.

1. Fairness & Goodness (iterative, similar to Kumar et al. "Edge Weight
   Prediction in Weighted Signed Networks", ICDM 2016):
   - Goodness(v) in [-1, 1]  = how trustworthy/good node v is, estimated from
     the (fairness-weighted) ratings it *receives*.
   - Fairness(u) in [0, 1]   = how reliable/fair rater u is, estimated from
     how close u's *given* ratings are to the receivers' goodness scores.
   Computed by mutual iteration until convergence.

2. Structural Balance Theory: for every triangle in the undirected "who has
   a relationship with whom" graph, classify it as *balanced* (product of
   the 3 edge signs is positive: 0 or 2 negative edges) or *unbalanced*
   (product negative: 1 or 3 negative edges). Real social/trust networks are
   expected to be overwhelmingly balanced.

Outputs:
- results/fairness_goodness_<name>.csv       (per-node scores)
- results/fairness_goodness_hist_<name>.png  (goodness distribution)
- results/balance_triads_<name>.csv          (summary counts)
- results/signed_analysis_overview.csv       (cross-dataset + cross-phase summary)
"""

from pathlib import Path
from collections import defaultdict
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_graph import load_edgelist, build_graph

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "bitcoin_otc": "soc-sign-bitcoinotc.csv",
    "bitcoin_alpha": "soc-sign-bitcoinalpha.csv",
}


def fairness_goodness(G: nx.DiGraph, max_iter=100, tol=1e-6):
    nodes = list(G.nodes())
    in_edges = defaultdict(list)
    out_edges = defaultdict(list)
    for u, v, data in G.edges(data=True):
        w = data["weight"] / 10.0  # normalize rating (-10..10) to (-1..1)
        in_edges[v].append((u, w))
        out_edges[u].append((v, w))

    goodness = {}
    for n in nodes:
        vals = [w for _, w in in_edges.get(n, [])]
        goodness[n] = float(np.mean(vals)) if vals else 0.0
    fairness = {n: 1.0 for n in nodes}

    for _ in range(max_iter):
        new_goodness = {}
        for v in nodes:
            edges = in_edges.get(v, [])
            if edges:
                g = sum(fairness[u] * w for u, w in edges) / len(edges)
            else:
                g = 0.0
            new_goodness[v] = max(-1.0, min(1.0, g))

        new_fairness = {}
        for u in nodes:
            edges = out_edges.get(u, [])
            if edges:
                f = 1 - sum(abs(w - new_goodness[v]) for v, w in edges) / (2 * len(edges))
            else:
                f = 1.0
            new_fairness[u] = max(0.0, min(1.0, f))

        diff = (sum(abs(new_goodness[n] - goodness[n]) for n in nodes)
                + sum(abs(new_fairness[n] - fairness[n]) for n in nodes))
        fairness, goodness = new_fairness, new_goodness
        if diff < tol:
            break

    return fairness, goodness


def build_undirected_sign_graph(G: nx.DiGraph) -> nx.Graph:
    UG = nx.Graph()
    for u, v, data in G.edges(data=True):
        w = data["weight"]
        if UG.has_edge(u, v):
            UG[u][v]["weights"].append(w)
        else:
            UG.add_edge(u, v, weights=[w])
    for u, v, data in UG.edges(data=True):
        avg_w = sum(data["weights"]) / len(data["weights"])
        data["avg_weight"] = avg_w
        data["sign"] = 1 if avg_w > 0 else (-1 if avg_w < 0 else 0)
    return UG


def enumerate_triangles(UG: nx.Graph):
    order = {n: i for i, n in enumerate(UG.nodes())}
    triangles = []
    for n in UG.nodes():
        higher_neighbors = [nb for nb in UG.neighbors(n) if order[nb] > order[n]]
        for i in range(len(higher_neighbors)):
            for j in range(i + 1, len(higher_neighbors)):
                a, b = higher_neighbors[i], higher_neighbors[j]
                if UG.has_edge(a, b):
                    triangles.append((n, a, b))
    return triangles


def balance_analysis(UG: nx.Graph, name: str) -> dict:
    triangles = enumerate_triangles(UG)
    balanced, unbalanced, with_neutral = 0, 0, 0
    for n, a, b in triangles:
        signs = [UG[n][a]["sign"], UG[n][b]["sign"], UG[a][b]["sign"]]
        if 0 in signs:
            with_neutral += 1
            continue
        neg_count = sum(1 for s in signs if s < 0)
        if neg_count % 2 == 0:
            balanced += 1
        else:
            unbalanced += 1

    total_strict = balanced + unbalanced
    result = {
        "dataset": name,
        "total_triangles": len(triangles),
        "triangles_with_neutral_edge": with_neutral,
        "balanced": balanced,
        "unbalanced": unbalanced,
        "pct_balanced": 100 * balanced / total_strict if total_strict else float("nan"),
        "pct_unbalanced": 100 * unbalanced / total_strict if total_strict else float("nan"),
    }
    return result


def run(name: str, csv_name: str):
    df = load_edgelist(csv_name)
    G = build_graph(df)

    fairness, goodness = fairness_goodness(G)
    in_degree = dict(G.in_degree())
    fg_df = pd.DataFrame({
        "node": list(G.nodes()),
        "fairness": [fairness[n] for n in G.nodes()],
        "goodness": [goodness[n] for n in G.nodes()],
        "num_ratings_received": [in_degree[n] for n in G.nodes()],
    }).sort_values("goodness")
    fg_df.to_csv(RESULTS_DIR / f"fairness_goodness_{name}.csv", index=False)

    # extreme goodness scores are often driven by nodes with only 1-2 ratings
    # (a single -10 or +10 immediately saturates the score); a >=5-ratings
    # filter gives a more evidence-backed ranking, comparable to Phase 6
    reliable = fg_df[fg_df["num_ratings_received"] >= 5]
    reliable.to_csv(RESULTS_DIR / f"fairness_goodness_reliable_{name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(fg_df["goodness"], bins=50, color="#3b6ea5")
    ax.set_xlabel("Goodness score (-1 = least trustworthy, +1 = most trustworthy)")
    ax.set_ylabel("Number of addresses")
    ax.set_title(f"{name} — distribution of Goodness scores")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"fairness_goodness_hist_{name}.png", dpi=150)
    plt.close(fig)

    UG = build_undirected_sign_graph(G)
    balance = balance_analysis(UG, name)

    # cross-check against Phase 6 negative-concentration heuristic
    neg_path = RESULTS_DIR / f"anomalies_negative_{name}.csv"
    overlap_info = {}
    if neg_path.exists():
        neg_df = pd.read_csv(neg_path)
        flagged_nodes = set(neg_df["node"])
        bottom_n = len(flagged_nodes) if len(flagged_nodes) > 0 else 20
        reliable_sorted = fg_df[fg_df["num_ratings_received"] >= 5]
        bottom_goodness_nodes = set(reliable_sorted.head(bottom_n)["node"])
        overlap = flagged_nodes & bottom_goodness_nodes
        overlap_info = {
            "phase6_flagged_count": len(flagged_nodes),
            "bottom_goodness_compared": bottom_n,
            "overlap_count": len(overlap),
            "overlap_fraction_of_phase6_flagged": len(overlap) / len(flagged_nodes) if flagged_nodes else float("nan"),
        }

    return fg_df, balance, overlap_info


if __name__ == "__main__":
    balance_rows = []
    overview_rows = []
    for name, csv in DATASETS.items():
        fg_df, balance, overlap_info = run(name, csv)
        balance_rows.append(balance)

        print(f"\n=== {name} ===")
        print("Top 5 LOWEST goodness (least trustworthy):")
        print(fg_df.head(5).to_string(index=False))
        print("Top 5 HIGHEST goodness (most trustworthy):")
        print(fg_df.tail(5).sort_values("goodness", ascending=False).to_string(index=False))
        print("Top 5 LOWEST fairness (least reliable raters):")
        print(fg_df.sort_values("fairness").head(5).to_string(index=False))
        reliable = fg_df[fg_df["num_ratings_received"] >= 5]
        print("Top 5 LOWEST goodness AMONG NODES WITH >=5 RATINGS (evidence-backed):")
        print(reliable.head(5).to_string(index=False))
        print(f"Balance: {balance}")
        print(f"Overlap with Phase 6 negative-concentration flags: {overlap_info}")

        overview_rows.append({"dataset": name, **balance, **overlap_info})

    balance_df = pd.DataFrame(balance_rows)
    balance_df.to_csv(RESULTS_DIR / "balance_triads_summary.csv", index=False)

    overview_df = pd.DataFrame(overview_rows)
    overview_df.to_csv(RESULTS_DIR / "signed_analysis_overview.csv", index=False)
