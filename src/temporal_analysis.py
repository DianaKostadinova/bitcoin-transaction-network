"""
Phase 5 - Temporal dynamics of the Bitcoin OTC / Bitcoin Alpha trust networks.

Outputs:
- results/growth_<name>.png            (cumulative nodes & edges over time)
- results/activity_<name>.png          (edges per month + positive/negative split)
- results/bursts_<name>.csv            (weeks with anomalously high activity, z-score)
- results/centrality_evolution_<name>.png (PageRank rank-over-time for early top nodes)
"""

from pathlib import Path
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


def growth_plot(df: pd.DataFrame, name: str) -> None:
    df_sorted = df.sort_values("time").reset_index(drop=True)
    df_sorted["edge_num"] = range(1, len(df_sorted) + 1)

    seen = set()
    cum_nodes = []
    for row in df_sorted.itertuples():
        seen.add(row.source)
        seen.add(row.target)
        cum_nodes.append(len(seen))
    df_sorted["cum_nodes"] = cum_nodes

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(df_sorted["time"], df_sorted["edge_num"], color="#3b6ea5", label="Cumulative edges (ratings)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Cumulative edges", color="#3b6ea5")
    ax2 = ax1.twinx()
    ax2.plot(df_sorted["time"], df_sorted["cum_nodes"], color="#c0392b", label="Cumulative nodes")
    ax2.set_ylabel("Cumulative nodes", color="#c0392b")
    fig.suptitle(f"{name} — network growth over time")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"growth_{name}.png", dpi=150)
    plt.close(fig)


def activity_and_bursts(df: pd.DataFrame, name: str) -> pd.DataFrame:
    weekly = df.set_index("time").resample("W")
    counts = weekly.size()
    pos = weekly.apply(lambda x: (x["rating"] > 0).sum())
    neg = weekly.apply(lambda x: (x["rating"] < 0).sum())

    monthly = df.set_index("time").resample("ME").size()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    axes[0].bar(monthly.index, monthly.values, width=20, color="#3b6ea5")
    axes[0].set_title(f"{name} — ratings per month")
    axes[0].set_ylabel("Count")

    axes[1].plot(pos.index, pos.values, color="#2e8b57", label="Positive ratings")
    axes[1].plot(neg.index, neg.values, color="#c0392b", label="Negative ratings")
    axes[1].set_title("Weekly positive vs negative ratings")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"activity_{name}.png", dpi=150)
    plt.close(fig)

    z = (counts - counts.mean()) / counts.std()
    bursts = pd.DataFrame({"week": counts.index, "num_ratings": counts.values, "z_score": z.values})
    bursts = bursts[bursts["z_score"] > 2.5].sort_values("z_score", ascending=False)
    bursts.to_csv(RESULTS_DIR / f"bursts_{name}.csv", index=False)
    return bursts


def centrality_evolution(df: pd.DataFrame, name: str, top_nodes: list) -> None:
    """Track PageRank rank of `top_nodes` (from Phase 3, computed on the
    full/final graph) across 10 expanding snapshots of the network's history —
    shows whether today's most central addresses were central early on or
    grew into it."""
    df_sorted = df.sort_values("time").reset_index(drop=True)
    n = len(df_sorted)
    checkpoints = [int(n * f) for f in np.linspace(0.1, 1.0, 10)]

    records = []
    for cp in checkpoints:
        snap_df = df_sorted.iloc[:cp]
        G = nx.DiGraph()
        for row in snap_df.itertuples(index=False):
            G.add_edge(row.source, row.target)
        pr = nx.pagerank(G, weight=None)
        max_rank = len(pr)
        ranked = {node: rank + 1 for rank, (node, _) in
                  enumerate(sorted(pr.items(), key=lambda x: x[1], reverse=True))}
        snapshot_time = snap_df["time"].iloc[-1]
        for node in top_nodes:
            records.append({
                "time": snapshot_time,
                "node": node,
                "rank": ranked.get(node, max_rank),
                "total_nodes": max_rank,
            })

    eh_df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(9, 5))
    for node in top_nodes:
        sub = eh_df[eh_df["node"] == node]
        ax.plot(sub["time"], sub["rank"], marker="o", label=f"node {node}")
    ax.invert_yaxis()
    ax.set_ylabel("PageRank rank (1 = most central)")
    ax.set_xlabel("Time (snapshot)")
    ax.set_title(f"{name} — centrality rank evolution of top-5 final-PageRank nodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"centrality_evolution_{name}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for name, csv in DATASETS.items():
        df = load_edgelist(csv)
        growth_plot(df, name)
        bursts = activity_and_bursts(df, name)
        print(f"\n=== {name}: burst weeks (z>2.5) ===")
        print(bursts.to_string(index=False) if not bursts.empty else "none")

        G_full = build_graph(df)
        pr_full = nx.pagerank(G_full, weight=None)
        top5 = [n for n, _ in sorted(pr_full.items(), key=lambda x: x[1], reverse=True)[:5]]
        centrality_evolution(df, name, top5)
