"""
Phase 4 - Community detection (Louvain) on the Bitcoin OTC / Bitcoin Alpha
trust networks.

Outputs:
- results/communities_<name>.csv          (node -> community_id)
- results/community_summary_<name>.csv    (per-community stats)
- results/network_communities_<name>.png  (force-directed layout colored by community)
- results/community_sizes_<name>.png      (bar chart of community sizes)
"""

from pathlib import Path
import networkx as nx
import pandas as pd
import numpy as np
import community as community_louvain  # python-louvain
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from load_graph import load_edgelist, build_graph

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "bitcoin_otc": "soc-sign-bitcoinotc.csv",
    "bitcoin_alpha": "soc-sign-bitcoinalpha.csv",
}


def analyze(name: str, csv_name: str):
    df = load_edgelist(csv_name)
    G = build_graph(df)
    UG = G.to_undirected()

    # Louvain/modularity require non-negative weights; ratings range -10..+10,
    # so community structure is detected on the *unweighted* connection graph
    # (who-rates-whom, regardless of sign) - sign is analyzed separately per
    # community via avg_rating_of_members below.
    UG_unweighted = nx.Graph()
    UG_unweighted.add_nodes_from(UG.nodes())
    UG_unweighted.add_edges_from(UG.edges())

    partition = community_louvain.best_partition(UG_unweighted, random_state=42)
    modularity = community_louvain.modularity(partition, UG_unweighted)

    part_df = pd.DataFrame(
        [(node, comm) for node, comm in partition.items()],
        columns=["node", "community"],
    )
    part_df.to_csv(RESULTS_DIR / f"communities_{name}.csv", index=False)

    # per-community stats: size, internal density, avg rating, avg pagerank
    pagerank = nx.pagerank(G, weight=None)
    rating_map = {}
    for u, v, data in G.edges(data=True):
        rating_map.setdefault(u, []).append(data["weight"])
        rating_map.setdefault(v, []).append(data["weight"])

    rows = []
    for comm_id, nodes in part_df.groupby("community")["node"].apply(list).items():
        sub = UG.subgraph(nodes)
        n = len(nodes)
        e = sub.number_of_edges()
        density = nx.density(sub) if n > 1 else 0.0
        avg_pr = np.mean([pagerank.get(node, 0) for node in nodes])
        ratings = [r for node in nodes for r in rating_map.get(node, [])]
        avg_rating = np.mean(ratings) if ratings else np.nan
        rows.append({
            "community": comm_id,
            "size": n,
            "internal_edges": e,
            "internal_density": density,
            "avg_pagerank": avg_pr,
            "avg_rating_of_members": avg_rating,
        })
    summary_df = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
    summary_df.to_csv(RESULTS_DIR / f"community_summary_{name}.csv", index=False)

    # community size distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    top20 = summary_df.head(20)
    ax.bar(top20["community"].astype(str), top20["size"], color="#3b6ea5")
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Number of nodes")
    ax.set_title(f"{name} — top 20 community sizes (of {len(summary_df)} total, modularity={modularity:.3f})")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"community_sizes_{name}.png", dpi=150)
    plt.close(fig)

    # network visualization colored by community (largest component only, for clarity)
    largest_cc = max(nx.connected_components(UG), key=len)
    sub = UG.subgraph(largest_cc)
    # cap the drawn graph to keep the figure legible/fast for large networks
    if sub.number_of_nodes() > 2000:
        top_nodes_by_degree = sorted(sub.degree, key=lambda x: x[1], reverse=True)[:2000]
        sub = sub.subgraph([n for n, _ in top_nodes_by_degree])

    pos = nx.spring_layout(sub, seed=42, k=None)
    node_colors = [partition.get(n, -1) for n in sub.nodes()]
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_edges(sub, pos, alpha=0.05, width=0.5, ax=ax)
    nodes = nx.draw_networkx_nodes(
        sub, pos, node_size=15, node_color=node_colors, cmap=cm.tab20, ax=ax
    )
    ax.set_title(f"{name} — network colored by Louvain community\n"
                 f"(largest component, top {sub.number_of_nodes()} nodes by degree, modularity={modularity:.3f})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"network_communities_{name}.png", dpi=150)
    plt.close(fig)

    return {
        "dataset": name,
        "num_communities": summary_df.shape[0],
        "modularity": modularity,
        "largest_community_size": summary_df["size"].max(),
        "largest_community_fraction": summary_df["size"].max() / G.number_of_nodes(),
    }


if __name__ == "__main__":
    overview = [analyze(name, csv) for name, csv in DATASETS.items()]
    overview_df = pd.DataFrame(overview)
    overview_df.to_csv(RESULTS_DIR / "community_overview.csv", index=False)
    print(overview_df.to_string(index=False))
