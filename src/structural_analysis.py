"""
Phase 2 - Structural analysis of the Bitcoin OTC / Bitcoin Alpha networks.

Computes: degree distributions, density, reciprocity, clustering coefficient,
weakly/strongly connected components, diameter and average shortest path
length (on the largest weakly connected component, treated as undirected
for distance purposes since the raw digraph is very sparse/directed).

Outputs:
- results/degree_distribution_<name>.png
- results/structural_summary.csv
"""

from pathlib import Path
import networkx as nx
import pandas as pd
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


def plot_degree_distribution(G: nx.DiGraph, name: str) -> None:
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, degrees, title in (
        (axes[0], in_degrees, "In-degree distribution"),
        (axes[1], out_degrees, "Out-degree distribution"),
    ):
        ax.hist(degrees, bins=50, color="#3b6ea5", edgecolor="none")
        ax.set_yscale("log")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Count (log scale)")
        ax.set_title(title)
    fig.suptitle(f"{name} — degree distribution")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"degree_distribution_{name}.png", dpi=150)
    plt.close(fig)


def analyze(name: str, csv_name: str) -> dict:
    df = load_edgelist(csv_name)
    G = build_graph(df)
    UG = G.to_undirected()

    largest_wcc_nodes = max(nx.weakly_connected_components(G), key=len)
    G_lcc = G.subgraph(largest_wcc_nodes).copy()
    UG_lcc = G_lcc.to_undirected()

    n_scc = nx.number_strongly_connected_components(G)

    # diameter / avg shortest path on the largest component (undirected view),
    # exact if small enough, otherwise estimated via sampling
    if UG_lcc.number_of_nodes() <= 6000:
        try:
            diameter = nx.diameter(UG_lcc)
        except Exception:
            diameter = None
        avg_shortest_path = nx.average_shortest_path_length(UG_lcc)
    else:
        diameter = None
        avg_shortest_path = None

    stats = {
        "dataset": name,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "reciprocity": nx.reciprocity(G),
        "avg_clustering_undirected": nx.average_clustering(UG),
        "weakly_connected_components": nx.number_weakly_connected_components(G),
        "strongly_connected_components": n_scc,
        "largest_wcc_size": len(largest_wcc_nodes),
        "largest_wcc_fraction": len(largest_wcc_nodes) / G.number_of_nodes(),
        "diameter_largest_wcc": diameter,
        "avg_shortest_path_largest_wcc": avg_shortest_path,
        "avg_in_degree": sum(d for _, d in G.in_degree()) / G.number_of_nodes(),
        "max_in_degree": max(d for _, d in G.in_degree()),
        "max_out_degree": max(d for _, d in G.out_degree()),
    }

    plot_degree_distribution(G, name)
    return stats


if __name__ == "__main__":
    rows = [analyze(name, csv) for name, csv in DATASETS.items()]
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(RESULTS_DIR / "structural_summary.csv", index=False)
    print(summary_df.to_string(index=False))
