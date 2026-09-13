"""
Phase 3 - Centrality analysis: identify the most active / most central
addresses in the Bitcoin OTC / Bitcoin Alpha networks.

Metrics:
- Degree centrality (in/out) -> most active addresses
- PageRank -> influence accounting for direction of trust
- Betweenness centrality -> brokers/bridges between parts of the network
- HITS (hub, authority) -> trusted-by-trusted addresses

Outputs:
- results/top_addresses_<name>.csv (top 15 per metric)
"""

from pathlib import Path
import networkx as nx
import pandas as pd

from load_graph import load_edgelist, build_graph

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "bitcoin_otc": "soc-sign-bitcoinotc.csv",
    "bitcoin_alpha": "soc-sign-bitcoinalpha.csv",
}

TOP_N = 15


def top_n(d: dict, n: int = TOP_N) -> pd.Series:
    return pd.Series(d).sort_values(ascending=False).head(n)


def analyze(name: str, csv_name: str) -> pd.DataFrame:
    df = load_edgelist(csv_name)
    G = build_graph(df)

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    pagerank = nx.pagerank(G, weight=None)
    # betweenness is expensive on the full graph; k-sample approximation
    # keeps it fast while staying representative for a few thousand nodes
    betweenness = nx.betweenness_centrality(G, k=500, normalized=True, seed=42)
    hubs, authorities = nx.hits(G, max_iter=1000)

    table = pd.DataFrame({
        "in_degree": top_n(in_deg),
    }).reset_index().rename(columns={"index": "node_in_degree"})

    metrics = {
        "in_degree": in_deg,
        "out_degree": out_deg,
        "pagerank": pagerank,
        "betweenness": betweenness,
        "hub": hubs,
        "authority": authorities,
    }

    frames = []
    for metric_name, values in metrics.items():
        s = top_n(values).reset_index()
        s.columns = [f"{metric_name}_node", f"{metric_name}_value"]
        frames.append(s)

    combined = pd.concat(frames, axis=1)
    combined.insert(0, "rank", range(1, len(combined) + 1))
    combined.insert(0, "dataset", name)
    combined.to_csv(RESULTS_DIR / f"top_addresses_{name}.csv", index=False)
    return combined


if __name__ == "__main__":
    for name, csv in DATASETS.items():
        result = analyze(name, csv)
        print(f"\n=== {name} — top {TOP_N} addresses per metric ===")
        print(result.drop(columns=["dataset"]).to_string(index=False))
