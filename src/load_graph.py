"""
Вчитување на Bitcoin trust/rating датасетите (soc-sign-bitcoin-otc / -alpha)
како насочени, тежински графови (NetworkX DiGraph).

Формат на изворните CSV-датотеки (SNAP): SOURCE, TARGET, RATING, TIME
- SOURCE, TARGET: анонимизирани ID-ови на кориснички сметки
- RATING: -10..+10, оценка на доверба доделена од SOURCE кон TARGET
- TIME: UNIX timestamp

Забелешка: ова не се сурови блокчејн трансакции туку мрежа на доверба
(who-trusts-whom) од платформите Bitcoin OTC и Bitcoin Alpha, која SNAP
ја нуди како proxy за анализа на Bitcoin-поврзани трансакциски мрежи.
"""

from pathlib import Path
import networkx as nx
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

COLUMNS = ["source", "target", "rating", "time"]


def load_edgelist(csv_name: str) -> pd.DataFrame:
    path = DATA_DIR / csv_name
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for row in df.itertuples(index=False):
        G.add_edge(row.source, row.target, weight=row.rating, time=row.time)
    return G


def summary(G: nx.DiGraph, name: str) -> None:
    print(f"--- {name} ---")
    print(f"Nodes (accounts): {G.number_of_nodes()}")
    print(f"Edges (ratings): {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.6f}")
    print(f"Weakly connected components: {nx.number_weakly_connected_components(G)}")
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    print(f"Largest weakly connected component: {len(largest_wcc)} nodes "
          f"({100 * len(largest_wcc) / G.number_of_nodes():.1f}%)")
    print()


if __name__ == "__main__":
    otc_df = load_edgelist("soc-sign-bitcoinotc.csv")
    alpha_df = load_edgelist("soc-sign-bitcoinalpha.csv")

    otc_G = build_graph(otc_df)
    alpha_G = build_graph(alpha_df)

    summary(otc_G, "Bitcoin OTC")
    summary(alpha_G, "Bitcoin Alpha")
