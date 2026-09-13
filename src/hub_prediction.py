"""
Phase 9 - "Will it become a hub?" prediction.

Question: using only a node's EARLY behavior (its first ~30% of the
network's chronological history), can we predict its FINAL PageRank /
whether it will end up among the top-50 most central addresses?

Method:
- Build two early snapshots at 15% and 30% of chronological edges.
- Features: in/out-degree at each snapshot, degree growth between them,
  PageRank/Fairness/Goodness at the 30% snapshot, average rating received.
- Target: PageRank on the FULL graph (final_pagerank), plus a binary
  is_top50 flag.
- Models: Linear Regression and Random Forest, compared against a naive
  baseline (rank by early PageRank alone).
- Evaluation: R^2, Spearman rank correlation, precision@50.

Outputs:
- results/hub_prediction_features_<name>.csv
- results/hub_prediction_metrics.csv
- results/hub_prediction_scatter_<name>.png
- results/hub_prediction_feature_importance_<name>.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_graph import load_edgelist, build_graph
from signed_network_analysis import fairness_goodness

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "bitcoin_otc": "soc-sign-bitcoinotc.csv",
    "bitcoin_alpha": "soc-sign-bitcoinalpha.csv",
}

EARLY_FRACTIONS = (0.15, 0.30)
TOP_K = 50


def snapshot_graph(df_sorted: pd.DataFrame, frac: float) -> nx.DiGraph:
    cutoff = int(len(df_sorted) * frac)
    sub = df_sorted.iloc[:cutoff]
    G = nx.DiGraph()
    for row in sub.itertuples(index=False):
        G.add_edge(row.source, row.target, weight=row.rating)
    return G


def build_features(name: str, csv_name: str) -> pd.DataFrame:
    df = load_edgelist(csv_name)
    df_sorted = df.sort_values("time").reset_index(drop=True)

    G15 = snapshot_graph(df_sorted, EARLY_FRACTIONS[0])
    G30 = snapshot_graph(df_sorted, EARLY_FRACTIONS[1])
    G_full = build_graph(df)

    pagerank30 = nx.pagerank(G30, weight=None)
    fairness30, goodness30 = fairness_goodness(G30)
    pagerank_full = nx.pagerank(G_full, weight=None)

    in15 = dict(G15.in_degree())
    out15 = dict(G15.out_degree())
    in30 = dict(G30.in_degree())
    out30 = dict(G30.out_degree())

    rating_sum30, rating_cnt30 = {}, {}
    for u, v, data in G30.edges(data=True):
        rating_sum30[v] = rating_sum30.get(v, 0) + data["weight"]
        rating_cnt30[v] = rating_cnt30.get(v, 0) + 1

    nodes = [n for n in G30.nodes() if (in30.get(n, 0) + out30.get(n, 0)) > 0]

    final_rank = {n: r + 1 for r, (n, _) in
                  enumerate(sorted(pagerank_full.items(), key=lambda x: x[1], reverse=True))}
    top50_final = set(n for n, r in final_rank.items() if r <= TOP_K)

    rows = []
    for n in nodes:
        deg15_in, deg15_out = in15.get(n, 0), out15.get(n, 0)
        deg30_in, deg30_out = in30.get(n, 0), out30.get(n, 0)
        rows.append({
            "node": n,
            "deg15_in": deg15_in,
            "deg15_out": deg15_out,
            "deg30_in": deg30_in,
            "deg30_out": deg30_out,
            "growth_in": deg30_in - deg15_in,
            "growth_out": deg30_out - deg15_out,
            "pagerank30": pagerank30.get(n, 0.0),
            "fairness30": fairness30.get(n, 1.0),
            "goodness30": goodness30.get(n, 0.0),
            "avg_rating30": (rating_sum30.get(n, 0) / rating_cnt30[n]) if n in rating_cnt30 else 0.0,
            "final_pagerank": pagerank_full.get(n, 0.0),
            "final_rank": final_rank.get(n, len(pagerank_full)),
            "is_top50_final": 1 if n in top50_final else 0,
        })

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(RESULTS_DIR / f"hub_prediction_features_{name}.csv", index=False)
    return feat_df


FEATURE_COLS = ["deg15_in", "deg15_out", "deg30_in", "deg30_out", "growth_in",
                 "growth_out", "pagerank30", "fairness30", "goodness30", "avg_rating30"]


def train_and_evaluate(feat_df: pd.DataFrame, name: str) -> dict:
    X = feat_df[FEATURE_COLS].values
    y_log = np.log1p(feat_df["final_pagerank"].values * 1e6)  # scale up before log for stability

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_log, feat_df.index, test_size=0.3, random_state=42
    )
    test_df = feat_df.loc[idx_test].copy()

    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=8).fit(X_train, y_train)

    pred_lr = lr.predict(X_test)
    pred_rf = rf.predict(X_test)

    # actual top-K computed WITHIN the test set only, so the metric's ceiling
    # is 1.0 (not capped by how many global-top-K nodes randomly fell into test)
    actual_top_in_test = set(test_df.sort_values("final_pagerank", ascending=False).head(TOP_K)["node"])

    def precision_at_k(pred_scores, k=TOP_K):
        top_pred_idx = np.argsort(pred_scores)[::-1][:k]
        predicted_nodes = set(test_df.iloc[top_pred_idx]["node"].values)
        hit = len(predicted_nodes & actual_top_in_test)
        return hit / k

    naive_scores = test_df["pagerank30"].values  # baseline: early PageRank itself

    metrics = {
        "dataset": name,
        "n_nodes": len(feat_df),
        "n_test": len(test_df),
        "r2_linear": r2_score(y_test, pred_lr),
        "r2_rf": r2_score(y_test, pred_rf),
        "mae_linear": mean_absolute_error(y_test, pred_lr),
        "mae_rf": mean_absolute_error(y_test, pred_rf),
        "spearman_naive": spearmanr(naive_scores, test_df["final_pagerank"]).correlation,
        "spearman_linear": spearmanr(pred_lr, test_df["final_pagerank"]).correlation,
        "spearman_rf": spearmanr(pred_rf, test_df["final_pagerank"]).correlation,
        "precision_at_50_naive": precision_at_k(naive_scores),
        "precision_at_50_linear": precision_at_k(pred_lr),
        "precision_at_50_rf": precision_at_k(pred_rf),
    }

    # scatter: predicted (RF) vs actual final pagerank
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(test_df["final_pagerank"], np.expm1(pred_rf) / 1e6, s=12, alpha=0.5, color="#3b6ea5")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual final PageRank")
    ax.set_ylabel("Predicted final PageRank (Random Forest)")
    ax.set_title(f"{name} — predicted vs actual final PageRank\n(from first 30% of history)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"hub_prediction_scatter_{name}.png", dpi=150)
    plt.close(fig)

    # feature importance
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(importances.index, importances.values, color="#3b6ea5")
    ax.set_title(f"{name} — Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"hub_prediction_feature_importance_{name}.png", dpi=150)
    plt.close(fig)

    return metrics


if __name__ == "__main__":
    all_metrics = []
    for name, csv in DATASETS.items():
        print(f"Building features for {name}...")
        feat_df = build_features(name, csv)
        print(f"  {len(feat_df)} nodes present by 30% snapshot")
        metrics = train_and_evaluate(feat_df, name)
        all_metrics.append(metrics)
        print(pd.Series(metrics))

    pd.DataFrame(all_metrics).to_csv(RESULTS_DIR / "hub_prediction_metrics.csv", index=False)
