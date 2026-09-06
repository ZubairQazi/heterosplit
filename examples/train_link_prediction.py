"""End-to-end link prediction: does the split regime change the result?

Trains a small GraphSAGE link predictor on a HeteroSplit split and compares held-out
metrics across regimes. The point is the *gap*: a random/transductive split lets test
drugs appear in the training message-passing graph (easy), while a both-entity cold-start
split holds those drugs out entirely (hard) — so the same model scores much lower under
cold-start, which is exactly the effect HeteroSplit exists to expose.

Requires the ``[pyg]`` extra::

    uv run --extra pyg python examples/train_link_prediction.py

Uses a DrugComb-shaped synthetic graph by default, or real DrugComb via
``HETEROSPLIT_DRUGCOMB_CSV``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling

from heterosplit import SplitSpec, split_records
from heterosplit.datasets.drugcomb import load_drugcomb_csv, records_from_drugcomb
from heterosplit.records import PredictionRecords

REGIMES = ("random", "either_cold_start", "both_cold_start")


class LinkPredictor(nn.Module):
    """Learnable node embeddings -> 2-layer GraphSAGE encoder -> dot-product decoder."""

    def __init__(self, num_nodes: int, dim: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, dim)
        self.conv1 = SAGEConv(dim, dim)
        self.conv2 = SAGEConv(dim, dim)

    def encode(self, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(self.embedding.weight, edge_index))
        return self.conv2(x, edge_index)

    @staticmethod
    def decode(z: Tensor, edge_index: Tensor) -> Tensor:
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


def _roc_auc(scores: Tensor, labels: Tensor) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (no scikit-learn dependency)."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    ranks = torch.cat([pos, neg]).argsort().argsort().float() + 1.0
    n_pos = float(pos.numel())
    return float((ranks[: pos.numel()].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * neg.numel()))


def _average_precision(scores: Tensor, labels: Tensor) -> float:
    order = scores.argsort(descending=True)
    lab = labels[order].float()
    if lab.sum() == 0:
        return float("nan")
    cumulative_tp = torch.cumsum(lab, dim=0)
    precision = cumulative_tp / torch.arange(1, lab.numel() + 1, dtype=torch.float)
    return float((precision * lab).sum() / lab.sum())


def _negatives_within(pool: Tensor, positives: set[tuple[int, int]], k: int, seed: int) -> Tensor:
    """Sample ``k`` negative pairs with *both* endpoints in ``pool`` (not real edges).

    Restricting evaluation negatives to the same node pool as the positives is the fair
    cold-start protocol: it asks "among these (possibly unseen) drugs, can the model tell
    real pairs from fake ones?" rather than letting node-set differences leak in.
    """
    if pool.numel() < 2 or k == 0:
        return torch.empty((2, 0), dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    src: list[int] = []
    dst: list[int] = []
    for _ in range(20):
        a = pool[torch.randint(pool.numel(), (4 * k,), generator=generator)].tolist()
        b = pool[torch.randint(pool.numel(), (4 * k,), generator=generator)].tolist()
        for x, y in zip(a, b, strict=True):
            if x != y and (min(x, y), max(x, y)) not in positives:
                src.append(x)
                dst.append(y)
                if len(src) >= k:
                    break
        if len(src) >= k:
            break
    return torch.tensor([src[:k], dst[:k]], dtype=torch.long)


def train_link_predictor(
    records: PredictionRecords,
    spec: SplitSpec,
    *,
    dim: int = 32,
    epochs: int = 60,
    lr: float = 0.01,
    seed: int = 0,
) -> dict[str, Any]:
    """Split, train on the training message-passing graph, and score held-out edges."""
    torch.manual_seed(seed)
    result = split_records(records, spec)
    num_nodes = records.n_entities(spec.schema.source_type)

    mp_edge = torch.as_tensor(result.message_passing_edge_index(), dtype=torch.long)
    train_pos = torch.as_tensor(result.supervision_edge_index("train"), dtype=torch.long)
    test_pos = torch.as_tensor(result.supervision_edge_index("test"), dtype=torch.long)
    all_pos_index = torch.stack(
        [
            torch.as_tensor(result.records.source_codes, dtype=torch.long),
            torch.as_tensor(result.records.destination_codes, dtype=torch.long),
        ]
    )

    model = LinkPredictor(num_nodes, dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(mp_edge)
        train_neg = negative_sampling(train_pos, num_nodes, train_pos.size(1))
        pos_score = model.decode(z, train_pos)
        neg_score = model.decode(z, train_neg)
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()

    # Fair evaluation: draw test negatives from the same node pool as the test positives.
    positives = {
        (min(a, b), max(a, b))
        for a, b in zip(all_pos_index[0].tolist(), all_pos_index[1].tolist(), strict=True)
    }
    test_pool = torch.unique(test_pos.reshape(-1))
    test_neg = _negatives_within(test_pool, positives, test_pos.size(1), seed)

    model.eval()
    with torch.no_grad():
        z = model.encode(mp_edge)
        scores = torch.cat([model.decode(z, test_pos), model.decode(z, test_neg)])
        labels = torch.cat([torch.ones(test_pos.size(1)), torch.zeros(test_neg.size(1))])

    return {
        "regime": spec.regime.value if hasattr(spec.regime, "value") else str(spec.regime),
        "train_pos": int(train_pos.size(1)),
        "test_pos": int(test_pos.size(1)),
        "excluded": result.n_excluded,
        "test_auc": _roc_auc(scores, labels),
        "test_ap": _average_precision(scores, labels),
    }


def compare_regimes(
    records: PredictionRecords,
    regimes: tuple[str, ...] = REGIMES,
    *,
    epochs: int = 60,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    for regime in regimes:
        spec = SplitSpec(
            supervision_edge=("drug", "synergy", "drug"),
            roles=dict(records.schema.roles),
            regime=regime,
            ratios=(0.8, 0.1, 0.1),
            undirected_pairs=True,
            seed=seed,
        )
        rows.append(train_link_predictor(records, spec, epochs=epochs, seed=seed))
    return rows


def structured_synthetic_records(
    n_drugs: int = 160,
    n_records: int = 6000,
    n_clusters: int = 8,
    p_in: float = 0.9,
    n_cells: int = 30,
    seed: int = 0,
) -> PredictionRecords:
    """A community-structured drug graph (stochastic block model) so structure is learnable.

    Each drug belongs to a cluster; with probability ``p_in`` an edge connects two drugs of
    the same cluster, otherwise two random drugs. A GNN can recover this structure, so a
    transductive split (test drugs seen during training) genuinely beats a cold-start one
    (test drugs held out entirely) — the effect this example exists to show, offline.
    """
    rng = np.random.default_rng(seed)
    cluster = rng.integers(0, n_clusters, n_drugs)
    members = [np.flatnonzero(cluster == k) for k in range(n_clusters)]

    candidates = 2 * n_records
    src = rng.integers(0, n_drugs, candidates)
    dst = rng.integers(0, n_drugs, candidates)
    use_same = rng.random(candidates) < p_in
    for k in range(n_clusters):
        pick = use_same & (cluster[src] == k)
        if pick.any() and members[k].size:
            dst[pick] = rng.choice(members[k], int(pick.sum()))
    keep = src != dst
    src, dst = src[keep][:n_records], dst[keep][:n_records]
    cells = rng.integers(0, n_cells, src.shape[0])
    return records_from_drugcomb(
        {
            "drug_row": src.astype(np.int64),
            "drug_col": dst.astype(np.int64),
            "cell_line_name": cells.astype(np.int64),
            "synergy_loewe": np.zeros(src.shape[0]),
        },
        with_label=False,
    )


def _load_records() -> PredictionRecords:
    csv_path = os.environ.get("HETEROSPLIT_DRUGCOMB_CSV")
    if csv_path and Path(csv_path).exists():
        print(f"Using real DrugComb data from {csv_path}")
        return load_drugcomb_csv(csv_path, max_rows=200_000, with_label=False)
    print("Using a structured DrugComb-shaped synthetic graph (set HETEROSPLIT_DRUGCOMB_CSV).")
    return structured_synthetic_records()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = _load_records()
    print(f"drugs={records.n_entities('drug')}  records={records.n_records}\n")
    rows = compare_regimes(records, epochs=args.epochs, seed=args.seed)

    print(f"{'regime':<20} {'train+':>8} {'test+':>7} {'excl':>7} {'test AUC':>9} {'test AP':>8}")
    for r in rows:
        print(
            f"{r['regime']:<20} {r['train_pos']:>8} {r['test_pos']:>7} {r['excluded']:>7} "
            f"{r['test_auc']:>9.3f} {r['test_ap']:>8.3f}"
        )
    print("\nLower cold-start AUC vs random = the split regime materially changes difficulty.")


if __name__ == "__main__":
    main()
