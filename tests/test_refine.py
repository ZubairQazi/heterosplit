"""Tests for the bounded local-search assignment refinement."""

from __future__ import annotations

import numpy as np

from heterosplit import split_records
from heterosplit.objective import distribution_divergence, size_deviation
from heterosplit.splitters.assignment import assign_groups, refine_assignment
from heterosplit.synthetic import make_synthetic_dataset


def _split_stats(
    assignment: np.ndarray, sizes: np.ndarray, vc: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    split_sizes = np.array([sizes[assignment == s].sum() for s in range(k)], dtype=float)
    split_vc = np.array([vc[assignment == s].sum(axis=0) for s in range(k)], dtype=float)
    return split_sizes, split_vc


class TestRefineAssignment:
    def _fixture(self, seed: int = 0):  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(seed)
        n_groups = 300
        sizes = rng.integers(1, 6, n_groups).astype(np.int64)
        vc = rng.integers(0, 5, size=(n_groups, 3)).astype(np.float64)
        ratios = (0.6, 0.2, 0.2)
        base = assign_groups(sizes, ratios, seed=seed)
        refined = refine_assignment(base, sizes.astype(float), vc, ratios)
        return sizes, vc, ratios, base, refined

    def test_size_never_worsens(self) -> None:
        sizes, vc, ratios, base, refined = self._fixture()
        bs, _ = _split_stats(base, sizes, vc, 3)
        rs, _ = _split_stats(refined, sizes, vc, 3)
        assert size_deviation(rs, ratios) <= size_deviation(bs, ratios) + 1e-9

    def test_divergence_not_worse(self) -> None:
        sizes, vc, ratios, base, refined = self._fixture()
        _, bvc = _split_stats(base, sizes, vc, 3)
        _, rvc = _split_stats(refined, sizes, vc, 3)
        assert distribution_divergence(rvc, ratios) <= distribution_divergence(bvc, ratios) + 1e-9

    def test_deterministic(self) -> None:
        sizes, vc, ratios, _, refined = self._fixture(seed=1)
        again = refine_assignment(
            assign_groups(sizes, ratios, seed=1), sizes.astype(float), vc, ratios
        )
        np.testing.assert_array_equal(refined, again)

    def test_no_values_is_noop(self) -> None:
        assignment = np.array([0, 1, 0, 1], dtype=np.int64)
        empty_vc = np.zeros((4, 0), dtype=np.float64)
        out = refine_assignment(assignment, np.ones(4), empty_vc, (0.5, 0.5))
        np.testing.assert_array_equal(out, assignment)


class TestRefineImprovesRealSplits:
    def test_label_balance_reasonable_without_stratify(self) -> None:
        # Soft refinement (default) should keep label distributions close across splits.
        ds = make_synthetic_dataset(n_records=3000, n_labels=3, seed=0)
        result = split_records(ds.records, ds.spec("source_cold_start", ratios=(0.7, 0.3)))
        labels = ds.records.labels
        assert labels is not None
        train = np.bincount(labels[result.train_indices], minlength=3) / result.counts["train"]
        test = np.bincount(labels[result.test_indices], minlength=3) / result.counts["test"]
        np.testing.assert_allclose(train, test, atol=0.08)
