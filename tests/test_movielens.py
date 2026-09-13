"""Tests for the MovieLens adapter (no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from heterosplit import SplitSpec, split_records
from heterosplit.datasets.movielens import (
    MOVIELENS_SMALL_URL,
    load_movielens_csv,
    movielens_schema,
    records_from_movielens,
)


class TestRecordsFromMovielens:
    def test_maps_bipartite_and_labels(self) -> None:
        table = {
            "userId": [1, 1, 2, 3],
            "movieId": [10, 20, 10, 30],
            "rating": ["5.0", "2.0", "4.0", "nan"],  # last dropped (nan) when labelling
        }
        records = records_from_movielens(table)
        assert records.n_records == 3
        assert not records.schema.is_self_relation
        assert records.schema.source_type == "user"
        assert records.schema.destination_type == "movie"
        decoded = records.label_codebook.decode(records.labels).tolist()  # type: ignore[union-attr]
        assert decoded == ["liked", "disliked", "liked"]

    def test_without_label_keeps_all(self) -> None:
        table = {"userId": [1, 2], "movieId": [10, 20], "rating": ["nan", "3"]}
        records = records_from_movielens(table, with_label=False)
        assert records.n_records == 2
        assert not records.has_labels


class TestSplitMovielens:
    def test_cold_item_split_is_clean(self) -> None:
        rng = np.random.default_rng(0)
        n = 3000
        table = {
            "userId": rng.integers(0, 200, n),
            "movieId": rng.integers(0, 400, n),
            "rating": rng.integers(1, 6, n).astype(float),
        }
        records = records_from_movielens(table, with_label=False)
        spec = SplitSpec(
            supervision_edge=("user", "rates", "movie"),
            roles=dict(records.schema.roles),
            regime="destination_cold_start",  # cold item
            seed=1,
        )
        result = split_records(records, spec)
        assert result.covers_all_records()
        result.audit.raise_for_leakage()
        # test movies never appear in training (cold items)
        dst = records.destination_codes
        train_items = set(dst[result.train_indices].tolist())
        test_items = set(dst[result.test_indices].tolist())
        assert train_items.isdisjoint(test_items)


class TestLoadMovielensCsv:
    def test_reads_ratings_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ratings.csv"
        csv_path.write_text(
            "userId,movieId,rating,timestamp\n1,10,5.0,1000\n1,20,3.0,1001\n2,10,4.5,1002\n",
            encoding="utf-8",
        )
        records = load_movielens_csv(csv_path)
        assert records.n_records == 3
        assert records.schema == movielens_schema()


def test_url_is_grouplens() -> None:
    assert "grouplens.org" in MOVIELENS_SMALL_URL
