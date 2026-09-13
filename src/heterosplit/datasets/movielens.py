"""MovieLens adapter: a non-biomedical, bipartite user--item corpus for link prediction.

[MovieLens](https://grouplens.org/datasets/movielens/) is a classic recommendation
dataset. Each rating is a ``(user, movie)`` interaction; this module maps ratings onto
:class:`~heterosplit.records.PredictionRecords` as a **bipartite** ``(user, rates, movie)``
relation with an optional "liked" label (rating >= threshold). It demonstrates that
HeteroSplit's cold-start regimes are domain-generic: ``source_cold_start`` = new users,
``destination_cold_start`` = new movies, ``both`` = new user *and* new movie.

The small ``ml-latest-small`` release (~1 MB) is downloaded on demand with
:func:`download_movielens`; it is not embedded here. Data © GroupLens, for research use
(see the dataset's README for terms).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import SchemaError
from ..records import PredictionRecords
from ..schema import EntityRole, TaskSchema

__all__ = [
    "MOVIELENS_SMALL_URL",
    "download_movielens",
    "load_movielens_csv",
    "movielens_schema",
    "records_from_movielens",
]

#: GroupLens "latest-small" release (~1 MB): 100k ratings, ~610 users, ~9.7k movies.
MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def movielens_schema() -> TaskSchema:
    """The bipartite ``(user, rates, movie)`` schema MovieLens maps to."""
    return TaskSchema(
        ("user", "rates", "movie"),
        {"user": EntityRole.source("user"), "movie": EntityRole.destination("movie")},
    )


def _column(table: Any, name: str) -> Any:
    try:
        return table[name]
    except (KeyError, TypeError, IndexError) as exc:
        raise SchemaError(f"MovieLens column {name!r} not found in table") from exc


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def records_from_movielens(
    table: Any,
    *,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
    liked_threshold: float = 4.0,
    with_label: bool = True,
) -> PredictionRecords:
    """Map a MovieLens-formatted column store to bipartite :class:`PredictionRecords`.

    Each ``(user, movie)`` rating becomes an edge. With ``with_label`` the record is
    labelled ``"liked"`` when ``rating >= liked_threshold`` else ``"disliked"``; rows with
    a non-finite rating are dropped in that case.
    """
    users = np.asarray(_column(table, user_col), dtype=object)
    movies = np.asarray(_column(table, item_col), dtype=object)
    ratings = np.array([_to_float(v) for v in _column(table, rating_col)], dtype=float)
    if not (users.shape == movies.shape == ratings.shape):
        raise SchemaError("MovieLens columns have inconsistent lengths")

    keep = np.ones(users.shape, dtype=bool)
    if with_label:
        keep &= np.isfinite(ratings)

    columns = {"user": users[keep], "movie": movies[keep]}
    labels = None
    if with_label:
        labels = np.where(ratings[keep] >= liked_threshold, "liked", "disliked")
    return PredictionRecords.from_columns(movielens_schema(), columns, labels=labels)


def load_movielens_csv(
    path: str | Path,
    *,
    liked_threshold: float = 4.0,
    max_rows: int | None = None,
    with_label: bool = True,
) -> PredictionRecords:
    """Load a MovieLens ``ratings.csv`` (``userId,movieId,rating,timestamp``) into records."""
    needed = ["userId", "movieId", "rating"]
    collected: dict[str, list[str]] = {name: [] for name in needed}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise SchemaError(f"{path} is empty")
        try:
            indices = {name: header.index(name) for name in needed}
        except ValueError as exc:
            raise SchemaError(f"MovieLens CSV missing expected column: {exc}") from exc
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            for name, index in indices.items():
                collected[name].append(row[index])
    return records_from_movielens(collected, liked_threshold=liked_threshold, with_label=with_label)


def download_movielens(dest_dir: str | Path, *, url: str = MOVIELENS_SMALL_URL) -> Path:
    """Download and extract the MovieLens small release; return the ratings.csv path.

    Verifies TLS. If GroupLens's server certificate has lapsed (it occasionally does),
    download ``ml-latest-small.zip`` manually and point :func:`load_movielens_csv` at the
    extracted ``ratings.csv`` instead.
    """
    import io
    import urllib.request
    import zipfile

    destination = Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    ratings_name = next(n for n in archive.namelist() if n.endswith("ratings.csv"))
    archive.extract(ratings_name, destination)
    return destination / ratings_name
