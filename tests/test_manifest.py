"""Tests for split manifests and reproducibility."""

from __future__ import annotations

from pathlib import Path

import pytest

from heterosplit import Manifest, split_records
from heterosplit.manifest import fingerprint_records
from heterosplit.synthetic import make_synthetic_dataset


@pytest.fixture
def result_and_records():  # type: ignore[no-untyped-def]
    ds = make_synthetic_dataset(n_records=500, n_labels=3, seed=0)
    result = split_records(ds.records, ds.spec("pair_cold_start", seed=7))
    return result, ds.records


class TestFingerprint:
    def test_stable_and_sensitive(self) -> None:
        a = make_synthetic_dataset(n_records=100, seed=0).records
        b = make_synthetic_dataset(n_records=100, seed=0).records
        c = make_synthetic_dataset(n_records=100, seed=1).records
        assert fingerprint_records(a) == fingerprint_records(b)
        assert fingerprint_records(a) != fingerprint_records(c)


class TestManifestContents:
    def test_core_fields(self, result_and_records) -> None:  # type: ignore[no-untyped-def]
        result, _ = result_and_records
        m = result.manifest
        assert m.manifest_version == "1"
        assert (
            m.record_counts["train"] + m.record_counts["val"] + m.record_counts["test"]
            == result.records.n_records
        )
        assert m.record_counts["excluded"] == 0
        assert set(m.entity_counts) == set(result.spec.roles)
        assert set(m.index_hashes) == {"train", "val", "test", "excluded"}
        assert m.spec["regime"] == "pair_cold_start"


class TestReproducibility:
    def test_same_inputs_same_digest(self) -> None:
        ds = make_synthetic_dataset(n_records=400, seed=0)
        d1 = split_records(ds.records, ds.spec("pair_cold_start", seed=3)).manifest.digest()
        d2 = split_records(ds.records, ds.spec("pair_cold_start", seed=3)).manifest.digest()
        assert d1 == d2

    def test_different_seed_changes_digest(self) -> None:
        ds = make_synthetic_dataset(n_records=400, seed=0)
        d1 = split_records(ds.records, ds.spec("pair_cold_start", seed=3)).manifest.digest()
        d2 = split_records(ds.records, ds.spec("pair_cold_start", seed=4)).manifest.digest()
        assert d1 != d2

    def test_measurements_excluded_from_digest(self, result_and_records) -> None:  # type: ignore[no-untyped-def]
        result, _ = result_and_records
        base = result.manifest.digest()
        with_meas = result.build_manifest(
            measurements={"runtime_seconds": 1.23, "peak_memory_bytes": 456}
        ).digest()
        assert base == with_meas

    def test_index_hashes_verify_a_rerun(self) -> None:
        ds = make_synthetic_dataset(n_records=400, seed=0)
        spec = ds.spec("pair_cold_start", seed=3)
        m1 = split_records(ds.records, spec).manifest
        m2 = split_records(ds.records, spec).manifest
        assert m1.index_hashes == m2.index_hashes


class TestSerialization:
    def test_roundtrip_dict(self, result_and_records) -> None:  # type: ignore[no-untyped-def]
        result, _ = result_and_records
        m = result.build_manifest(measurements={"runtime_seconds": 0.5})
        restored = Manifest.from_dict(m.to_dict())
        assert restored.digest() == m.digest()
        assert restored.measurements == {"runtime_seconds": 0.5}

    def test_save_and_load(self, tmp_path: Path, result_and_records) -> None:  # type: ignore[no-untyped-def]
        result, _ = result_and_records
        m = result.manifest
        path = tmp_path / "split-manifest.json"
        m.save(path)
        loaded = Manifest.load(path)
        assert loaded.digest() == m.digest()
        assert loaded.input_fingerprint == m.input_fingerprint

    def test_to_json_is_sorted_and_parseable(self, result_and_records) -> None:  # type: ignore[no-untyped-def]
        import json

        result, _ = result_and_records
        parsed = json.loads(result.manifest.to_json())
        assert parsed["heterosplit_version"]
        assert "measurements" in parsed
