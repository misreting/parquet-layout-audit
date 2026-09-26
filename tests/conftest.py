"""Shared fixtures: real Parquet files with deliberately good and bad layouts.

Written with pyarrow (a *dev*-only dependency -- the package under test never
imports it) so tests exercise real Parquet bytes on disk, not synthetic
footer buffers.
"""

from __future__ import annotations

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def sorted_file(tmp_path: Path) -> Path:
    """20 row groups, event_id sorted ascending -- ranges are disjoint."""
    n = 200_000
    ids = list(range(n))
    table = pa.table({"event_id": pa.array(ids, type=pa.int64())})
    path = tmp_path / "sorted.parquet"
    pq.write_table(table, path, row_group_size=10_000)
    return path


@pytest.fixture
def unsorted_file(tmp_path: Path) -> Path:
    """Same data as ``sorted_file``, shuffled -- every row group's range
    ends up spanning nearly the entire domain."""
    n = 200_000
    ids = list(range(n))
    random.Random(42).shuffle(ids)
    table = pa.table({"event_id": pa.array(ids, type=pa.int64())})
    path = tmp_path / "unsorted.parquet"
    pq.write_table(table, path, row_group_size=10_000)
    return path


@pytest.fixture
def tiny_row_groups_file(tmp_path: Path) -> Path:
    """400 row groups of 100 rows each -- overhead-dominated."""
    n = 40_000
    table = pa.table({"event_id": pa.array(range(n), type=pa.int64())})
    path = tmp_path / "tiny_groups.parquet"
    pq.write_table(table, path, row_group_size=100)
    return path


@pytest.fixture
def single_huge_row_group_file(tmp_path: Path) -> Path:
    """One row group covering the whole file -- no pruning granularity."""
    n = 600_000
    table = pa.table({"event_id": pa.array(range(n), type=pa.int64())})
    path = tmp_path / "huge_group.parquet"
    pq.write_table(table, path, row_group_size=n)
    return path


@pytest.fixture
def no_statistics_file(tmp_path: Path) -> Path:
    n = 50_000
    table = pa.table({"event_id": pa.array(range(n), type=pa.int64())})
    path = tmp_path / "no_stats.parquet"
    pq.write_table(table, path, row_group_size=10_000, write_statistics=False)
    return path


@pytest.fixture
def with_statistics_file(tmp_path: Path) -> Path:
    n = 50_000
    table = pa.table({"event_id": pa.array(range(n), type=pa.int64())})
    path = tmp_path / "with_stats.parquet"
    pq.write_table(table, path, row_group_size=10_000, write_statistics=True)
    return path


@pytest.fixture
def string_column_file(tmp_path: Path) -> Path:
    """Sorted string column, to exercise BYTE_ARRAY statistics decoding."""
    n = 50_000
    words = [f"user-{i:07d}" for i in range(n)]
    table = pa.table({"user_id": pa.array(words)})
    path = tmp_path / "strings.parquet"
    pq.write_table(table, path, row_group_size=5_000, use_dictionary=False)
    return path


@pytest.fixture
def int96_file(tmp_path: Path) -> Path:
    """A column using the deprecated INT96 timestamp encoding."""
    import datetime as dt

    table = pa.table(
        {"ts": pa.array([dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(1000)], type=pa.timestamp("us"))}
    )
    path = tmp_path / "int96.parquet"
    pq.write_table(table, path, use_deprecated_int96_timestamps=True)
    return path


@pytest.fixture
def many_small_files(tmp_path: Path) -> Path:
    """200 small files that together form one dataset."""
    out = tmp_path / "many_small"
    out.mkdir()
    for i in range(20):
        table = pa.table({"event_id": pa.array(range(i * 100, (i + 1) * 100), type=pa.int64())})
        pq.write_table(table, out / f"part_{i:03d}.parquet")
    return out


@pytest.fixture
def bogus_file(tmp_path: Path) -> Path:
    path = tmp_path / "bogus.parquet"
    path.write_bytes(b"not a parquet file")
    return path
