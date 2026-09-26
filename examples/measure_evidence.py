"""Measure the real cost of bad Parquet layout, and check this tool's
prunability estimate against DuckDB's actual bytes read.

Everything printed here is measured on this machine, freshly, each run --
nothing is hardcoded. Needs the dev extras (``pip install -e '.[dev]'``),
which include ``pyarrow`` (to write the files) and ``duckdb`` (to query
them) -- neither is a runtime dependency of the package itself.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from parquet_layout_audit import analyze_file, read_layout

N = 2_000_000


def bytes_read_for(con: duckdb.DuckDBPyConnection, sql: str, profile_path: Path) -> tuple[int, float]:
    con.execute("PRAGMA enable_profiling='json'")
    con.execute(f"PRAGMA profiling_output='{profile_path}'")
    t0 = time.perf_counter()
    con.execute(sql).fetchall()
    dt = time.perf_counter() - t0
    con.execute("PRAGMA disable_profiling")
    profile = json.loads(profile_path.read_text())
    return int(profile["total_bytes_read"]), dt


def main() -> None:
    random.seed(42)
    con = duckdb.connect()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        profile_path = tmp_path / "profile.json"

        print("=== 1. Sorted vs. unsorted: same data, same row-group size, one selective filter ===")
        ids = list(range(N))
        shuffled = ids[:]
        random.shuffle(shuffled)

        sorted_path = tmp_path / "sorted.parquet"
        unsorted_path = tmp_path / "unsorted.parquet"
        pq.write_table(pa.table({"event_id": pa.array(ids, type=pa.int64())}), sorted_path, row_group_size=100_000)
        pq.write_table(pa.table({"event_id": pa.array(shuffled, type=pa.int64())}), unsorted_path, row_group_size=100_000)

        filter_sql = "SELECT count(*) FROM '{}' WHERE event_id BETWEEN 1000 AND 2000"
        b_sorted, t_sorted = bytes_read_for(con, filter_sql.format(sorted_path), profile_path)
        b_unsorted, t_unsorted = bytes_read_for(con, filter_sql.format(unsorted_path), profile_path)

        report_sorted = analyze_file(read_layout(sorted_path))
        report_unsorted = analyze_file(read_layout(unsorted_path))
        print(f"  this tool's estimate, sorted:   {report_sorted.columns[0].prunability.describe()}")
        print(f"  this tool's estimate, unsorted: {report_unsorted.columns[0].prunability.describe()}")
        print(f"  DuckDB, sorted:   total_bytes_read={b_sorted:,}  wall={t_sorted * 1000:.2f}ms")
        print(f"  DuckDB, unsorted: total_bytes_read={b_unsorted:,}  wall={t_unsorted * 1000:.2f}ms")
        print(f"  -> unsorted reads {b_unsorted / max(b_sorted, 1):.1f}x more bytes for the identical filter")

        print()
        print("=== 2. Row group count: too many tiny groups vs. one giant group vs. a reasonable count ===")
        variants = {
            "tiny_groups (row_group_size=500)": 500,
            "huge_group (row_group_size=N, i.e. 1 group)": N,
            "normal_groups (row_group_size=100_000)": 100_000,
        }
        for label, rg_size in variants.items():
            path = tmp_path / f"variant_{rg_size}.parquet"
            pq.write_table(pa.table({"event_id": pa.array(ids, type=pa.int64())}), path, row_group_size=rg_size)
            b, t = bytes_read_for(con, filter_sql.format(path), profile_path)
            layout = read_layout(path)
            print(
                f"  {label}: row_groups={layout.num_row_groups:5d}  "
                f"total_bytes_read={b:>10,}  wall={t * 1000:7.2f}ms  file_size={path.stat().st_size:,}"
            )

        print()
        print("=== 3. Missing statistics ===")
        with_stats_path = tmp_path / "with_stats.parquet"
        no_stats_path = tmp_path / "no_stats.parquet"
        pq.write_table(
            pa.table({"event_id": pa.array(ids, type=pa.int64())}), with_stats_path,
            row_group_size=100_000, write_statistics=True,
        )
        pq.write_table(
            pa.table({"event_id": pa.array(ids, type=pa.int64())}), no_stats_path,
            row_group_size=100_000, write_statistics=False,
        )
        b_stats, t_stats = bytes_read_for(con, filter_sql.format(with_stats_path), profile_path)
        b_nostats, t_nostats = bytes_read_for(con, filter_sql.format(no_stats_path), profile_path)
        print(f"  with statistics: total_bytes_read={b_stats:,}  wall={t_stats * 1000:.2f}ms")
        print(f"  no statistics:   total_bytes_read={b_nostats:,}  wall={t_nostats * 1000:.2f}ms")
        print(f"  -> missing statistics reads {b_nostats / max(b_stats, 1):.1f}x more bytes for the identical filter")

        print()
        print("=== 4. Too many small files vs. one compacted file (same total rows) ===")
        many_dir = tmp_path / "many_small"
        many_dir.mkdir()
        chunk = N // 200
        for i in range(200):
            part = pa.table({"event_id": pa.array(ids[i * chunk : (i + 1) * chunk], type=pa.int64())})
            pq.write_table(part, many_dir / f"part_{i:03d}.parquet")

        t0 = time.perf_counter()
        con.execute(
            f"SELECT count(*) FROM read_parquet('{many_dir}/*.parquet') WHERE event_id BETWEEN 1000 AND 2000"
        ).fetchall()
        dt_many = time.perf_counter() - t0

        t0 = time.perf_counter()
        con.execute(filter_sql.format(sorted_path)).fetchall()
        dt_one = time.perf_counter() - t0
        print(f"  200 small files: wall={dt_many * 1000:.2f}ms")
        print(f"  1 compacted file (same total rows): wall={dt_one * 1000:.2f}ms")
        print(f"  -> {dt_many / max(dt_one, 1e-9):.1f}x slower wall time for the same query over the same rows")


if __name__ == "__main__":
    main()
