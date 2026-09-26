"""Rule-level tests.

Most of these run against real Parquet files (via the fixtures in
conftest.py) so the whole pipeline -- footer parsing, prunability
estimation, and finding generation -- is exercised end to end. A couple of
rules (PLA005's dictionary-encoding check) need a per-row-group
``distinct_count`` statistic that pyarrow's writer never sets (confirmed
empirically: writing a low-cardinality column and reading it back with
pyarrow itself shows ``distinct_count=None``), so those are tested against
directly constructed layout objects instead -- documented here, not hidden.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_layout_audit.analyze import AuditOptions, Severity, analyze_dataset, analyze_file
from parquet_layout_audit.footer import read_layout
from parquet_layout_audit.types import ColumnChunkLayout, ColumnStatistics, FileLayout, RowGroupLayout


def _codes(findings: list) -> set[str]:
    return {f.code for f in findings}


def test_sorted_file_prunes_well_and_raises_no_prunability_finding(sorted_file: Path) -> None:
    layout = read_layout(sorted_file)
    report = analyze_file(layout)
    assert "PLA004" not in _codes(report.findings)
    (col,) = report.columns
    assert col.prunability.avg_pruned is not None
    assert col.prunability.avg_pruned > 15  # out of 20 row groups


def test_unsorted_file_is_flagged_as_effectively_unsorted(unsorted_file: Path) -> None:
    layout = read_layout(unsorted_file)
    report = analyze_file(layout)
    findings = [f for f in report.findings if f.code == "PLA004"]
    assert len(findings) == 1
    assert findings[0].severity is Severity.WARNING
    assert "event_id" in findings[0].column
    assert "sort by" in findings[0].remedy


def test_too_many_row_groups_is_flagged(tiny_row_groups_file: Path) -> None:
    layout = read_layout(tiny_row_groups_file)
    report = analyze_file(layout, AuditOptions(max_row_groups=100))
    assert "PLA001" in _codes(report.findings)


def test_reasonable_row_group_count_is_not_flagged(sorted_file: Path) -> None:
    layout = read_layout(sorted_file)
    report = analyze_file(layout)  # 20 row groups, default threshold 200
    assert "PLA001" not in _codes(report.findings)


def test_single_huge_row_group_is_flagged(single_huge_row_group_file: Path) -> None:
    layout = read_layout(single_huge_row_group_file)
    report = analyze_file(layout, AuditOptions(large_file_min_rows=100_000))
    findings = [f for f in report.findings if f.code == "PLA002"]
    assert len(findings) == 1
    assert "no row-group pruning granularity" in findings[0].message


def test_missing_statistics_is_an_error(no_statistics_file: Path) -> None:
    layout = read_layout(no_statistics_file)
    report = analyze_file(layout)
    findings = [f for f in report.findings if f.code == "PLA003"]
    assert len(findings) == 1
    assert findings[0].severity is Severity.ERROR


def test_present_statistics_do_not_trigger_pla003(with_statistics_file: Path) -> None:
    layout = read_layout(with_statistics_file)
    report = analyze_file(layout)
    assert "PLA003" not in _codes(report.findings)


def test_int96_column_gets_an_info_level_finding_not_an_error(int96_file: Path) -> None:
    layout = read_layout(int96_file)
    report = analyze_file(layout)
    findings = [f for f in report.findings if f.code == "PLA003"]
    assert len(findings) == 1
    assert findings[0].severity is Severity.INFO
    assert "INT96" in findings[0].message


def test_small_pages_are_flagged(tmp_path: Path) -> None:
    n = 200_000
    table = pa.table({"x": pa.array(range(n), type=pa.int64())})
    path = tmp_path / "smallpages.parquet"
    pq.write_table(table, path, data_page_size=1024, row_group_size=n)
    layout = read_layout(path)
    # The actual average page size the writer produces for a 1024-byte
    # target isn't exactly 1024 (RLE/dictionary framing adds overhead), so
    # this pins the threshold comfortably above what was measured (~9.6KB)
    # rather than assuming the writer's target is met exactly.
    report = analyze_file(layout, AuditOptions(large_file_min_rows=10_000_000, tiny_page_bytes=32 * 1024))
    assert "PLA006" in _codes(report.findings)


def test_too_many_small_files_is_a_dataset_level_finding(many_small_files: Path) -> None:
    files = sorted(many_small_files.glob("*.parquet"))
    layouts = [read_layout(f) for f in files]
    findings = analyze_dataset(layouts, AuditOptions(small_file_bytes=1024 * 1024))
    assert len(findings) == 1
    assert findings[0].file is None
    assert findings[0].code == "PLA007"


def test_few_files_or_all_large_produces_no_dataset_finding(sorted_file: Path, unsorted_file: Path) -> None:
    layouts = [read_layout(sorted_file), read_layout(unsorted_file)]
    findings = analyze_dataset(layouts, AuditOptions(small_file_bytes=1))
    assert findings == []


def _make_layout_with_stats(num_values: int, distinct_count: int, encodings: frozenset[str]) -> FileLayout:
    """A minimal synthetic single-row-group file, for rules that need a
    per-column statistic pyarrow's writer doesn't set (distinct_count)."""
    col = ColumnChunkLayout(
        path=("category",),
        physical_type="BYTE_ARRAY",
        num_values=num_values,
        total_uncompressed_size=num_values * 8,
        total_compressed_size=num_values * 8,
        encodings=encodings,
        codec="SNAPPY",
        has_dictionary_page=bool(encodings & {"PLAIN_DICTIONARY", "RLE_DICTIONARY"}),
        statistics=ColumnStatistics(has_stats=True, min_value=b"a", max_value=b"z", null_count=0, distinct_count=distinct_count),
        data_page_count=1,
    )
    rg = RowGroupLayout(num_rows=num_values, total_byte_size=num_values * 8, total_compressed_size=num_values * 8, sorting_columns=(), columns=(col,))
    return FileLayout(path="synthetic.parquet", file_size=num_values * 8, num_rows=num_values, created_by=None, row_groups=(rg,))


def test_low_cardinality_without_dictionary_is_flagged() -> None:
    layout = _make_layout_with_stats(num_values=100_000, distinct_count=5, encodings=frozenset({"PLAIN", "RLE"}))
    report = analyze_file(layout)
    findings = [f for f in report.findings if f.code == "PLA005"]
    assert len(findings) == 1
    assert "5" in findings[0].message


def test_low_cardinality_with_dictionary_is_not_flagged() -> None:
    layout = _make_layout_with_stats(
        num_values=100_000, distinct_count=5, encodings=frozenset({"RLE_DICTIONARY", "PLAIN"})
    )
    report = analyze_file(layout)
    assert "PLA005" not in _codes(report.findings)


def test_high_cardinality_without_dictionary_is_not_flagged() -> None:
    layout = _make_layout_with_stats(num_values=100_000, distinct_count=90_000, encodings=frozenset({"PLAIN"}))
    report = analyze_file(layout)
    assert "PLA005" not in _codes(report.findings)


def test_empty_row_groups_produce_no_row_group_summary() -> None:
    layout = FileLayout(path="empty.parquet", file_size=8, num_rows=0, created_by=None, row_groups=())
    report = analyze_file(layout)
    assert report.row_group_summary is None
    assert report.findings == ()
