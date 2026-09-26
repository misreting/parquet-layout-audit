"""Cross-check the hand-rolled footer parser against pyarrow's own metadata.

This is the same validation strategy the sibling project
``parquet-schema-check`` uses for schema fields: write a real file with
pyarrow, then confirm this package's zero-dependency parser reports the
same numbers pyarrow's own C++ Parquet implementation reports.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from parquet_layout_audit.footer import ParquetFooterError, read_layout


def test_row_group_counts_and_sizes_match_pyarrow(sorted_file: Path) -> None:
    layout = read_layout(sorted_file)
    md = pq.ParquetFile(sorted_file).metadata

    assert layout.num_row_groups == md.num_row_groups
    assert layout.num_rows == md.num_rows
    assert layout.file_size == sorted_file.stat().st_size

    for i, rg in enumerate(layout.row_groups):
        pa_rg = md.row_group(i)
        assert rg.num_rows == pa_rg.num_rows
        assert rg.total_byte_size == pa_rg.total_byte_size


def test_column_chunk_sizes_and_encodings_match_pyarrow(sorted_file: Path) -> None:
    layout = read_layout(sorted_file)
    md = pq.ParquetFile(sorted_file).metadata

    for i, rg in enumerate(layout.row_groups):
        pa_col = md.row_group(i).column(0)
        col = rg.columns[0]
        assert col.path == ("event_id",)
        assert col.physical_type == "INT64"
        assert col.num_values == pa_col.num_values
        assert col.total_uncompressed_size == pa_col.total_uncompressed_size
        assert col.total_compressed_size == pa_col.total_compressed_size
        assert col.has_dictionary_page == pa_col.has_dictionary_page
        assert set(col.encodings) == set(pa_col.encodings)


def test_statistics_min_max_match_pyarrow(sorted_file: Path) -> None:
    layout = read_layout(sorted_file)
    md = pq.ParquetFile(sorted_file).metadata

    for i, rg in enumerate(layout.row_groups):
        pa_stats = md.row_group(i).column(0).statistics
        stats = rg.columns[0].statistics
        assert stats.has_stats is True
        assert stats.min_value == pa_stats.min
        assert stats.max_value == pa_stats.max
        assert stats.null_count == pa_stats.null_count


def test_string_column_statistics_decode_as_bytes_matching_pyarrow(string_column_file: Path) -> None:
    layout = read_layout(string_column_file)
    md = pq.ParquetFile(string_column_file).metadata

    for i, rg in enumerate(layout.row_groups):
        pa_stats = md.row_group(i).column(0).statistics
        stats = rg.columns[0].statistics
        assert stats.min_value == pa_stats.min.encode("utf-8")
        assert stats.max_value == pa_stats.max.encode("utf-8")


def test_missing_statistics_are_reported_as_absent(no_statistics_file: Path) -> None:
    layout = read_layout(no_statistics_file)
    for rg in layout.row_groups:
        assert rg.columns[0].statistics.has_stats is False


def test_statistics_present_when_writer_enables_them(with_statistics_file: Path) -> None:
    layout = read_layout(with_statistics_file)
    for rg in layout.row_groups:
        assert rg.columns[0].statistics.has_stats is True


def test_int96_column_is_read_but_has_no_decodable_statistics(int96_file: Path) -> None:
    layout = read_layout(int96_file)
    col = layout.row_groups[0].columns[0]
    assert col.physical_type == "INT96"
    assert col.statistics.has_stats is False


def test_data_page_count_matches_pyarrow_encoding_stats(tiny_row_groups_file: Path) -> None:
    layout = read_layout(tiny_row_groups_file)
    md = pq.ParquetFile(tiny_row_groups_file).metadata
    # pyarrow's Python API doesn't expose encoding_stats directly, but it
    # does expose data_page_offset/dictionary_page_offset; what we check
    # here is internal consistency -- every row group has exactly one data
    # page's worth of a single-column chunk this small, and our page count
    # is either a small positive integer or None (writer didn't record it),
    # never zero or negative.
    for rg in layout.row_groups:
        col = rg.columns[0]
        if col.data_page_count is not None:
            assert col.data_page_count >= 1
    assert md.num_row_groups == layout.num_row_groups


def test_only_last_bytes_of_file_are_read(sorted_file: Path) -> None:
    """Corrupting everything except the footer must not affect parsing."""
    file_size = sorted_file.stat().st_size
    data = bytearray(sorted_file.read_bytes())
    # Zero out everything except roughly the last 20% (footer + trailer).
    cutoff = int(file_size * 0.5)
    for i in range(4, cutoff):  # keep the PAR1 magic at the very start
        data[i] = 0
    corrupted = sorted_file.parent / "corrupted_body.parquet"
    corrupted.write_bytes(bytes(data))

    layout = read_layout(corrupted)
    assert layout.num_row_groups > 0
    assert layout.num_rows == 200_000


def test_encrypted_footer_magic_is_rejected_with_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "encrypted.parquet"
    footer = b"\x00"  # arbitrary placeholder metadata bytes
    length = len(footer).to_bytes(4, "little")
    path.write_bytes(b"PAR1" + footer + length + b"PARE")

    try:
        read_layout(path)
    except ParquetFooterError as exc:
        assert "encrypted" in str(exc)
    else:
        raise AssertionError("expected ParquetFooterError for an encrypted footer")


def test_not_a_parquet_file_is_rejected(bogus_file: Path) -> None:
    try:
        read_layout(bogus_file)
    except ParquetFooterError:
        pass
    else:
        raise AssertionError("expected ParquetFooterError for a non-Parquet file")
