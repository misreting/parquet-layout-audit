"""Data model for a decoded Parquet file layout.

These are the shapes :mod:`parquet_layout_audit.footer` produces and
:mod:`parquet_layout_audit.analyze` consumes. Unlike a schema, layout is
inherently about row groups and column chunks, so the model is a tree:
file -> row groups -> column chunks, plus per-column statistics needed to
estimate prunability.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ColumnStatistics:
    """Decoded ``Statistics`` for one column chunk.

    ``min_value``/``max_value`` are already decoded into comparable Python
    objects (``int``, ``float``, ``bool``, or ``bytes``) per
    :mod:`parquet_layout_audit.stats` -- never raw Thrift bytes. They are
    ``None`` when the writer didn't record them (as opposed to recording an
    all-null column, which is ``has_stats=True`` with both ``None`` and
    ``null_count == num_values``).
    """

    has_stats: bool
    min_value: object | None = None
    max_value: object | None = None
    null_count: int | None = None
    distinct_count: int | None = None
    is_min_exact: bool = True
    is_max_exact: bool = True


@dataclass(frozen=True)
class ColumnChunkLayout:
    """One column chunk within one row group."""

    path: tuple[str, ...]
    physical_type: str
    num_values: int
    total_uncompressed_size: int
    total_compressed_size: int
    encodings: frozenset[str]
    codec: str | None
    has_dictionary_page: bool
    statistics: ColumnStatistics
    data_page_count: int | None
    """Number of DATA_PAGE/DATA_PAGE_V2 pages, from ``encoding_stats`` if the
    writer recorded it. ``None`` means the writer didn't record it -- this
    tool does not guess a page count by scanning page headers."""

    @property
    def dotted_path(self) -> str:
        return ".".join(self.path)


@dataclass(frozen=True)
class RowGroupLayout:
    """One row group: its size and the column chunks inside it."""

    num_rows: int
    total_byte_size: int
    total_compressed_size: int | None
    sorting_columns: tuple[tuple[int, bool], ...]
    """(column_index, descending) pairs from the row group's declared
    ``sorting_columns``, when the writer set it. Usually empty -- most
    writers (including pandas/pyarrow's default path) never set this even
    when the data happens to be sorted, which is why this tool estimates
    prunability from statistics rather than relying on this field alone."""
    columns: tuple[ColumnChunkLayout, ...] = field(default_factory=tuple)

    def column_by_path(self) -> dict[tuple[str, ...], ColumnChunkLayout]:
        return {c.path: c for c in self.columns}


@dataclass(frozen=True)
class FileLayout:
    """The decoded row-group/column-chunk layout of a single Parquet file."""

    path: str
    file_size: int
    num_rows: int
    created_by: str | None
    row_groups: tuple[RowGroupLayout, ...] = field(default_factory=tuple)

    @property
    def num_row_groups(self) -> int:
        return len(self.row_groups)

    def column_paths(self) -> list[tuple[str, ...]]:
        seen: dict[tuple[str, ...], None] = {}
        for rg in self.row_groups:
            for col in rg.columns:
                seen.setdefault(col.path, None)
        return list(seen)
