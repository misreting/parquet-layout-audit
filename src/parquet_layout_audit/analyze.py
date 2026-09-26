"""Turn a decoded :class:`~parquet_layout_audit.types.FileLayout` into findings.

This is where the thresholds live: how many row groups is "too many", how
small a row group is "too small", how poor a prunability estimate counts as
"effectively unsorted". They're deliberately conservative defaults, all
overridable, documented in the README rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .prunability import PrunabilityResult, estimate_prunability
from .types import ColumnChunkLayout, FileLayout, RowGroupLayout

_DICTIONARY_ENCODINGS = {"PLAIN_DICTIONARY", "RLE_DICTIONARY"}


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Finding:
    severity: Severity
    code: str
    file: str | None
    """``None`` for a dataset-level finding (spans multiple files)."""
    column: str | None
    message: str
    remedy: str


@dataclass(frozen=True)
class AuditOptions:
    max_row_groups: int = 200
    """Row groups beyond this count are flagged: per-group overhead (footer
    entries, open/seek cost) starts to dominate well before this in
    practice, but this stays conservative to avoid false positives on
    files that are merely big."""
    min_row_group_bytes: int = 8 * 1024 * 1024
    max_row_group_bytes: int = 1024 * 1024 * 1024
    large_file_min_rows: int = 500_000
    """A single row group is only flagged as "too large" on its own (rather
    than needing to also exceed ``max_row_group_bytes``) once the file has
    at least this many rows -- a small file with one row group is normal,
    not a layout problem."""
    small_file_bytes: int = 16 * 1024 * 1024
    small_file_fraction: float = 0.3
    """Dataset-level: flag "too many small files" once at least this
    fraction of the given files are under ``small_file_bytes``."""
    low_cardinality_ratio: float = 0.05
    low_cardinality_max_distinct: int = 10_000
    poor_prunability_fraction: float = 0.1
    """Below this fraction of row groups pruned on average -> WARNING
    ("effectively unsorted")."""
    partial_prunability_fraction: float = 0.5
    """Below this (and at/above ``poor_prunability_fraction``) -> INFO
    ("partially clustered")."""
    tiny_page_bytes: int = 8 * 1024
    huge_page_bytes: int = 32 * 1024 * 1024


@dataclass(frozen=True)
class RowGroupSizeSummary:
    count: int
    avg_rows: float
    avg_bytes: float
    min_bytes: int
    max_bytes: int


@dataclass(frozen=True)
class ColumnPrunabilityReport:
    column: str
    physical_type: str
    row_groups_with_stats: int
    row_groups_total: int
    prunability: PrunabilityResult
    has_dictionary_encoding: bool
    data_page_count: int | None


@dataclass(frozen=True)
class FileAuditReport:
    path: str
    file_size: int
    num_rows: int
    num_row_groups: int
    row_group_summary: RowGroupSizeSummary | None
    columns: tuple[ColumnPrunabilityReport, ...] = field(default_factory=tuple)
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def infos(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.INFO]


def _row_group_bytes(rg: RowGroupLayout) -> int:
    """On-disk (compressed) size when the writer recorded it, else the
    uncompressed in-memory size -- both are optional per the Parquet spec,
    but at least one is always present for a row group with any data."""
    if rg.total_compressed_size is not None and rg.total_compressed_size > 0:
        return rg.total_compressed_size
    return rg.total_byte_size


def _summarize_row_groups(layout: FileLayout) -> RowGroupSizeSummary | None:
    if not layout.row_groups:
        return None
    sizes = [_row_group_bytes(rg) for rg in layout.row_groups]
    rows = [rg.num_rows for rg in layout.row_groups]
    n = len(sizes)
    return RowGroupSizeSummary(
        count=n,
        avg_rows=sum(rows) / n,
        avg_bytes=sum(sizes) / n,
        min_bytes=min(sizes),
        max_bytes=max(sizes),
    )


def _columns_by_path(layout: FileLayout) -> dict[tuple[str, ...], list[ColumnChunkLayout]]:
    by_path: dict[tuple[str, ...], list[ColumnChunkLayout]] = {}
    for rg in layout.row_groups:
        for col in rg.columns:
            by_path.setdefault(col.path, []).append(col)
    return by_path


def _row_group_findings(layout: FileLayout, options: AuditOptions, summary: RowGroupSizeSummary) -> list[Finding]:
    findings: list[Finding] = []
    n = layout.num_row_groups

    if n > options.max_row_groups:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                code="PLA001",
                file=layout.path,
                column=None,
                message=(
                    f"{n} row groups averaging {summary.avg_rows:,.0f} rows / "
                    f"{summary.avg_bytes:,.0f} bytes each -- per-row-group overhead "
                    "(footer entries, per-group open/seek cost) dominates at this count"
                ),
                remedy=(
                    "increase row_group_size when writing (e.g. "
                    "pyarrow.parquet.write_table(..., row_group_size=500_000)) so each "
                    "row group amortizes overhead across more rows"
                ),
            )
        )

    if n == 1 and layout.num_rows >= options.large_file_min_rows:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                code="PLA002",
                file=layout.path,
                column=None,
                message=(
                    f"single row group covering all {layout.num_rows:,} rows -- no "
                    "row-group pruning granularity is possible for this file regardless "
                    "of sort order or filter selectivity"
                ),
                remedy=(
                    "reduce row_group_size when writing (e.g. row_group_size=500_000) so "
                    "a selective filter can skip whole row groups instead of always "
                    "reading the entire file"
                ),
            )
        )
    else:
        for i, rg in enumerate(layout.row_groups):
            size = _row_group_bytes(rg)
            if size > options.max_row_group_bytes:
                findings.append(
                    Finding(
                        severity=Severity.WARNING,
                        code="PLA002",
                        file=layout.path,
                        column=None,
                        message=(
                            f"row group {i} is {size:,} bytes ({rg.num_rows:,} rows) -- "
                            "larger than a single row group needs to be for pruning "
                            "granularity, and costly to decompress in full when it is "
                            "the one a filter can't rule out"
                        ),
                        remedy="reduce row_group_size when writing this dataset",
                    )
                )

    return findings


def _column_report_and_findings(
    layout: FileLayout, path: tuple[str, ...], chunks: list[ColumnChunkLayout], options: AuditOptions
) -> tuple[ColumnPrunabilityReport, list[Finding]]:
    findings: list[Finding] = []
    dotted = ".".join(path)
    physical_type = chunks[0].physical_type
    total = len(chunks)
    with_stats = sum(1 for c in chunks if c.statistics.has_stats)

    ranges = [(c.statistics.min_value, c.statistics.max_value) for c in chunks if c.statistics.has_stats]
    prunability = estimate_prunability(ranges, row_groups_total=layout.num_row_groups)

    has_dictionary = any(
        c.has_dictionary_page or bool(c.encodings & _DICTIONARY_ENCODINGS) for c in chunks
    )
    data_page_counts = [c.data_page_count for c in chunks if c.data_page_count is not None]
    total_data_pages = sum(data_page_counts) if len(data_page_counts) == total else None

    report = ColumnPrunabilityReport(
        column=dotted,
        physical_type=physical_type,
        row_groups_with_stats=with_stats,
        row_groups_total=total,
        prunability=prunability,
        has_dictionary_encoding=has_dictionary,
        data_page_count=total_data_pages,
    )

    if physical_type == "INT96":
        findings.append(
            Finding(
                severity=Severity.INFO,
                code="PLA003",
                file=layout.path,
                column=dotted,
                message="physical type INT96 (deprecated) has no reliably decodable statistics; pruning cannot be assessed for this column",
                remedy="migrate this column to a proper TIMESTAMP (INT64) logical type at write time -- INT96 is deprecated and most modern writers no longer produce it by default",
            )
        )
    elif with_stats < total:
        missing = total - with_stats
        if with_stats == 0:
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    code="PLA003",
                    file=layout.path,
                    column=dotted,
                    message="no row group has usable statistics for this column -- pruning is impossible for it, regardless of sort order",
                    remedy="write with statistics enabled (most writers, including pyarrow and parquet-mr, do this by default -- check that write_statistics wasn't explicitly disabled)",
                )
            )
        else:
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    code="PLA003",
                    file=layout.path,
                    column=dotted,
                    message=f"{missing} of {total} row groups have no usable statistics for this column -- those row groups can never be pruned regardless of sort order",
                    remedy="write with statistics enabled consistently for this column across the whole dataset",
                )
            )

    if prunability.applicable:
        n = prunability.row_groups_with_stats
        pruned = prunability.avg_pruned
        assert pruned is not None
        fraction = pruned / n if n else 0.0
        if fraction < options.poor_prunability_fraction:
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    code="PLA004",
                    file=layout.path,
                    column=dotted,
                    message=prunability.describe() + " -- effectively unsorted: an equality filter must open almost every row group",
                    remedy=f"sort by `{dotted}` before writing (or partition/cluster writes by it); this file currently prunes {pruned:.1f} of {n} row groups for that column",
                )
            )
        elif fraction < options.partial_prunability_fraction:
            findings.append(
                Finding(
                    severity=Severity.INFO,
                    code="PLA004",
                    file=layout.path,
                    column=dotted,
                    message=prunability.describe() + " -- partially clustered; a full sort would prune substantially more",
                    remedy=f"sort by `{dotted}` before writing for full pruning; currently prunes {pruned:.1f} of {n} row groups on average",
                )
            )

    if not has_dictionary:
        for c in chunks:
            dc = c.statistics.distinct_count
            if dc is None or c.num_values <= 0:
                continue
            ratio = dc / c.num_values
            if ratio <= options.low_cardinality_ratio or dc <= options.low_cardinality_max_distinct:
                findings.append(
                    Finding(
                        severity=Severity.INFO,
                        code="PLA005",
                        file=layout.path,
                        column=dotted,
                        message=f"low cardinality ({dc:,} distinct of {c.num_values:,} values in one row group) but not dictionary-encoded",
                        remedy="verify dictionary encoding wasn't disabled at write time (e.g. pyarrow's use_dictionary) -- dictionary encoding on a low-cardinality column typically shrinks storage and speeds scans",
                    )
                )
                break  # one representative finding per column, not one per row group

    if total_data_pages is not None and total_data_pages > 0:
        total_uncompressed = sum(c.total_uncompressed_size for c in chunks)
        avg_page_bytes = total_uncompressed / total_data_pages
        if avg_page_bytes < options.tiny_page_bytes:
            findings.append(
                Finding(
                    severity=Severity.INFO,
                    code="PLA006",
                    file=layout.path,
                    column=dotted,
                    message=f"average page size ~{avg_page_bytes:,.0f} bytes across {total_data_pages} data pages -- very small pages add per-page decode overhead",
                    remedy="increase the writer's target page size (e.g. pyarrow's data_page_size)",
                )
            )
        elif avg_page_bytes > options.huge_page_bytes:
            findings.append(
                Finding(
                    severity=Severity.INFO,
                    code="PLA006",
                    file=layout.path,
                    column=dotted,
                    message=f"average page size ~{avg_page_bytes:,.0f} bytes across {total_data_pages} data pages -- very large pages give page-level (sub-row-group) skipping almost no granularity",
                    remedy="decrease the writer's target page size (e.g. pyarrow's data_page_size)",
                )
            )

    return report, findings


def analyze_file(layout: FileLayout, options: AuditOptions | None = None) -> FileAuditReport:
    options = options or AuditOptions()
    summary = _summarize_row_groups(layout)
    findings: list[Finding] = []

    if summary is not None:
        findings.extend(_row_group_findings(layout, options, summary))

    columns: list[ColumnPrunabilityReport] = []
    for path, chunks in _columns_by_path(layout).items():
        report, col_findings = _column_report_and_findings(layout, path, chunks, options)
        columns.append(report)
        findings.extend(col_findings)

    return FileAuditReport(
        path=layout.path,
        file_size=layout.file_size,
        num_rows=layout.num_rows,
        num_row_groups=layout.num_row_groups,
        row_group_summary=summary,
        columns=tuple(columns),
        findings=tuple(findings),
    )


def analyze_dataset(layouts: list[FileLayout], options: AuditOptions | None = None) -> list[Finding]:
    """Cross-file findings that only make sense given more than one file."""
    options = options or AuditOptions()
    if len(layouts) < 2:
        return []

    small = [layout for layout in layouts if layout.file_size < options.small_file_bytes]
    if len(small) >= 2 and (len(small) / len(layouts)) >= options.small_file_fraction:
        return [
            Finding(
                severity=Severity.WARNING,
                code="PLA007",
                file=None,
                column=None,
                message=(
                    f"{len(small)} of {len(layouts)} files are under {options.small_file_bytes:,} bytes -- "
                    "per-file overhead (footer parsing, open/seek, scheduler task overhead) dominates at this size"
                ),
                remedy="compact small files into fewer, larger ones (e.g. a periodic compaction job), targeting on the order of a few hundred MB to a few GB per file",
            )
        ]
    return []
