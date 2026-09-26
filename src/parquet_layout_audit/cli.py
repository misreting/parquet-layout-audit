"""Command-line entry point: ``parquet-layout-audit data/*.parquet``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from .analyze import AuditOptions, Finding, FileAuditReport, Severity, analyze_dataset, analyze_file
from .footer import ParquetFooterError, read_layout

EXIT_OK = 0
EXIT_WARNINGS = 1
EXIT_ERRORS = 2
EXIT_USAGE = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parquet-layout-audit",
        description=(
            "Audit Parquet files for row-group sizing, missing statistics, and "
            "unsorted data that defeats row-group pruning."
        ),
    )
    parser.add_argument("files", nargs="+", help="Parquet files to audit (shell-expanded globs work fine).")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text.")
    parser.add_argument(
        "--fail-on",
        choices=("error", "warning", "none"),
        default="error",
        help="Minimum severity that causes a non-zero exit code (default: error).",
    )
    parser.add_argument(
        "--max-row-groups", type=int, default=AuditOptions.max_row_groups,
        help="Row groups beyond this count are flagged as too many (default: %(default)s).",
    )
    parser.add_argument(
        "--max-row-group-bytes", type=int, default=AuditOptions.max_row_group_bytes,
        help="Row groups above this size (bytes) are flagged as too large (default: %(default)s).",
    )
    parser.add_argument(
        "--small-file-bytes", type=int, default=AuditOptions.small_file_bytes,
        help="Files under this size (bytes) count toward the too-many-small-files check (default: %(default)s).",
    )
    parser.add_argument(
        "--poor-prunability-fraction", type=float, default=AuditOptions.poor_prunability_fraction,
        help="Below this fraction of row groups pruned on average, a column is flagged as effectively unsorted (default: %(default)s).",
    )
    parser.add_argument(
        "--partial-prunability-fraction", type=float, default=AuditOptions.partial_prunability_fraction,
        help="Below this fraction (and at/above --poor-prunability-fraction), a column is flagged as partially clustered (default: %(default)s).",
    )
    return parser


def _options_from_args(args: argparse.Namespace) -> AuditOptions:
    return AuditOptions(
        max_row_groups=args.max_row_groups,
        max_row_group_bytes=args.max_row_group_bytes,
        small_file_bytes=args.small_file_bytes,
        poor_prunability_fraction=args.poor_prunability_fraction,
        partial_prunability_fraction=args.partial_prunability_fraction,
    )


def _severity_label(sev: Severity) -> str:
    return sev.value.upper()


def _finding_to_dict(finding: Finding) -> dict[str, Any]:
    return {
        "severity": finding.severity.value,
        "code": finding.code,
        "file": finding.file,
        "column": finding.column,
        "message": finding.message,
        "remedy": finding.remedy,
    }


def _report_to_dict(report: FileAuditReport) -> dict[str, Any]:
    summary = report.row_group_summary
    return {
        "path": report.path,
        "file_size": report.file_size,
        "num_rows": report.num_rows,
        "num_row_groups": report.num_row_groups,
        "row_group_summary": None
        if summary is None
        else {
            "count": summary.count,
            "avg_rows": summary.avg_rows,
            "avg_bytes": summary.avg_bytes,
            "min_bytes": summary.min_bytes,
            "max_bytes": summary.max_bytes,
        },
        "columns": [
            {
                "column": c.column,
                "physical_type": c.physical_type,
                "row_groups_with_stats": c.row_groups_with_stats,
                "row_groups_total": c.row_groups_total,
                "has_dictionary_encoding": c.has_dictionary_encoding,
                "data_page_count": c.data_page_count,
                "prunability": {
                    "applicable": c.prunability.applicable,
                    "overlap_fraction": c.prunability.overlap_fraction,
                    "avg_row_groups_touched": c.prunability.avg_row_groups_touched,
                    "avg_pruned": c.prunability.avg_pruned,
                    "fully_partitioned": c.prunability.fully_partitioned,
                },
            }
            for c in report.columns
        ],
        "findings": [_finding_to_dict(f) for f in report.findings],
    }


def _print_text(reports: list[FileAuditReport], dataset_findings: list[Finding], unreadable: dict[str, str]) -> None:
    for report in reports:
        print(f"\n{report.path}")
        print(f"  {report.file_size:,} bytes, {report.num_rows:,} rows, {report.num_row_groups} row group(s)")
        summary = report.row_group_summary
        if summary is not None:
            print(
                f"  row groups: avg {summary.avg_rows:,.0f} rows / {summary.avg_bytes:,.0f} bytes "
                f"(min {summary.min_bytes:,}, max {summary.max_bytes:,})"
            )
        print("  columns:")
        for c in sorted(report.columns, key=lambda c: c.column):
            stats_note = f"{c.row_groups_with_stats}/{c.row_groups_total} have stats"
            dict_note = "dictionary" if c.has_dictionary_encoding else "no dictionary"
            print(f"    {c.column} ({c.physical_type}): {stats_note}, {dict_note}")
            print(f"      pruning: {c.prunability.describe()}")

        if not report.findings:
            print("  No findings.")
        else:
            print(
                f"  Findings ({len(report.errors)} error, {len(report.warnings)} warning, {len(report.infos)} info):"
            )
            for finding in report.findings:
                label = f"{finding.code} {finding.column}" if finding.column else finding.code
                print(f"    [{_severity_label(finding.severity)}] {label}: {finding.message}")
                print(f"        remedy: {finding.remedy}")

    if unreadable:
        print("\nCould not read:")
        for f, err in unreadable.items():
            print(f"  {f}: {err}")

    if dataset_findings:
        print("\nDataset-level findings:")
        for finding in dataset_findings:
            print(f"  [{_severity_label(finding.severity)}] {finding.code}: {finding.message}")
            print(f"      remedy: {finding.remedy}")


def _all_findings(reports: list[FileAuditReport], dataset_findings: list[Finding]) -> list[Finding]:
    findings: list[Finding] = []
    for report in reports:
        findings.extend(report.findings)
    findings.extend(dataset_findings)
    return findings


def run(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    options = _options_from_args(args)

    layouts = []
    unreadable: dict[str, str] = {}
    for path in args.files:
        try:
            layouts.append(read_layout(path))
        except (ParquetFooterError, OSError) as exc:
            unreadable[path] = str(exc)

    if not layouts:
        if args.json:
            print(json.dumps({"error": "no readable files", "unreadable_files": unreadable}, indent=2))
        else:
            print("No files could be read.", file=sys.stderr)
            for f, err in unreadable.items():
                print(f"  {f}: {err}", file=sys.stderr)
        return EXIT_USAGE

    reports = [analyze_file(layout, options) for layout in layouts]
    dataset_findings = analyze_dataset(layouts, options)
    all_findings = _all_findings(reports, dataset_findings)

    if args.json:
        payload = {
            "files": [_report_to_dict(r) for r in reports],
            "dataset_findings": [_finding_to_dict(f) for f in dataset_findings],
            "unreadable_files": unreadable,
            "summary": {
                "errors": sum(1 for f in all_findings if f.severity is Severity.ERROR),
                "warnings": sum(1 for f in all_findings if f.severity is Severity.WARNING),
                "infos": sum(1 for f in all_findings if f.severity is Severity.INFO),
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_text(reports, dataset_findings, unreadable)

    if args.fail_on == "none":
        return EXIT_OK
    if any(f.severity is Severity.ERROR for f in all_findings):
        return EXIT_ERRORS
    if args.fail_on == "warning" and any(f.severity is Severity.WARNING for f in all_findings):
        return EXIT_WARNINGS
    return EXIT_OK


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
