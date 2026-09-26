from __future__ import annotations

import json
from pathlib import Path

import pytest

from parquet_layout_audit.cli import EXIT_ERRORS, EXIT_OK, EXIT_USAGE, EXIT_WARNINGS, run


def test_clean_file_exits_zero(sorted_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run([str(sorted_file)])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "No findings." in out


def test_missing_statistics_exits_with_error_code(no_statistics_file: Path) -> None:
    code = run([str(no_statistics_file)])
    assert code == EXIT_ERRORS


def test_fail_on_none_always_exits_zero(no_statistics_file: Path) -> None:
    code = run([str(no_statistics_file), "--fail-on", "none"])
    assert code == EXIT_OK


def test_fail_on_warning_treats_unsorted_data_as_nonzero(unsorted_file: Path) -> None:
    code = run([str(unsorted_file), "--fail-on", "warning"])
    assert code == EXIT_WARNINGS


def test_json_output_is_well_formed(unsorted_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run([str(unsorted_file), "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert len(payload["files"]) == 1
    file_report = payload["files"][0]
    assert file_report["num_row_groups"] == 20
    assert any(c["column"] == "event_id" for c in file_report["columns"])
    assert payload["summary"]["warnings"] >= 1


def test_json_includes_prunability_fields(sorted_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run([str(sorted_file), "--json"])
    payload = json.loads(capsys.readouterr().out)
    col = payload["files"][0]["columns"][0]
    assert col["prunability"]["applicable"] is True
    assert col["prunability"]["avg_pruned"] > 15


def test_unreadable_file_is_reported_but_does_not_crash(
    sorted_file: Path, bogus_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run([str(sorted_file), str(bogus_file), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert str(bogus_file) in payload["unreadable_files"]
    assert code == EXIT_OK  # the one readable file has no findings


def test_no_readable_files_is_a_usage_error(bogus_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run([str(bogus_file)])
    assert code == EXIT_USAGE


def test_multiple_small_files_produce_dataset_level_finding(
    many_small_files: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = sorted(str(p) for p in many_small_files.glob("*.parquet"))
    code = run([*files, "--json", "--small-file-bytes", str(1024 * 1024), "--fail-on", "warning"])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["dataset_findings"]) == 1
    assert code == EXIT_WARNINGS  # small-files finding is a warning
