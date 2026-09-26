"""Audit Parquet files for row-group sizing, missing statistics, and unsorted
data that defeats row-group pruning.

```python
from parquet_layout_audit import read_layout, analyze_file

layout = read_layout("data.parquet")
report = analyze_file(layout)
for finding in report.findings:
    print(finding.severity.value, finding.code, finding.column, finding.message)
```
"""

from __future__ import annotations

from .analyze import (
    AuditOptions,
    ColumnPrunabilityReport,
    Finding,
    FileAuditReport,
    RowGroupSizeSummary,
    Severity,
    analyze_dataset,
    analyze_file,
)
from .footer import ParquetFooterError, read_layout
from .prunability import PrunabilityResult, estimate_prunability
from .types import ColumnChunkLayout, ColumnStatistics, FileLayout, RowGroupLayout

__all__ = [
    "AuditOptions",
    "ColumnChunkLayout",
    "ColumnPrunabilityReport",
    "ColumnStatistics",
    "Finding",
    "FileAuditReport",
    "FileLayout",
    "ParquetFooterError",
    "PrunabilityResult",
    "RowGroupLayout",
    "RowGroupSizeSummary",
    "Severity",
    "analyze_dataset",
    "analyze_file",
    "estimate_prunability",
    "read_layout",
]
