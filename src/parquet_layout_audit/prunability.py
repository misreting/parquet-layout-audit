"""Estimate how prunable a column is, from its row groups' min/max statistics.

Row-group pruning works like this: a reader holding a predicate ``col = v``
(or a range) can skip a whole row group without decompressing it, as long
as the group's ``[min, max]`` statistics prove ``v`` can't be inside. That
guarantee only pays off when row groups' ranges are mostly disjoint. If
every row group's range spans close to the whole column's domain --
because the data was written in random order -- then no value can ever be
ruled out anywhere, and every row group must be opened regardless of the
filter.

## Method

The estimate is built from **pairwise range overlap**: for every pair of
row groups, do their ``[min, max]`` ranges intersect at all? If two ranges
don't overlap, no filter value can ever match both -- a query hitting one
of them structurally can't also need the other. If they do overlap, there
exists *some* value that would force a reader to open both, and the more
pairs overlap, the more row groups a typical query ends up touching.

Concretely: let ``overlap_fraction`` be the fraction of all
:math:`\\binom{n}{2}` row-group pairs whose ranges intersect. A row group
touched by a query is, on average, touched alongside ``(n - 1) *
overlap_fraction`` others (each of the other ``n - 1`` row groups overlaps
it with probability ``overlap_fraction``, by definition of the fraction),
plus itself -- so the average number of row groups a single-value
equality filter must open is ``1 + (n - 1) * overlap_fraction``, and the
average number *pruned* is ``n`` minus that.

This needs only pairwise comparison (``<=``), not subtraction, so it works
identically for numbers, dates encoded as integers, and byte-string
columns alike -- there is no need to define a numeric "distance" between
two strings. It is computed in ``O(n log n)``: sort row groups by their
minimum, then for each row group binary-search how many later-sorted row
groups have a minimum at or below this one's maximum (those overlap it;
earlier scans of Parquet layout tools use ``O(n^2)`` here since footers
rarely carry more than a few thousand row groups, but this stays fast even
into the tens of thousands).

### Why not just probe each row group's own minimum?

An earlier version of this estimate simulated an equality filter at each
row group's own recorded minimum and counted how many ranges contained it.
That is *biased* whenever minimums and maximums cluster at opposite ends of
the domain -- which is exactly what happens for a column shuffled
uniformly at random into fixed-size row groups: each row group's minimum
tends to land near the global minimum (the smallest of many samples) and
its maximum near the global maximum, for the same reason a hand of cards
dealt from a shuffled deck has a good chance of holding both a very low and
a very high card. Probing only at each group's own minimum then measures a
value artificially close to the global minimum, which only a handful of
other groups' ranges also reach down to -- understating the true overlap.
Measured on 20 row groups of 2,000,000 uniformly shuffled integers (see
``examples/measure_evidence.py``), the own-minimum probe reported "prunes
9.5 of 20 row groups" (adds up to a plausible-looking partial win) where
the pairwise method correctly reports "prunes 0.0 of 20" -- confirmed
against DuckDB, which reads effectively the same bytes from this file as
from one with a single row group, i.e. no row-group pruning happens at all.

This is a structural estimate from statistics alone -- it says nothing
about which values a real workload's filters actually ask for, and it
provides no information when fewer than two row groups have usable
statistics for a column (:attr:`PrunabilityResult.applicable` is ``False``
in that case).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrunabilityResult:
    """Estimated pruning effectiveness for one column of one file."""

    applicable: bool
    """False when fewer than 2 row groups have usable statistics for this
    column -- there is nothing to prune relative to, or nothing to measure."""
    row_groups_with_stats: int
    row_groups_total: int
    overlap_fraction: float | None = None
    """Fraction of row-group range pairs whose [min, max] ranges intersect.
    0.0 means every row group's range is disjoint from every other's (the
    ideal case); 1.0 means every pair overlaps (no pair of row groups can
    ever be told apart by a single-value filter)."""
    avg_row_groups_touched: float | None = None
    """Average number of row groups an equality filter on an observed value
    would have to open, out of ``row_groups_with_stats``."""
    fully_partitioned: bool | None = None
    """True (``overlap_fraction == 0.0``) if every row group's range is
    disjoint from every other's -- the ideal case, where any equality or
    narrow range filter touches at most one row group."""

    @property
    def avg_pruned(self) -> float | None:
        if self.avg_row_groups_touched is None:
            return None
        return self.row_groups_with_stats - self.avg_row_groups_touched

    def describe(self) -> str:
        """A column-name-free description, for callers that already show the
        column name elsewhere (e.g. as a finding's own ``column`` field)."""
        if not self.applicable:
            if self.row_groups_total <= 1:
                return f"only {self.row_groups_total} row group -- pruning across row groups is not applicable"
            return (
                f"only {self.row_groups_with_stats} of {self.row_groups_total} row group(s) have "
                "usable statistics -- not enough to estimate pruning"
            )
        assert self.avg_row_groups_touched is not None and self.overlap_fraction is not None
        n = self.row_groups_with_stats
        pruned = self.avg_pruned
        assert pruned is not None
        return (
            f"prunes {pruned:.1f} of {n} row groups on average for an equality filter "
            f"on an observed value (row-group ranges overlap {self.overlap_fraction * 100:.0f}% of the time pairwise)"
        )

    def summary(self, *, column: str) -> str:
        return f"{column}: {self.describe()}"


def estimate_prunability(ranges: list[tuple[Any, Any]], *, row_groups_total: int) -> PrunabilityResult:
    """Estimate prunability for one column from its row groups' (min, max) pairs.

    ``ranges`` holds one ``(min, max)`` tuple per row group that has usable
    statistics for this column, in any order. ``row_groups_total`` is the
    file's total row group count (which may be larger than ``len(ranges)``
    if some row groups lack statistics for this column).
    """
    n = len(ranges)
    if n < 2:
        return PrunabilityResult(applicable=False, row_groups_with_stats=n, row_groups_total=row_groups_total)

    try:
        sorted_ranges = sorted(ranges, key=lambda r: r[0])
        mins = [r[0] for r in sorted_ranges]
        # A dry-run comparison to fail fast (via TypeError) on a mixed or
        # otherwise incomparable set of bounds, rather than partway through
        # the bisect calls below.
        _ = [mins[i] <= mins[i + 1] for i in range(len(mins) - 1)]
    except TypeError:
        # Mixed/incomparable value types across row groups -- shouldn't
        # happen for a well-formed file, but don't crash on a malformed one.
        return PrunabilityResult(applicable=False, row_groups_with_stats=n, row_groups_total=row_groups_total)

    overlapping_pairs = 0
    for i in range(n):
        max_i = sorted_ranges[i][1]
        # Rightmost index (exclusive) among mins[i+1:] that is <= max_i.
        # Every j in (i, k) has min_j <= max_i, and since the list is
        # sorted by min ascending and max_i >= min_i, that's exactly the
        # condition for range i and range j to overlap.
        k = bisect.bisect_right(mins, max_i, lo=i + 1)
        overlapping_pairs += max(0, k - (i + 1))

    total_pairs = n * (n - 1) // 2
    overlap_fraction = overlapping_pairs / total_pairs if total_pairs else 0.0
    avg_touched = 1 + (n - 1) * overlap_fraction

    return PrunabilityResult(
        applicable=True,
        row_groups_with_stats=n,
        row_groups_total=row_groups_total,
        overlap_fraction=overlap_fraction,
        avg_row_groups_touched=avg_touched,
        fully_partitioned=(overlapping_pairs == 0),
    )
