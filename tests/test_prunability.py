from __future__ import annotations

from parquet_layout_audit.prunability import estimate_prunability


def test_fully_disjoint_ranges_prune_almost_everything() -> None:
    # 10 row groups, each covering a disjoint decade: [0,9],[10,19],...
    ranges = [(i * 10, i * 10 + 9) for i in range(10)]
    result = estimate_prunability(ranges, row_groups_total=10)
    assert result.applicable
    assert result.fully_partitioned is True
    assert result.overlap_fraction == 0.0
    assert result.avg_row_groups_touched == 1.0
    assert result.avg_pruned == 9.0


def test_fully_overlapping_ranges_prune_nothing() -> None:
    # every row group's range covers the entire domain
    ranges = [(0, 1000) for _ in range(10)]
    result = estimate_prunability(ranges, row_groups_total=10)
    assert result.applicable
    assert result.fully_partitioned is False
    assert result.overlap_fraction == 1.0
    assert result.avg_row_groups_touched == 10.0
    assert result.avg_pruned == 0.0


def test_shuffled_data_is_not_underestimated_by_edge_clustering() -> None:
    """Regression test for the bias in an earlier "probe at each row
    group's own minimum" design: when every row group is a uniform random
    sample of the same domain, every range's min clusters near the global
    minimum and every max clusters near the global maximum, so *every*
    pair of ranges overlaps -- this must report ~0 pruning, not a
    misleadingly optimistic partial score.
    """
    # Mimics what 20 uniformly-shuffled row groups of a 0..2,000,000 column
    # actually look like: small mins (all within the first ~50 values) and
    # large maxes (all within the last ~100 values of the domain).
    mins = [3, 18, 45, 15, 8, 0, 4, 17, 11, 5, 12, 9, 1, 10, 7, 2, 20, 25, 13, 19]
    maxes = [
        1999983, 1999996, 1999986, 1999984, 1999994, 1999987, 1999997, 1999963,
        1999990, 1999992, 1999988, 1999909, 1999998, 1999944, 1999982, 1999995,
        1999975, 1999999, 1999951, 1999968,
    ]
    ranges = list(zip(mins, maxes, strict=True))
    result = estimate_prunability(ranges, row_groups_total=20)
    assert result.applicable
    assert result.overlap_fraction == 1.0
    assert result.avg_pruned == 0.0


def test_partial_overlap_lands_between_the_extremes() -> None:
    # Two clusters of 5 disjoint-from-each-other-cluster ranges, but each
    # cluster's 5 ranges all overlap each other (e.g. daily-batched but
    # unsorted-within-day data).
    cluster_a = [(0, 100) for _ in range(5)]
    cluster_b = [(1000, 1100) for _ in range(5)]
    ranges = cluster_a + cluster_b
    result = estimate_prunability(ranges, row_groups_total=10)
    assert result.applicable
    assert 0.0 < result.overlap_fraction < 1.0
    assert 0.0 < result.avg_pruned < 9.0


def test_string_ranges_are_supported_without_subtraction() -> None:
    ranges: list[tuple[object, object]] = [
        (b"aaa", b"aaz"),
        (b"bbb", b"bbz"),
        (b"ccc", b"ccz"),
    ]
    result = estimate_prunability(ranges, row_groups_total=3)
    assert result.applicable
    assert result.fully_partitioned is True
    assert result.avg_pruned == 2.0


def test_fewer_than_two_row_groups_with_stats_is_not_applicable() -> None:
    assert estimate_prunability([], row_groups_total=5).applicable is False
    assert estimate_prunability([(1, 2)], row_groups_total=5).applicable is False


def test_single_row_group_file_reports_not_applicable_via_total() -> None:
    result = estimate_prunability([], row_groups_total=1)
    assert result.applicable is False
    assert "not applicable" in result.summary(column="x")


def test_mixed_incomparable_types_does_not_crash() -> None:
    ranges: list[tuple[object, object]] = [(1, 2), (b"a", b"b")]
    result = estimate_prunability(ranges, row_groups_total=2)
    assert result.applicable is False


def test_summary_text_mentions_prune_count_and_total() -> None:
    ranges = [(i * 10, i * 10 + 9) for i in range(4)]
    result = estimate_prunability(ranges, row_groups_total=4)
    text = result.summary(column="event_date")
    assert "event_date" in text
    assert "of 4 row groups" in text
