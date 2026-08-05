"""The end-of-run summary and the machine-readable report.

Mostly about one trap: a failed file has an ``original_size`` but no ``new_size``,
because nothing was ever committed. Read raw, that is a file which compressed to
nothing -- so a run that failed outright would report the best rate it ever had.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from media_condenser import logging_setup, planner, report


def record(
    outcome: report.Outcome, original: int, new: int = 0, kind: planner.Kind = planner.Kind.IMAGE
) -> report.Record:
    return report.Record(
        path=Path(f"/lib/{outcome}.jpg"), kind=kind, outcome=outcome, original_size=original, new_size=new
    )


# ---------------------------------------------------------------------------
# size_after
# ---------------------------------------------------------------------------


def test_a_failed_file_still_occupies_its_original_size() -> None:
    """Nothing was committed, so the original is exactly where it was."""
    assert record(report.Outcome.FAILED, 1000).size_after == 1000
    assert record(report.Outcome.FAILED, 1000).saved == 0


def test_a_processed_file_occupies_its_new_size() -> None:
    assert record(report.Outcome.SUCCEEDED, 1000, 400).size_after == 400
    assert record(report.Outcome.DOWNGRADED, 1000, 400).size_after == 400


def test_a_skipped_file_is_unchanged() -> None:
    assert record(report.Outcome.SKIPPED, 1000, 1000).size_after == 1000
    assert record(report.Outcome.SKIPPED, 1000, 1000).saved == 0


# ---------------------------------------------------------------------------
# compression_rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (1000, 400, "60%"),
        (1000, 1000, "0%"),
        (0, 0, "-"),  # nothing ran; not the same as running and saving nothing
        (1000, 1200, "-20%"),  # only reachable if the size guard is ever relaxed
    ],
)
def test_compression_rate(before: int, after: int, expected: str) -> None:
    assert report.compression_rate(before, after) == expected


# ---------------------------------------------------------------------------
# The summary table
# ---------------------------------------------------------------------------


def test_the_totals_row_does_not_count_a_failure_as_a_saving(capsys) -> None:
    reporter = report.Reporter()
    reporter.add(record(report.Outcome.SUCCEEDED, 1000, 400))
    reporter.add(record(report.Outcome.FAILED, 9000))

    reporter.print_summary()
    out = capsys.readouterr().out

    # Before 10000, after 9400: the failure contributes its own size to both sides.
    assert report.human_bytes(10000) in out
    assert report.human_bytes(9400) in out
    assert "6%" in out  # 600 of 10000, not 96% as a raw new_size sum would give


def test_every_size_column_survives_an_eighty_column_pipe(capsys, monkeypatch) -> None:
    """The summary is what `mcon photos/ -o out/ > report.txt` is for, and rich falls back to
    80 columns when it cannot see a terminal. A truncated size reads as a smaller
    number, so the numeric columns must never be the ones that give."""
    monkeypatch.setattr(logging_setup.RESULT_CONSOLE, "_width", 80)
    reporter = report.Reporter()
    reporter.add(record(report.Outcome.SUCCEEDED, 823296, 301235, kind=planner.Kind.MOTION_PHOTO))
    reporter.add(record(report.Outcome.SUCCEEDED, 1170497, 485341, kind=planner.Kind.VIDEO))

    reporter.print_summary()
    out = capsys.readouterr().out

    assert "…" not in out
    for value in ("804.0 KB", "294.2 KB", "1.1 MB", "474.0 KB"):
        assert value in out, f"{value} missing from:\n{out}"


# ---------------------------------------------------------------------------
# The JSON report
# ---------------------------------------------------------------------------


def test_the_json_totals_are_internally_consistent() -> None:
    """`original - new == saved` has to hold, or a consumer computing its own rate
    from those two fields gets a different answer from the one reported."""
    reporter = report.Reporter()
    reporter.add(record(report.Outcome.SUCCEEDED, 1000, 400))
    reporter.add(record(report.Outcome.FAILED, 9000))
    reporter.add(record(report.Outcome.SKIPPED, 500, 500))

    totals = json.loads(json.dumps(reporter.as_dict()))["totals"]

    assert totals["original_bytes"] - totals["new_bytes"] == totals["saved_bytes"]
    assert totals["original_bytes"] == 10500
    assert totals["saved_bytes"] == 600
    assert totals["compression_rate"] == pytest.approx(600 / 10500, abs=1e-4)


def test_the_rate_is_null_rather_than_zero_when_nothing_ran() -> None:
    assert report.Reporter().as_dict()["totals"]["compression_rate"] is None
