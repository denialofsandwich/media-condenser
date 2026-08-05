"""Creation date resolution tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import fixtures
import pytest

from media_condenser import dates


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (fixtures.MP_FULL_COMBO, datetime(2020, 1, 18, 12, 0, 0)),
        (fixtures.IMAGE_ALREADY_SMALL, datetime(2020, 1, 9, 12, 0, 0)),
        (fixtures.VIDEO_ALREADY_SMALL, datetime(2020, 1, 6, 12, 0, 0)),
        (fixtures.SCREENSHOT_PNG, datetime(2020, 1, 10, 12, 0, 0)),
        (fixtures.VIDEO_WITH_GPS, datetime(2020, 1, 4, 12, 0, 0)),
        # WhatsApp exports carry no time component at all.
        (fixtures.VIDEO_SQUARE, datetime(2020, 1, 5, 0, 0, 0)),
    ],
)
def test_filename_patterns_across_sources(name: str, expected: datetime) -> None:
    assert dates.from_filename(name) == expected


@pytest.mark.parametrize("name", [fixtures.VIDEO_ROTATED_PLUS90, "holiday.jpg", "DSC_0001.jpg"])
def test_unrecognised_names_infer_nothing(name: str) -> None:
    """Guessing is worse than admitting there is no date."""
    assert dates.from_filename(name) is None


def test_metadata_wins_over_the_filename() -> None:
    """The GPS clip is the reason this ordering matters.

    Its filename says 12:00:00 while its metadata says 03:00:00 -- the fixture
    reproduces a real drone file whose name was in local time and whose metadata was
    in UTC. Preferring the filename would overwrite a correct timestamp with a wrong
    one, so the disagreement is deliberate.
    """
    path = fixtures.VIDEOS / fixtures.VIDEO_WITH_GPS
    result = dates.resolve(path, {"CreateDate": "2020:01:04 03:00:00"})
    assert result.source == "metadata"
    assert result.value == datetime(2020, 1, 4, 3, 0, 0)
    assert dates.from_filename(path.name) != result.value


def test_the_fixture_really_does_disagree_with_its_own_name() -> None:
    """Guards the fixture, not just the function.

    If regeneration ever stamped the metadata date to match the filename, the test
    above would still pass while no longer testing anything.
    """
    import subprocess

    stamped = subprocess.run(
        ["exiftool", "-s3", "-CreateDate", str(fixtures.VIDEOS / fixtures.VIDEO_WITH_GPS)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert stamped == "2020:01:04 03:00:00"
    assert dates.from_filename(fixtures.VIDEO_WITH_GPS) == datetime(2020, 1, 4, 12, 0, 0)


def test_filename_is_used_only_when_metadata_is_absent() -> None:
    path = fixtures.VIDEOS / fixtures.VIDEO_ALREADY_SMALL
    result = dates.resolve(path, {})
    assert result.source == "filename"
    assert result.exif_value == "2020:01:06 12:00:00"


def test_no_date_anywhere_is_reported_honestly() -> None:
    assert dates.resolve(Path("/tmp/holiday.jpg"), {}).source == "none"


def test_timezone_suffixes_are_tolerated() -> None:
    result = dates.resolve(Path("x.jpg"), {"CreateDate": "2020:01:04 03:00:00+02:00"})
    assert result.value == datetime(2020, 1, 4, 3, 0, 0)


def test_empty_metadata_values_are_ignored() -> None:
    """exiftool returns empty strings for absent tags in some modes."""
    result = dates.resolve(Path(fixtures.MP_FULL_COMBO), {"CreateDate": "", "DateTimeOriginal": None})
    assert result.source == "filename"
