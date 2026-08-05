"""Creation date resolution.

Metadata always wins. Filename inference is a strict fallback for files that carry
no date at all -- filenames are not a reliable source, and the fixtures show why:
``DJI_20200104_120000_1_null_video.mp4`` records ``CreateDate 03:00:00`` against a
filename saying ``120000``. It reproduces a real drone file whose name was in local
time while its metadata was in UTC, nine hours apart. Trusting the name there would
overwrite a correct timestamp with a wrong one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

#: Patterns seen across the mixed-source fixtures. Each must capture a
#: ``YYYYMMDD`` group and, optionally, ``HHMMSS``.
_FILENAME_PATTERNS = [
    # PXL_20200118_120000000.MP.jpg, IMG_20200109_120000_000.jpg, VID_20200106_120000_000.mp4
    re.compile(r"(?:PXL|IMG|VID|DJI)[_-](?P<date>\d{8})[_-](?P<time>\d{6})"),
    # Screenshot_20200110-120000-bearbeitet.png
    re.compile(r"Screenshot[_-](?P<date>\d{8})[_-](?P<time>\d{6})"),
    # VID-20200105-WA0001.mp4 -- WhatsApp, date only
    re.compile(r"(?:VID|IMG)-(?P<date>\d{8})-WA\d+"),
    # 2020-01-05 20.48.48.jpg and similar desktop exports
    re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})[ _T](?P<time>\d{2}[.\-:]\d{2}[.\-:]\d{2})"),
    # bare 20200105_204848
    re.compile(r"(?P<date>\d{8})[_-](?P<time>\d{6})"),
]

#: Creation-date tags, in the order they are trusted. Also what the readers ask
#: exiftool for, so adding a tag here is the whole change.
METADATA_KEYS = ("DateTimeOriginal", "CreateDate", "CreationDate", "MediaCreateDate")


@dataclass(frozen=True)
class DateResult:
    value: datetime | None
    source: str
    """``metadata``, ``filename`` or ``none``."""

    @property
    def exif_value(self) -> str:
        return self.value.strftime("%Y:%m:%d %H:%M:%S") if self.value else ""


def from_metadata(exif: dict[str, Any]) -> datetime | None:
    for key in METADATA_KEYS:
        raw = exif.get(key)
        if not raw:
            continue
        parsed = _parse_exif_datetime(str(raw))
        if parsed is not None:
            return parsed
    return None


def _parse_exif_datetime(raw: str) -> datetime | None:
    text = raw.strip()
    # Strip a trailing timezone offset; the local wall-clock time is what matters
    # for library ordering and it is what EXIF stores.
    text = re.sub(r"(?:[+-]\d{2}:?\d{2}|Z)$", "", text).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def from_filename(name: str) -> datetime | None:
    """Infer a date from the filename, or ``None`` if no pattern matches."""
    for pattern in _FILENAME_PATTERNS:
        match = pattern.search(name)
        if not match:
            continue
        groups = match.groupdict()
        date_text = groups["date"].replace("-", "")
        time_text = re.sub(r"[.\-:]", "", groups.get("time") or "000000")
        try:
            return datetime.strptime(f"{date_text}{time_text[:6]}", "%Y%m%d%H%M%S")
        except ValueError:
            continue
    return None


def resolve(path: Path, exif: dict[str, Any]) -> DateResult:
    """Best available creation date for a file."""
    metadata = from_metadata(exif)
    if metadata is not None:
        return DateResult(value=metadata, source="metadata")
    inferred = from_filename(path.name)
    if inferred is not None:
        return DateResult(value=inferred, source="filename")
    return DateResult(value=None, source="none")


def build_date_write_command(target: Path, result: DateResult, exiftool: str) -> list[str] | None:
    """An exiftool command writing an inferred date, or ``None`` if not needed.

    Only ever called for the ``filename`` source: a file that already had a date in
    its metadata keeps it untouched.
    """
    if result.source != "filename" or result.value is None:
        return None
    stamp = result.exif_value
    return [
        exiftool,
        "-quiet",
        "-overwrite_original",
        f"-DateTimeOriginal={stamp}",
        f"-CreateDate={stamp}",
        f"-ModifyDate={stamp}",
        str(target),
    ]
