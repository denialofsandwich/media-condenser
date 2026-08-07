"""Post-processing verification.

Re-reads finished output as an independent reader would: re-parse the container,
re-extract every embedded component, confirm each decodes, and confirm the original
metadata survived.

**What this cannot prove.** For motion photos this establishes structural and
spec-level correctness only. Whether Google Photos or a phone gallery still renders
HDR and plays the motion clip can only be confirmed by opening the result on a real
device. :func:`caveat` returns that statement so the CLI always says it out loud
rather than letting a row of green ticks imply more than was tested.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from media_condenser import config, container, planner, probe

log = logging.getLogger(__name__)

MOTION_PHOTO_CAVEAT = (
    "Motion photo checks prove structural correctness only. Confirming HDR and "
    "motion still work requires opening the output on a real device."
)


def caveat(actions: list[planner.Action]) -> str | None:
    """The device caveat, when any motion photo was involved."""
    if any(action.kind is planner.Kind.MOTION_PHOTO and action.will_process for action in actions):
        return MOTION_PHOTO_CAVEAT
    return None


@dataclass
class VerifyReport:
    path: Path
    checks: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Derived rather than stored, so a problem cannot be recorded without it.

        The two used to be separate fields kept in step by hand, which meant any
        path appending to ``problems`` directly would leave ``ok`` true -- and an
        output that failed a check would be committed.
        """
        return not self.problems

    def passed(self, label: str) -> None:
        self.checks.append(label)

    def failed(self, label: str) -> None:
        self.problems.append(label)


def verify_output(
    source: Path,
    output: Path,
    kind: planner.Kind,
    prober: probe.Prober,
    cfg: config.GlobalConfig,
    *,
    before: Mapping[str, Any] | None = None,
) -> VerifyReport:
    """Verify one processed file against its original.

    ``output`` is normally the uncommitted temp file: :mod:`media_condenser.pipeline` calls this
    before the rename, so that a rejected output can simply not be committed. The
    original is therefore still on disk and its metadata is read here.

    ``before`` covers the case where it is not -- a caller checking an output that has
    already replaced its own source, where ``output`` and ``source`` are one path and
    reading it would compare the finished file against itself. Without a snapshot
    that comparison is reported as impossible rather than quietly passing.
    """
    report = VerifyReport(path=output)

    if not output.exists():
        report.failed("output file is missing")
        return report

    # Planning cached this path's measurements from before the encode -- and under
    # `replace` that is the very path being checked. Verification exists to read what
    # was actually written, so every probe below has to reach the disk.
    prober.forget(output)

    if kind is planner.Kind.VIDEO:
        _verify_video(output, report, prober)
    elif kind is planner.Kind.MOTION_PHOTO:
        _verify_motion_photo(output, report, prober, cfg)
    else:
        _verify_image(output, report, prober)

    _verify_metadata(source, output, report, prober, before)
    return report


def _verify_image(output: Path, report: VerifyReport, prober: probe.Prober) -> None:
    try:
        info = prober.image_info(output)
    except probe.ProbeError as exc:
        report.failed(f"image unreadable: {exc}")
        return
    report.passed(f"image decodes ({info.width}x{info.height})")


def _verify_video(output: Path, report: VerifyReport, prober: probe.Prober) -> None:
    try:
        info = prober.video_info(output)
    except probe.ProbeError as exc:
        report.failed(f"video unreadable: {exc}")
        return
    width, height = info.display_size
    report.passed(f"video decodes ({width}x{height}, {info.codec})")
    if info.codec == "hevc":
        report.passed("codec is H.265")


def _verify_motion_photo(output: Path, report: VerifyReport, prober: probe.Prober, cfg: config.GlobalConfig) -> None:
    data = output.read_bytes()
    exif = prober.exif(output)
    layout = container.resolve_layout(
        output,
        data,
        mpf_offset=probe.as_int(exif.get("MPImageStart")),
        mpf_length=probe.as_int(exif.get("MPImageLength")),
    )

    if not layout.is_motion_photo:
        report.failed("GCamera:MotionPhoto flag did not survive")
    else:
        report.passed("motion photo flag present")

    if layout.primary is None:
        report.failed("primary image could not be located")
        return

    # Every declared component is re-extracted and independently decoded, because a
    # correct-looking length is not evidence that the bytes are usable.
    if layout.gain_map is not None:
        blob = data[layout.gain_map.offset : layout.gain_map.end]
        if _decodes_as_image(blob, cfg):
            report.passed(f"gain map re-extracts and decodes ({layout.gain_map.length} bytes)")
        else:
            report.failed("gain map does not decode")

    if layout.video is not None:
        blob = data[layout.video.offset : layout.video.end]
        if _decodes_as_video(blob, cfg):
            report.passed(f"embedded video re-extracts and decodes ({layout.video.length} bytes)")
        else:
            report.failed("embedded video does not decode")

    for note in layout.notes:
        report.failed(f"re-parse reported: {note}")

    total = layout.primary.length + sum(
        component.length for component in (layout.gain_map, layout.video) if component is not None
    )
    if total == len(data):
        report.passed("declared component lengths account for the whole file")
    else:
        report.failed(f"declared lengths total {total} but the file is {len(data)} bytes")


def _verify_metadata(
    source: Path,
    output: Path,
    report: VerifyReport,
    prober: probe.Prober,
    before: Mapping[str, Any] | None,
) -> None:
    """Confirm the timestamps and camera identity survived processing."""
    if before is None and source == output:
        # Reading `source` here would read the finished file, so every tag would
        # trivially "match" itself. Say the check could not be made.
        report.failed("metadata not comparable: the original was overwritten and no pre-processing snapshot was taken")
        return

    try:
        if before is None:
            before = prober.exif(source)
        after = prober.exif(output)
    except probe.ProbeError as exc:
        report.failed(f"metadata unreadable: {exc}")
        return

    for key in ("CreateDate", "DateTimeOriginal", "Model", "Make", "GPSCoordinates", "GPSPosition"):
        original = before.get(key)
        if original in (None, ""):
            continue
        current = after.get(key)
        if current in (None, ""):
            report.failed(f"{key} was lost")
        elif _values_match(key, original, current):
            report.passed(f"{key} preserved")
        else:
            report.failed(f"{key} changed from {original!r} to {current!r}")


#: Coordinate agreement threshold in degrees, about 11 metres.
_GPS_TOLERANCE = 1e-4


def _values_match(key: str, original: object, current: object) -> bool:
    """Compare two tag values, numerically for coordinates.

    Location tags are re-serialized with different decimal precision on a
    round-trip ("37.5021" vs "37.50209" -- roughly a metre apart), so a string
    comparison reports a loss that did not happen. Comparing the numbers keeps the
    check meaningful: a genuinely dropped or corrupted coordinate is nowhere near
    the original, while a rounding difference is.
    """
    if str(original) == str(current):
        return True
    if "GPS" not in key:
        return False
    before_numbers = _numbers(str(original))
    after_numbers = _numbers(str(current))
    if not before_numbers or len(before_numbers) != len(after_numbers):
        return False
    return all(abs(a - b) <= _GPS_TOLERANCE for a, b in zip(before_numbers, after_numbers, strict=True))


def _numbers(text: str) -> list[float]:
    return [float(match) for match in re.findall(r"-?\d+\.?\d*", text)]


def _decodes(argv: list[str], blob: bytes, *, require_exit_zero: bool) -> bool:
    """Whether the tool in ``argv`` reads ``blob`` off stdin and describes it.

    Output on stdout is the evidence: both tools stay quiet there when they cannot
    make sense of what they were handed.

    ``require_exit_zero`` is deliberately not the same for both callers. ffprobe
    exits non-zero over a stream it nonetheless identified -- a truncated tail is
    enough -- and the codec name it printed is the answer being asked for, so
    demanding a clean exit there would reject embedded clips that do decode.
    """
    log.debug("$ %s  (%d bytes on stdin)", shlex.join(argv), len(blob))
    result = subprocess.run(argv, input=blob, capture_output=True, check=False)
    if result.returncode:
        log.debug("%s exit %d\n%s", Path(argv[0]).name, result.returncode, probe.decode(result.stderr))
        if require_exit_zero:
            return False
    return bool(result.stdout.strip())


def _decodes_as_image(blob: bytes, cfg: config.GlobalConfig) -> bool:
    return _decodes([cfg.tools.magick, "-", "-format", "%wx%h", "info:"], blob, require_exit_zero=True)


def _decodes_as_video(blob: bytes, cfg: config.GlobalConfig) -> bool:
    # Hoisted into a local so the flag-and-value pairing below can be preserved: a
    # `fmt: off` comment is only valid between statements, not inside a call's
    # argument list.
    # fmt: off
    argv = [
        cfg.tools.ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "csv=p=0",
        "-",
    ]
    # fmt: on
    return _decodes(argv, blob, require_exit_zero=False)
