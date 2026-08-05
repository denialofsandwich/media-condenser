"""Wrappers around ``ffprobe`` and ``exiftool``.

Everything the planner decides is based on *measured* properties read here.
Nothing downstream is allowed to infer a media property from a filename or
extension -- that shortcut was the root cause of multiple real bugs.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import logging
import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from media_condenser import config, handlers

log = logging.getLogger(__name__)

# Magic byte prefixes, checked against file content rather than the extension.
_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"mif1", b"heim", b"heis"}

#: ISO base media brands that really are video. Deliberately generous, because an
#: unlisted brand is reported as unsupported rather than transcoded -- see
#: :func:`sniff_kind`.
_VIDEO_BRANDS = {
    b"isom", b"iso2", b"iso4", b"iso5", b"iso6", b"iso8",
    b"mp41", b"mp42", b"mp4v", b"avc1", b"av01", b"dash",
    b"qt  ", b"3gp4", b"3gp5", b"3gp6", b"3gp7", b"3g2a",
    b"M4V ", b"M4VH", b"M4VP", b"mmp4", b"MSNV", b"XAVC",
    b"CAEP", b"f4v ",
}  # fmt: skip


#: ``-fast2`` stops exiftool at the JPEG image data instead of scanning through to
#: the end of the file looking for a trailer, and skips parsing MakerNotes. Every tag
#: in :data:`_EXIF_TAGS` lives in the header -- the MPF offsets in APP2, the XMP
#: packet in APP1, the dimensions in the SOF marker -- so the results are unchanged
#: and only the reading is shorter. On a motion photo that is the difference between
#: reading a hundred kilobytes and reading the whole several-megabyte container,
#: which over a phone library on a network share is most of the planning phase.
#:
#: ``-fast3`` would go further and read only the header, but it returns none of these
#: tags at all.
_EXIF_OPTIONS = ("-json", "-n", "-b", "-fast2")

# fmt: off
_EXIF_TAGS = (
    "-ImageWidth",
    "-ImageHeight",
    "-CreateDate",
    "-DateTimeOriginal",
    "-MPImageStart",
    "-MPImageLength",
    "-Model",
    "-Make",
    "-GPSCoordinates",
    "-GPSPosition",
    "-XMP",
)
# fmt: on

# fmt: off
_FFPROBE_OPTIONS = (
    "-v", "error",
    "-print_format", "json",
    "-show_streams",
    "-show_format",
)
# fmt: on

EXIF_BATCH_SIZE = 200

_BASE64_PREFIX = "base64:"


class ProbeError(RuntimeError):
    """A probe tool failed or returned unusable output."""


def _run(argv: list[str]) -> str:
    """Run a probe tool. The synchronous counterpart to ``handlers.video.run_command``.

    Only the last line of stderr reaches the :class:`ProbeError` message, so the
    full text is logged before it is thrown away -- ffprobe's useful diagnostics are
    usually several lines above the one it ends on.
    """
    log.debug("$ %s", shlex.join(argv))
    result = subprocess.run(argv, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = decode(result.stderr)
        log.debug("%s exit %d\n%s", Path(argv[0]).name, result.returncode, stderr)
        raise ProbeError(f"{argv[0]} failed: {handlers.tail(stderr) or f'exit {result.returncode}'}")
    return result.stdout.decode("utf-8", "replace")


async def run_async(argv: list[str], *, stdin_text: str | None = None) -> tuple[int, bytes, bytes]:
    """Run a probe tool without blocking the event loop.

    Returns the exit status rather than raising, unlike :func:`_run`, because a
    *batched* read exits non-zero when any one of its files was unreadable while
    still reporting every file that was fine. Throwing there would discard a few
    hundred good results over one bad photo.
    """
    log.debug("$ %s", shlex.join(argv))
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin_text is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(stdin_text.encode() if stdin_text is not None else None)
    return process.returncode or 0, stdout, stderr


def decode(raw: bytes) -> str:
    """Subprocess output as text, for logging. Never raises on malformed bytes."""
    return raw.decode("utf-8", "replace").strip()


def _decode_binary_tag(value: object) -> str:
    """The text of a tag read under ``-b``.

    exiftool JSON-encodes a binary value as ``base64:...`` when it is not valid
    text, which a malformed XMP packet can be. Decoding it keeps such a file
    classifiable rather than reading as "no XMP at all" -- which for a motion photo
    means being planned as a plain image and losing its embedded components.
    """
    if not isinstance(value, str):
        return ""
    if not value.startswith(_BASE64_PREFIX):
        return value
    try:
        return base64.b64decode(value[len(_BASE64_PREFIX) :]).decode("utf-8", "replace")
    except ValueError:
        # binascii.Error is a ValueError. A tag we cannot decode is reported as
        # absent, which is what the pre-batch code did with an unreadable packet.
        return ""


def _in_argument_file(path: Path) -> bool:
    """Whether a path can be expressed in an exiftool ``-@`` argument file.

    Lines there are split on newlines, stripped of surrounding whitespace, and
    treated as comments when they begin with ``#``. A name that trips any of those
    cannot go in a batch -- so it does not, and falls back to the per-file probe
    instead. Reading the wrong file would be far worse than reading this one slowly.
    """
    text = str(path)
    return not any(c in text for c in "\n\r") and text == text.strip() and not text.startswith("#")


JPEG = "jpeg"
PNG = "png"
HEIF = "heif"
VIDEO = "video"
ISO_OTHER = "iso-bmff"
"""An ISO base media file that is not a recognised video brand -- an AVIF or CR3
still, or a container this tool has not been taught about."""

UNKNOWN = "unknown"


def as_int(value: object) -> int | None:
    """Coerce a JSON-decoded exiftool value to an int, or ``None``.

    Narrowed by ``isinstance`` rather than by catching ``TypeError`` from ``int()``,
    so the accepted shapes are stated rather than discovered at runtime -- exiftool
    returns numbers for some tags and quoted strings for others, depending on whether
    ``-n`` applies to that tag. ``bool`` is rejected explicitly because it is an
    ``int`` subclass and would otherwise coerce to 0/1 silently.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def sniff_kind(path: Path) -> str:
    """Identify a file by its leading bytes.

    Returns one of ``jpeg``/``png``/``heif``/``video``/``iso-bmff``/``unknown``. The
    fixture ``Screenshot_20200111-120000~2.png`` is a real JPEG behind a ``.png``
    name, so the extension is never consulted.

    An ISO base media file is only called a video when its ``ftyp`` brand says so.
    Treating every unrecognised brand as video is what would hand an AVIF or CR3
    *still* to the video planner, which transcodes it to H.265 and commits an MP4
    under the original name -- destroying it. An unlisted brand therefore comes back
    as :data:`ISO_OTHER`, which is reported as an unsupported type and copied through
    untouched. That is a visible, recoverable outcome, and adding a brand to
    :data:`_VIDEO_BRANDS` is the fix; silently mangling a still is neither.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(32)
    except OSError:
        return UNKNOWN

    if head.startswith(_JPEG_MAGIC):
        return JPEG
    if head.startswith(_PNG_MAGIC):
        return PNG
    # ISO base media: 4-byte size, then 'ftyp', then the brand.
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in _HEIF_BRANDS:
            return HEIF
        if brand in _VIDEO_BRANDS:
            return VIDEO
        return ISO_OTHER
    return UNKNOWN


@dataclass(frozen=True)
class VideoInfo:
    """Measured properties of a video's first video stream."""

    width: int
    height: int
    rotation: int
    codec: str
    has_audio: bool
    duration: float | None
    audio_codec: str | None = None

    @property
    def is_rotated_quarter(self) -> bool:
        """True when the display matrix swaps the visual axes."""
        return abs(self.rotation) % 180 == 90

    @cached_property
    def display_size(self) -> tuple[int, int]:
        """Dimensions *as the viewer sees them*, after autorotation.

        Many "1920x1080" files are portrait, rotated by a container-level display
        matrix. Deciding anything from the stored width/height scales the wrong
        axis; every downstream size decision must use this instead.
        """
        if self.is_rotated_quarter:
            return self.height, self.width
        return self.width, self.height

    @property
    def short_edge(self) -> int:
        return min(self.display_size)

    @property
    def long_edge(self) -> int:
        return max(self.display_size)


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int

    @property
    def long_edge(self) -> int:
        return max(self.width, self.height)

    @property
    def short_edge(self) -> int:
        return min(self.width, self.height)


class Prober:
    """Runs the external probe tools, caching per path."""

    def __init__(self, tools: config.ToolPaths) -> None:
        self._tools = tools
        self._ffprobe_cache: dict[Path, dict[str, Any]] = {}
        self._exif_cache: dict[Path, dict[str, Any]] = {}

    def forget(self, path: Path) -> None:
        """Drop every cached probe for one path.

        Caching by path is safe while planning, because nothing has written anything
        yet. Verification is the opposite situation: it re-reads a path whose bytes
        were just rewritten, and under the ``replace`` strategy that path *is* the one
        probed during planning. Without this, the post-encode checks would be answered
        from the pre-encode probe and could never fail.
        """
        self._ffprobe_cache.pop(path, None)
        self._exif_cache.pop(path, None)

    # -- bulk prefetch ---------------------------------------------------

    async def prefetch(
        self,
        *,
        exif: Sequence[Path] = (),
        video: Sequence[Path] = (),
        concurrency: int = 4,
        on_progress: Callable[[int], None] | None = None,
    ) -> None:
        """Fill the caches for many files at once, before anything asks for them.

        Planning probes every file it classifies, and doing that one lazy call at a
        time is what made a large library spend most of its wall clock forking
        interpreters rather than reading media: exiftool costs ~95 ms per launch and
        almost none of that is the file. Batching those reads took a measured 100-file
        sample from 9.4 s to 0.27 s. ffprobe has no batch mode, so videos are probed
        concurrently instead -- 3.7 s to 0.34 s over the same sample.

        Every part of this is optional by construction. A file this fails to cache is
        simply probed lazily by whichever call needs it, and reports its own failure
        exactly as it did before, so an unreadable file in a batch costs nothing but
        its own re-probe and a bad batch costs nothing but its own re-reads. That is
        also why this returns nothing: there is no error here for a caller to handle.

        ``on_progress`` is called with a count of files finished, matching
        :meth:`media_condenser.progress.RunProgress.advance`.
        """
        pending_exif = [path for path in exif if path not in self._exif_cache]
        pending_video = [path for path in video if path not in self._ffprobe_cache]
        if on_progress:
            # Anything already cached is already done, and the bar's total was set
            # from the full list rather than from what turned out to need work.
            done = (len(exif) - len(pending_exif)) + (len(video) - len(pending_video))
            if done:
                on_progress(done)

        batchable = [path for path in pending_exif if _in_argument_file(path)]
        if len(batchable) != len(pending_exif):
            log.debug("%d path(s) cannot go in an exiftool batch", len(pending_exif) - len(batchable))

        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def read_exif_batch(chunk: Sequence[Path]) -> None:
            async with semaphore:
                await self._fill_exif(chunk)
            if on_progress:
                on_progress(len(chunk))

        async def read_video(path: Path) -> None:
            async with semaphore:
                await self._fill_ffprobe(path)
            if on_progress:
                on_progress(1)

        await asyncio.gather(
            # strict=False: the final batch is short whenever the count is not a
            # multiple of the batch size, which is the normal case.
            *(read_exif_batch(chunk) for chunk in itertools.batched(batchable, EXIF_BATCH_SIZE, strict=False)),
            *(read_video(path) for path in pending_video),
        )

        # Reported after the fact rather than skipped: these still count towards the
        # phase total, and their lazy probe is the fallback working as intended.
        if on_progress and (unbatchable := len(pending_exif) - len(batchable)):
            on_progress(unbatchable)

    async def _fill_exif(self, paths: Sequence[Path]) -> None:
        """Read one batch of files with a single exiftool process."""
        lines = [*_EXIF_OPTIONS, *_EXIF_TAGS, *(str(path) for path in paths)]
        # Arguments arrive on stdin (``-@ -``) rather than on the command line, where
        # a few hundred paths would risk ARG_MAX, and rather than via a temp file,
        # which would need cleaning up on every failure path.
        code, stdout, stderr = await run_async(
            [self._tools.exiftool, "-@", "-"],
            stdin_text="\n".join(lines) + "\n",
        )
        if code != 0:
            # Expected rather than exceptional -- see :func:`run_async`.
            log.debug("exiftool exit %d over a batch of %d\n%s", code, len(paths), decode(stderr))
        if not stdout.strip():
            return
        try:
            parsed = json.loads(stdout.decode("utf-8", "replace"))
        except json.JSONDecodeError as exc:
            log.debug("unparseable batched exiftool output for %d file(s)", len(paths), exc_info=exc)
            return

        # Keyed by the argument exiftool echoes back, not by position: a file it could
        # not read is absent from the array entirely, so index N is not path N.
        by_argument = {str(path): path for path in paths}
        for entry in parsed:
            source = entry.get("SourceFile") if isinstance(entry, dict) else None
            if not isinstance(source, str):
                continue
            self._exif_cache[by_argument.get(source, Path(source))] = entry

    async def _fill_ffprobe(self, path: Path) -> None:
        """Probe one video, leaving the cache untouched if it could not be read.

        A failure here is deliberately silent: the lazy :meth:`ffprobe_raw` call in
        the planner then runs for real and raises the :class:`ProbeError` whose message
        the user sees, so the diagnostic is produced in exactly one place.
        """
        code, stdout, stderr = await run_async(self._ffprobe_argv(path))
        if code != 0:
            log.debug("ffprobe exit %d on %s\n%s", code, path, decode(stderr))
            return
        try:
            self._ffprobe_cache[path] = json.loads(stdout.decode("utf-8", "replace"))
        except json.JSONDecodeError as exc:
            log.debug("ffprobe returned invalid JSON for %s", path, exc_info=exc)

    # -- ffprobe ---------------------------------------------------------

    def _ffprobe_argv(self, path: Path) -> list[str]:
        """The one ffprobe invocation, shared by the lazy and prefetched reads.

        Both paths fill the same cache, so they have to ask for the same thing: a
        flag added to only one would give an entry whose contents depend on which
        call happened to fill it.
        """
        return [self._tools.ffprobe, *_FFPROBE_OPTIONS, str(path)]

    def ffprobe_raw(self, path: Path) -> dict[str, Any]:
        if path in self._ffprobe_cache:
            return self._ffprobe_cache[path]
        out = _run(self._ffprobe_argv(path))
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError as exc:
            raise ProbeError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc
        self._ffprobe_cache[path] = parsed
        return parsed

    def video_info(self, path: Path) -> VideoInfo:
        raw = self.ffprobe_raw(path)
        streams = raw.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        if video is None:
            raise ProbeError(f"no video stream in {path}")

        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

        rotation = 0
        for side_data in video.get("side_data_list", []):
            if "rotation" in side_data:
                try:
                    rotation = int(side_data["rotation"])
                except (TypeError, ValueError):
                    rotation = 0
                break

        duration_raw = raw.get("format", {}).get("duration")
        try:
            duration = float(duration_raw) if duration_raw is not None else None
        except (TypeError, ValueError):
            duration = None

        return VideoInfo(
            width=int(video["width"]),
            height=int(video["height"]),
            rotation=rotation,
            codec=str(video.get("codec_name", "")),
            has_audio=audio is not None,
            audio_codec=str(audio["codec_name"]) if audio and audio.get("codec_name") else None,
            duration=duration,
        )

    # -- exiftool --------------------------------------------------------

    def exif(self, path: Path) -> dict[str, Any]:
        """Selected exiftool tags as a dict, cached per path.

        The lazy path, and the fallback for anything :meth:`prefetch` could not read
        in bulk. Both ask for the same :data:`_EXIF_TAGS`, so a cached entry means the
        same thing however it got there -- which is what lets :meth:`xmp_packet` trust
        that a missing ``XMP`` key means the file has none.
        """
        if path in self._exif_cache:
            return self._exif_cache[path]
        out = _run([self._tools.exiftool, *_EXIF_OPTIONS, *_EXIF_TAGS, str(path)])
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError as exc:
            raise ProbeError(f"exiftool returned invalid JSON for {path}: {exc}") from exc
        data = parsed[0] if parsed else {}
        self._exif_cache[path] = data
        return data

    def image_info(self, path: Path) -> ImageInfo:
        """Pixel dimensions of an image.

        For a motion photo this reports the *primary* image, which is the correct
        thing to gate the skip decision on -- not any container-level property.
        """
        data = self.exif(path)
        width = data.get("ImageWidth")
        height = data.get("ImageHeight")
        if width is None or height is None:
            raise ProbeError(f"could not read dimensions of {path}")
        return ImageInfo(width=int(width), height=int(height))

    def xmp_packet(self, path: Path) -> str:
        """The raw XMP packet as text, or ``""`` when the file carries none.

        Comes from the same read as every other tag. This used to be a second
        ``exiftool -b -XMP`` launch per JPEG, which doubled the cost of planning an
        image library to answer one boolean -- see :meth:`prefetch`.
        """
        return _decode_binary_tag(self.exif(path).get("XMP"))
