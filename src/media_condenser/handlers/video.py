"""Video transcoding to 720p H.265.

Every flag in :func:`build_ffmpeg_command` is there because its absence produced a
concrete, observed defect. The comments record which one, so none of them get
"cleaned up" later.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path

from media_condenser import config, handlers, probe

log = logging.getLogger(__name__)

#: Scale filter evaluated against the *decoded* frame, i.e. after ffmpeg has
#: applied the container's display matrix. Picking the axis with ``gt(iw,ih)``
#: rather than hardcoding ``-2:720`` is what makes portrait-stored-as-landscape
#: files scale on the correct axis. A square frame takes the false branch and
#: comes out square, not distorted.
_SCALE_TEMPLATE = "scale='if(gt(iw,ih),-2,{edge})':'if(gt(iw,ih),{edge},-2)'"


def build_scale_filter(max_short_edge: int) -> str:
    return _SCALE_TEMPLATE.format(edge=max_short_edge)


def build_ffmpeg_command(
    source: Path,
    target: Path,
    info: probe.VideoInfo,
    rules: config.VideoRules,
    cfg: config.GlobalConfig,
) -> list[str]:
    """Assemble the ffmpeg invocation for one video."""
    # fmt: off
    # Kept one flag-and-value pair per line.
    argv = [
        cfg.tools.ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(source),
        # Take the first video stream, and audio only if there is one. Timelapses
        # and some motion-photo videos have no audio track at all, and mapping a
        # nonexistent stream is a hard error. The '?' makes the map optional as a
        # second line of defence behind the has_audio check below.
        "-map", "0:v:0",
    ]
    if rules.copy_audio and info.has_audio:
        argv += ["-map", "0:a:0?"]

    argv += [
        "-vf", build_scale_filter(rules.max_short_edge),
        "-c:v", rules.codec,
        "-crf", str(rules.crf),
        "-preset", rules.preset,
        # libx265 tags its output 'hev1' by default while originals use 'hvc1';
        # some players reject the former outright.
        "-tag:v", rules.tag,
    ]

    if rules.codec == "libx265":
        # x265's own thread pool grabs every core by default, so N concurrent jobs
        # each try to use the whole machine and thrash. Capping the pool per job
        # keeps jobs x pools within the core count.
        argv += ["-x265-params", f"pools={cfg.encoder_pools}:log-level=error"]

    if rules.copy_audio and info.has_audio:
        argv += ["-c:a", "copy"]

    argv += [
        # ffmpeg drops creation_time and the com.android.* tags whenever a video
        # filter is applied. These three restore container and per-stream metadata.
        "-map_metadata", "0",
        "-map_metadata:s:v", "0:s:v",
        "-movflags", "+use_metadata_tags+faststart",
        str(target),
    ]
    # fmt: on
    return argv


async def run_command(argv: list[str]) -> tuple[int, str]:
    """Run a subprocess, returning its exit code and captured stderr.

    Every async subprocess in the tool goes through here, which makes it the one
    place worth logging: the caller only ever keeps the last line of stderr (see
    :func:`media_condenser.handlers.tail`), so without the debug record below the actual ffmpeg
    or exiftool complaint is gone and a failed encode can only be diagnosed by
    reproducing it.
    """
    log.debug("$ %s", shlex.join(argv))
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, raw = await process.communicate()
    code = process.returncode or 0
    stderr = raw.decode("utf-8", "replace").strip()
    if code and stderr:
        log.debug("%s exit %d\n%s", Path(argv[0]).name, code, stderr)
    return code, stderr


def build_metadata_restore_command(
    source: Path,
    target: Path,
    cfg: config.GlobalConfig,
) -> list[str]:
    """Restore container metadata atoms that ``-map_metadata`` does not carry.

    ``-map_metadata`` handles the common tags but misses some QuickTime atoms
    entirely. A drone fixture lost its ``GPSCoordinates`` this way -- the location of
    the footage, silently gone -- which only showed up when diffing the full tag set
    of an output against its original.

    Scoped to ``UserData`` and ``Keys`` on purpose. Those hold metadata only, so
    copying them wholesale cannot disturb the stream properties or reintroduce a
    stale rotation matrix (rotation lives in the track header, and by this point it
    has already been baked into the pixels).
    """
    # fmt: off
    return [
        cfg.tools.exiftool,
        "-quiet",
        "-overwrite_original",
        "-tagsFromFile", str(source),
        "-UserData:all",
        "-Keys:all",
        str(target),
    ]
    # fmt: on


async def transcode(
    source: Path,
    target: Path,
    info: probe.VideoInfo,
    rules: config.VideoRules,
    cfg: config.GlobalConfig,
) -> handlers.HandlerResult:
    """Transcode one video file to ``target``."""
    original_size = source.stat().st_size
    argv = build_ffmpeg_command(source, target, info, rules, cfg)
    code, stderr = await run_command(argv)

    if code != 0 or not target.exists():
        return handlers.HandlerResult(
            ok=False,
            original_size=original_size,
            error=handlers.tail(stderr) or f"ffmpeg exit {code}",
        )

    notes: list[str] = []
    code, stderr = await run_command(build_metadata_restore_command(source, target, cfg))
    if code != 0:
        # The video itself is fine, so this is a downgrade rather than a failure:
        # the result is usable but may have lost its location metadata.
        notes.append(f"container metadata restore failed ({handlers.tail(stderr) or f'exiftool exit {code}'})")

    return handlers.HandlerResult(
        ok=True,
        original_size=original_size,
        new_size=target.stat().st_size,
        notes=notes,
    )
