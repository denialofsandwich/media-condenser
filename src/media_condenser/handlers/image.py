"""Image downscaling with metadata preservation.

ImageMagick and ffmpeg both drop EXIF, GPS and timestamps when they resize, so the
metadata is copied back explicitly afterwards. That second step is not optional --
without it the output loses its creation date and location.
"""

from __future__ import annotations

from pathlib import Path

from media_condenser import config, handlers
from media_condenser.handlers import video


def build_convert_command(
    source: Path,
    target: Path,
    rules: config.ImageRules,
    cfg: config.GlobalConfig,
) -> list[str]:
    """Assemble the ImageMagick resize.

    The ``>`` suffix on the geometry is ImageMagick's own shrink-only flag: it
    refuses to enlarge an image that is already smaller than the target. The
    planner already gates on this, so the flag is a redundant safeguard -- an
    upscale should be impossible by two independent mechanisms.
    """
    # fmt: off
    return [
        cfg.tools.convert,
        str(source),
        "-auto-orient",
        "-resize", f"{rules.max_edge}x{rules.max_edge}>",
        "-quality", str(rules.quality),
        "-sampling-factor", rules.subsampling,
        "-strip",
        str(target),
    ]
    # fmt: on


def build_metadata_restore_command(
    source: Path,
    target: Path,
    cfg: config.GlobalConfig,
) -> list[str]:
    """Copy every tag from the original onto the resized output.

    ``-all:all`` carries EXIF, XMP (including the Google container namespaces) and
    GPS. Orientation is reset because the pixels were already physically rotated by
    ``-auto-orient``; leaving the original value would rotate the image a second
    time when viewed.
    """
    # fmt: off
    return [
        cfg.tools.exiftool,
        "-quiet",
        "-overwrite_original",
        "-tagsFromFile", str(source),
        "-all:all",
        "-Orientation=1",
        "-n",
        str(target),
    ]
    # fmt: on


def build_container_clear_command(target: Path, cfg: config.GlobalConfig) -> list[str]:
    """Delete every declaration of embedded motion photo components.

    ``GCamera`` holds the ``MotionPhoto`` flag, ``GContainer`` the directory of
    component lengths, and MPF the gain map's absolute offset. All three describe
    appended data that a plain resize does not reproduce, so all three have to go --
    a surviving ``Container:Directory`` would have a reader slicing past the end of
    the primary image and treating whatever it finds as a video.
    """
    # fmt: off
    return [
        cfg.tools.exiftool,
        "-quiet",
        "-overwrite_original",
        "-XMP-GCamera:all=",
        "-XMP-GContainer:all=",
        "-MPF:all=",
        str(target),
    ]
    # fmt: on


async def resize(
    source: Path,
    target: Path,
    rules: config.ImageRules,
    cfg: config.GlobalConfig,
    *,
    clear_motion_container: bool = False,
) -> handlers.HandlerResult:
    """Resize ``source`` into ``target`` and restore its metadata."""
    original_size = source.stat().st_size

    code, stderr = await video.run_command(build_convert_command(source, target, rules, cfg))
    if code != 0 or not target.exists():
        return handlers.HandlerResult(
            ok=False, original_size=original_size, error=handlers.tail(stderr) or f"convert exit {code}"
        )

    code, stderr = await video.run_command(build_metadata_restore_command(source, target, cfg))
    if code != 0:
        # The pixels are fine but the file would ship without its timestamps and
        # GPS. That is a real loss, so it counts as a failure rather than a
        # downgrade -- the original is still untouched at this point.
        return handlers.HandlerResult(
            ok=False,
            original_size=original_size,
            error=f"metadata restore failed: {handlers.tail(stderr) or f'exiftool exit {code}'}",
        )

    if clear_motion_container:
        # Runs after the restore, because the restore is what copied these tags over.
        code, stderr = await video.run_command(build_container_clear_command(target, cfg))
        if code != 0:
            return handlers.HandlerResult(
                ok=False,
                original_size=original_size,
                error=(
                    f"could not clear stale motion photo declarations: "
                    f"{handlers.tail(stderr) or f'exiftool exit {code}'}"
                ),
            )

    return handlers.HandlerResult(ok=True, original_size=original_size, new_size=target.stat().st_size)
