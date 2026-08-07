"""Motion photo rebuilding.

Splits the container into its components, processes each one, and concatenates the
results with the XMP directory lengths corrected to match the bytes actually
written. See :mod:`media_condenser.container` for the parsing rules and why they are what they
are.

Two behaviours are deliberate and worth stating up front:

- A component that cannot be processed is **dropped, not fatal**. The file still
  gets rebuilt from whatever survived and is reported as *downgraded*. Losing the
  motion clip off a photo is a far better outcome than failing the file, and
  lumping the two together makes the summary unreadable.
- The primary image's resolution alone decides whether the file is skipped, which
  is what keeps reruns idempotent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from media_condenser import config, container, handlers, probe
from media_condenser.handlers import image, video

log = logging.getLogger(__name__)


@dataclass
class ScratchPaths:
    """Temp paths and rules for one motion photo rebuild.

    Every intermediate lives beside the final destination so that it shares the
    destination's filesystem -- the same constraint that governs the main temp file
    in :mod:`media_condenser.storage`.
    """

    base: Path
    """Prefix for intermediates, already inside the destination directory."""

    image_rules: config.ImageRules
    video_rules: config.VideoRules

    _created: list[Path] = field(default_factory=list)
    """Every path :meth:`path` has handed out, which is exactly what
    :meth:`cleanup` has to remove."""

    def path(self, tag: str, suffix: str = "") -> Path:
        """A temp path for one intermediate.

        ``suffix`` matters for ffmpeg outputs: it infers the output container from
        the extension, and fails with a bare "Invalid argument" on an extensionless
        path.
        """
        minted = self.base.with_name(f"{self.base.name}_{tag}{suffix}")
        self._created.append(minted)
        return minted

    def cleanup(self) -> None:
        """Remove the intermediates this rebuild created.

        Tracked rather than globbed: the glob this replaces scanned the whole
        destination directory once per motion photo, and that directory is where the
        run's outputs accumulate -- so the cost grew with every file processed.
        """
        for leftover in self._created:
            leftover.unlink(missing_ok=True)
        self._created.clear()


async def rebuild(
    layout: container.MotionPhotoLayout,
    target: Path,
    prober: probe.Prober,
    cfg: config.GlobalConfig,
    rules: config.MotionPhotoRules,
    scratch: ScratchPaths,
) -> handlers.HandlerResult:
    """Rebuild one motion photo into ``target``."""
    source = layout.path
    original_size = layout.file_size
    image_rules = scratch.image_rules
    notes: list[str] = list(layout.notes)

    data = source.read_bytes()

    # -- primary image -------------------------------------------------
    # The primary is sliced out rather than handed to ImageMagick whole: passing the
    # full container would make it read past the primary's EOI into the appended
    # components.
    primary_raw = scratch.path("primary_in", ".jpg")
    primary_out = scratch.path("primary_out", ".jpg")
    assert layout.primary is not None
    primary_raw.write_bytes(data[layout.primary.offset : layout.primary.end])

    code, stderr = await video.run_command(image.build_convert_command(primary_raw, primary_out, image_rules, cfg))
    if code != 0 or not primary_out.exists():
        return handlers.HandlerResult(
            ok=False,
            original_size=original_size,
            error=f"primary resize failed: {handlers.tail(stderr) or f'convert exit {code}'}",
        )

    # Copy metadata from the original container. exiftool reads the primary's
    # tags -- including the GContainer XMP -- and ignores the appended blobs.
    code, stderr = await video.run_command(image.build_metadata_restore_command(source, primary_out, cfg))
    if code != 0:
        return handlers.HandlerResult(
            ok=False,
            original_size=original_size,
            error=f"primary metadata restore failed: {handlers.tail(stderr) or f'exiftool exit {code}'}",
        )

    # -- gain map ------------------------------------------------------
    gain_bytes: bytes | None = None
    if layout.gain_map is not None and rules.process_gain_map:
        gain_bytes, gain_note = await _process_gain_map(data, layout.gain_map, scratch, prober, cfg)
        if gain_note:
            notes.append(gain_note)
    elif layout.gain_map is not None:
        notes.append("gain map dropped (disabled by config)")

    # -- embedded video ------------------------------------------------
    video_bytes: bytes | None = None
    if layout.video is not None and rules.process_video:
        video_bytes, video_note = await _process_video(data, layout.video, scratch, prober, cfg)
        if video_note:
            notes.append(video_note)
    elif layout.video is not None:
        notes.append("embedded video dropped (disabled by config)")

    # -- assemble ------------------------------------------------------
    rebuilt = container.assemble(primary_out.read_bytes(), gain_bytes, video_bytes)
    target.write_bytes(rebuilt)

    return handlers.HandlerResult(
        ok=True,
        original_size=original_size,
        new_size=len(rebuilt),
        notes=notes,
    )


async def _process_gain_map(
    data: bytes,
    component: container.Component,
    scratch: ScratchPaths,
    prober: probe.Prober,
    cfg: config.GlobalConfig,
) -> tuple[bytes | None, str]:
    """Carry the gain map across, re-encoding it only if it is oversized.

    Passed through **verbatim** in the normal case. The Ultra HDR spec lets the gain
    map be a different resolution from the primary -- renderers scale it to fit --
    so shrinking it in proportion buys nothing. Real gain maps here are 8-104 KB, so
    re-encoding would spend a generation of quality on HDR data to save a few KB.
    Verified that ImageMagick does preserve the ``hdrgm:GainMapMin``/``Max``
    parameters, so the passthrough is a size/quality choice rather than a
    metadata-safety one -- but the same skip-if-already-small rule applies here as
    everywhere else in the tool.

    Sliced manually from the resolved offset. ExifTool's ``GainMapImage`` tag is
    never used: on real files it returned the *video* bytes instead of the gain map
    on 2 of 3 samples, and inconsistently enough that it cannot be worked around.
    """
    blob = data[component.offset : component.end]
    rules = scratch.image_rules

    raw = scratch.path("gain_in", ".jpg")
    raw.write_bytes(blob)
    try:
        info = prober.image_info(raw)
    except probe.ProbeError as exc:
        log.debug("gain map is not a readable image; carried across unchanged", exc_info=exc)
        return blob, ""
    if info.long_edge <= rules.max_edge:
        return blob, ""

    out = scratch.path("gain_out", ".jpg")
    # fmt: off
    argv = [
        cfg.tools.convert,
        str(raw),
        "-resize", f"{rules.max_edge}x{rules.max_edge}>",
        "-quality", str(rules.quality),
        str(out),
    ]
    # fmt: on
    code, stderr = await video.run_command(argv)
    if code != 0 or not out.exists():
        # Keeping the original gain map bytes preserves HDR at the cost of a few
        # KB, which beats dropping it and silently downgrading the image.
        return blob, f"gain map kept unresized: rescale failed ({handlers.tail(stderr) or f'convert exit {code}'})"
    return out.read_bytes(), ""


async def _process_video(
    data: bytes,
    component: container.Component,
    scratch: ScratchPaths,
    prober: probe.Prober,
    cfg: config.GlobalConfig,
) -> tuple[bytes | None, str]:
    """Re-encode the embedded video through the standalone video path.

    The embedded clip uses the same container-level rotation trick as standalone
    videos -- one fixture's is 1792x1008 with a -90 display matrix -- so it needs
    the same orientation-aware filter rather than a simplified one.
    """
    raw = scratch.path("video_in", ".mp4")
    out = scratch.path("video_out", ".mp4")
    blob = data[component.offset : component.end]
    raw.write_bytes(blob)

    try:
        info = prober.video_info(raw)
    except probe.ProbeError as exc:
        return None, f"embedded video dropped: unreadable ({exc})"

    rules = scratch.video_rules
    if info.short_edge <= rules.max_short_edge:
        # Already small enough: keep the original bytes rather than re-encoding for
        # no gain, which would also cost a generation of quality.
        return blob, ""

    argv = video.build_ffmpeg_command(raw, out, info, rules, cfg)
    code, stderr = await video.run_command(argv)
    if code != 0 or not out.exists():
        return None, f"embedded video dropped: re-encode failed ({handlers.tail(stderr) or f'ffmpeg exit {code}'})"
    return out.read_bytes(), ""
