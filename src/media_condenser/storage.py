"""Where output goes, and how it gets committed safely.

Both strategies build the result in a temporary file first and only move it into
place once the work has fully succeeded. The temp file is always created **in the
destination's own directory**, which is the one detail that matters here:
``os.replace`` is atomic only within a single filesystem, and a temp file under
``/tmp`` fails with ``EXDEV`` the moment the media library lives on another mount.
That failure hit 1,747 files in a single real run.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from media_condenser import config

log = logging.getLogger(__name__)

TMP_PREFIX = ".mcon_tmp_"
"""Prefix of every file this module writes before committing it.

Public because the walk has to recognise -- and refuse -- its own leftovers.
"""


class StorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class Destination:
    """Resolved paths for one file's output."""

    source: Path
    final: Path
    tmp: Path

    @property
    def replaces_source(self) -> bool:
        return self.final == self.source


def plan_destination(
    source: Path,
    cfg: config.GlobalConfig,
    scan_root: Path | None = None,
) -> Destination:
    """Decide the final and temporary paths for ``source``.

    ``copy`` (the default) never touches the source: output goes into the mirror
    tree at ``output_dir``, which is why a copy run without one is refused rather
    than guessed at. Writing ``photo_compressed.jpg`` beside every original was the
    old fallback, and it produced a library interleaving two copies of everything
    under names no downstream tool understands -- a worse outcome than stopping.
    """
    if cfg.strategy is config.Strategy.REPLACE:
        final = source
    elif cfg.output_dir is not None:
        root = (scan_root or source.parent).resolve()
        try:
            relative = source.resolve().relative_to(root)
        except ValueError:
            relative = Path(source.name)
        final = cfg.output_dir.resolve() / relative
    else:
        # The CLI refuses this combination up front, so reaching here means a
        # programmatic caller built the config itself.
        raise StorageError(
            "the copy strategy needs an output_dir to copy into; "
            "set one, or use the replace strategy to overwrite originals in place"
        )

    tmp = final.parent / f"{TMP_PREFIX}{os.getpid()}_{final.name}"
    return Destination(source=source, final=final, tmp=tmp)


def prepare(destination: Destination) -> None:
    """Create the destination directory and clear any stale temp file."""
    try:
        destination.final.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageError(f"cannot create output directory {destination.final.parent}: {exc}") from exc
    if destination.tmp.exists():
        destination.tmp.unlink()


def assert_same_filesystem(destination: Destination) -> None:
    """Verify the temp file will land on the destination's filesystem.

    Deriving the temp path from ``final.parent`` already guarantees this; the check
    exists to fail loudly and early if that ever stops being true, rather than
    surfacing as a cross-device error at the final rename -- after the expensive
    encode has already been done.
    """
    try:
        target_dev = destination.final.parent.stat().st_dev
        tmp_dev = destination.tmp.parent.stat().st_dev
    except OSError as exc:
        raise StorageError(f"cannot stat destination for {destination.final}: {exc}") from exc
    if target_dev != tmp_dev:
        raise StorageError(
            f"temp file {destination.tmp} is on a different filesystem than {destination.final}; "
            "the final rename would fail with a cross-device link error"
        )


def commit(destination: Destination) -> int:
    """Move the finished temp file into its final location atomically.

    Returns the committed size in bytes. The source's modification time is carried
    over so library sort order survives processing -- the embedded timestamps are
    preserved separately by the handlers.
    """
    if not destination.tmp.exists():
        raise StorageError(f"nothing to commit: {destination.tmp} does not exist")

    size = destination.tmp.stat().st_size
    if size == 0:
        destination.tmp.unlink(missing_ok=True)
        raise StorageError("refusing to commit an empty output file")

    try:
        stat = destination.source.stat()
        os.utime(destination.tmp, (stat.st_atime, stat.st_mtime))
    except OSError as exc:
        # A missing timestamp is not worth failing the whole file over, but it
        # does defeat the rerun-is-cheap check in `_already_produced`, so it is
        # worth a trace when someone wonders why a file keeps re-encoding.
        log.debug("could not carry the mtime onto %s", destination.tmp, exc_info=exc)

    try:
        os.replace(destination.tmp, destination.final)
    except OSError as exc:
        destination.tmp.unlink(missing_ok=True)
        raise StorageError(f"could not move {destination.tmp} into place: {exc}") from exc
    return size


def discard(destination: Destination) -> None:
    """Remove the temp file after a failure, leaving the source untouched."""
    destination.tmp.unlink(missing_ok=True)


def sibling_tmp(destination: Destination, tag: str) -> Path:
    """An extra temp path beside the destination, for multi-stage work.

    Motion photos need several intermediates (resized primary, rescaled gain map,
    re-encoded video). They go next to the destination for the same
    same-filesystem reason as the main temp file.
    """
    return destination.final.parent / f"{TMP_PREFIX}{os.getpid()}_{tag}_{destination.final.name}"


def cleanup_stale(root: Path) -> int:
    """Delete leftover temp files under ``root`` from an interrupted run.

    Temp names carry the pid that created them, and a file belonging to a *live*
    process is left alone: two concurrent runs over one library would otherwise
    delete each other's half-written output, turning a working run into a spray of
    "nothing to commit" failures. Our own pid is exempt from that protection --
    a fresh process cannot have written those, so they are leftovers from a previous
    run that happened to be assigned the same pid.
    """
    removed = 0
    for path in root.rglob(f"{TMP_PREFIX}*"):
        if not path.is_file():
            continue
        pid = _pid_of(path.name)
        if pid is not None and pid != os.getpid() and _pid_is_running(pid):
            continue
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            log.debug("could not remove stale temp file %s", path, exc_info=exc)
    return removed


def _pid_of(name: str) -> int | None:
    """The pid embedded in a temp filename, or ``None`` if it has none."""
    rest = name.removeprefix(TMP_PREFIX)
    digits, _, remainder = rest.partition("_")
    if not remainder or not digits.isdigit():
        return None
    return int(digits)


def _pid_is_running(pid: int) -> bool:
    """Whether ``pid`` is a live process.

    Errs towards "running" whenever the answer is unclear (an ``EPERM`` means some
    other user's process holds that pid), because the consequence of guessing wrong
    in that direction is only an uncollected temp file.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True
