"""Concurrent execution of planned actions.

The CPU work all happens inside ffmpeg/ImageMagick/exiftool child processes, so
Python only needs to supervise. A semaphore-bounded set of asyncio tasks does that
without the overhead or pickling constraints of multiprocessing, and it keeps every
core busy because each slot holds a real subprocess.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import signal
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from media_condenser import config, dates, handlers, planner, probe, progress, report, storage, verify
from media_condenser.handlers import image, motion, video

log = logging.getLogger(__name__)

_OUTCOME_LEVEL = {
    report.Outcome.SUCCEEDED: logging.INFO,
    report.Outcome.DOWNGRADED: logging.WARNING,
    report.Outcome.SKIPPED: logging.DEBUG,
    report.Outcome.FAILED: logging.ERROR,
}

#: Short labels for the per-file progress rows, aligned into a fixed-width column.
_VERB_LABEL = {
    planner.Verb.RESIZE_IMAGE: "resize",
    planner.Verb.TRANSCODE_VIDEO: "transcode",
    planner.Verb.REBUILD_MOTION_PHOTO: "motion",
}


def _publish(reporter: report.Reporter, record: report.Record) -> None:
    """Record one file's outcome and log it.

    Every outcome goes through here so that nothing can be counted in the summary
    without also being visible while the run is happening -- which is what the tool
    used to do with all of them.
    """
    reporter.add(record)
    level = _OUTCOME_LEVEL[record.outcome]
    if log.isEnabledFor(level):
        log.log(
            level,
            "%s %s%s",
            record.outcome.value,
            record.path.name,
            _outcome_suffix(record),
            extra={
                "file": str(record.path),
                "kind": str(record.kind),
                "outcome": record.outcome.value,
                "original_size": record.original_size,
                "new_size": record.new_size,
                "saved": record.saved,
            },
        )


def _outcome_suffix(record: report.Record) -> str:
    if record.outcome in (report.Outcome.SUCCEEDED, report.Outcome.DOWNGRADED) and record.original_size:
        shrink = f"  {report.human_bytes(record.original_size)} -> {report.human_bytes(record.new_size)}"
        shrink += f" ({report.compression_rate(record.original_size, record.new_size)} smaller)"
        return f"{shrink}: {record.detail}" if record.detail else shrink
    return f": {record.detail}" if record.detail else ""


async def execute_all(
    actions: list[planner.Action],
    cfg: config.GlobalConfig,
    prober: probe.Prober,
    reporter: report.Reporter,
    *,
    scan_root: Path | None = None,
    display: progress.RunProgress | None = None,
    verify_outputs: bool = False,
) -> bool:
    """Run every action, bounded by ``cfg.jobs``.

    Returns whether the run was interrupted, so the caller can still print what did
    finish and then exit 130.

    ``verify_outputs`` checks each output before it is committed, so a file that
    fails verification is discarded and reported as failed rather than being left in
    place. See :func:`_execute_one`.
    """
    semaphore = asyncio.Semaphore(cfg.jobs)
    display = display or progress.RunProgress(enabled=False)

    colliding = _colliding_destinations(actions, cfg, scan_root)
    for action in colliding:
        _publish(
            reporter,
            _failed_record(action, "two or more inputs map to the same output path; refusing to overwrite"),
        )

    doomed = {id(action) for action in colliding}
    actions = [action for action in actions if id(action) not in doomed]
    to_process = [action for action in actions if action.will_process]

    mirroring = mirrors_skipped_files(cfg)
    to_copy = [action for action in actions if action.verb is planner.Verb.SKIP] if mirroring else []

    for action in actions:
        if action.verb is planner.Verb.SKIP and not mirroring:
            _publish(reporter, _skipped_record(action, action.reason))
        elif action.verb is planner.Verb.FAIL:
            _publish(reporter, _failed_record(action, action.reason))

    if not to_process and not to_copy:
        return False

    queued = to_process + to_copy
    queued_bytes = sum(a.candidate.size for a in queued)
    display.phase("Compressing", total=queued_bytes, unit="bytes")
    log.info("processing %d files (%s)", len(queued), report.human_bytes(queued_bytes))

    async def worker(action: planner.Action, *, copy_only: bool) -> None:
        async with semaphore:
            label = "copy" if copy_only else _VERB_LABEL.get(action.verb, str(action.verb))
            with display.file(label, action.path.name, action.candidate.size):
                if copy_only:
                    record = await _copy_through(action, cfg, scan_root)
                else:
                    record = await _execute_one(action, cfg, prober, scan_root, verify_outputs=verify_outputs)
            _publish(reporter, record)
            display.advance(action.candidate.size)
            display.add_saved(record.saved)

    pending = asyncio.gather(
        *(worker(action, copy_only=False) for action in to_process),
        *(worker(action, copy_only=True) for action in to_copy),
    )
    with _cancel_on_interrupt(pending) as interrupted:
        try:
            await pending
        except asyncio.CancelledError:
            if not interrupted.is_set():
                raise
    if interrupted.is_set():
        log.warning(
            "interrupted: %d of %d files finished, the rest were left untouched",
            len(reporter.records),
            len(queued),
        )
        return True
    return False


@contextlib.contextmanager
def _cancel_on_interrupt(pending: asyncio.Future[Any]) -> Iterator[asyncio.Event]:
    """Turn Ctrl-C into cancellation of the in-flight encodes.

    Without this the default handler raises ``KeyboardInterrupt`` straight out of
    ``asyncio.run``: the workers never unwind, so their half-written ``.mcon_tmp_*``
    files stay on disk until some later run's ``cleanup_stale`` finds them, and the
    summary of everything that had already finished is lost. Cancelling instead lets
    each worker run its cleanup and lets the caller report what it got.
    """
    interrupted = asyncio.Event()

    def _on_interrupt() -> None:
        if not interrupted.is_set():
            interrupted.set()
            pending.cancel()

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, _on_interrupt)
    except NotImplementedError:
        # No signal handlers on this platform (Windows). Ctrl-C keeps its default
        # behaviour there rather than the tool failing to start.
        yield interrupted
        return
    try:
        yield interrupted
    finally:
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.remove_signal_handler(signal.SIGINT)


def _colliding_destinations(
    actions: list[planner.Action],
    cfg: config.GlobalConfig,
    scan_root: Path | None,
) -> list[planner.Action]:
    """Actions whose output path is claimed by more than one input.

    Two files committing to one path means the second silently destroys the first,
    and both are reported as succeeded -- a mirror tree that looks complete while
    holding one file where two were expected. Failing both is the honest outcome: it
    is visible in the summary, non-zero in the exit code, and leaves every original
    where it was.
    """
    by_final: dict[Path, list[planner.Action]] = {}
    for action in actions:
        final = storage.plan_destination(action.path, cfg, scan_root).final
        by_final.setdefault(final, []).append(action)
    return [action for group in by_final.values() if len(group) > 1 for action in group]


def mirrors_skipped_files(cfg: config.GlobalConfig) -> bool:
    """Whether skipped files should be copied into the output tree.

    Only for the ``copy`` strategy, whose mirror tree would otherwise have holes
    in it wherever a file needed no work. ``replace`` leaves every file where it
    already is, so a skip is already "in the output" and there is nothing to fill.

    The ``output_dir`` test is redundant now that a copy run without one is refused
    outright, and kept as the cheaper of the two ways to state the same thing: this
    returns an answer for any config, rather than only for one that could run.
    """
    return cfg.strategy is config.Strategy.COPY and cfg.output_dir is not None


def _skipped_record(action: planner.Action, detail: str) -> report.Record:
    return report.Record(
        path=action.path,
        kind=action.kind,
        outcome=report.Outcome.SKIPPED,
        original_size=action.candidate.size,
        new_size=action.candidate.size,
        detail=detail,
        notes=action.notes,
    )


def _failed_record(action: planner.Action, detail: str, **extra: Any) -> report.Record:
    """A failure for one file. ``new_size`` stays zero: nothing was committed.

    Every failure in this module goes through here so that a field added to
    :class:`~media_condenser.report.Record` reaches all of them rather than the subset someone
    remembered to update.
    """
    return report.Record(
        path=action.path,
        kind=action.kind,
        outcome=report.Outcome.FAILED,
        original_size=action.candidate.size,
        detail=detail,
        **extra,
    )


async def _copy_through(
    action: planner.Action,
    cfg: config.GlobalConfig,
    scan_root: Path | None,
    *,
    reason: str | None = None,
) -> report.Record:
    """Copy an unmodified file into the mirror tree.

    Committed through the same temp-file-then-rename path as processed output, so an
    interrupted run cannot leave a half-written file that a later run would mistake
    for a complete copy.
    """
    destination = storage.plan_destination(action.path, cfg, scan_root)
    reason = action.reason if reason is None else reason

    try:
        if _already_copied(action.path, destination.final):
            return _skipped_record(action, f"{reason}; already present in the output tree")

        storage.prepare(destination)
        storage.assert_same_filesystem(destination)
        await asyncio.to_thread(shutil.copy2, action.path, destination.tmp)
        storage.commit(destination)
        return _skipped_record(action, f"{reason}; copied unchanged")

    except asyncio.CancelledError:
        storage.discard(destination)
        raise
    except Exception as exc:
        storage.discard(destination)
        return _failed_record(action, f"could not copy unchanged file into the output tree: {exc}")


def existing_output(
    action: planner.Action,
    cfg: config.GlobalConfig,
    prober: probe.Prober,
    scan_root: Path | None,
) -> str | None:
    """The reason this action would be skipped because its output is already there.

    The same two guards the run itself uses, exposed so ``--dry-run`` can report what
    a real run would actually do. Without it, a rerun over a populated ``--output-dir``
    previews hundreds of re-encodes and then skips every one of them -- and a dry run
    that does not predict the run is worse than no dry run, because it is trusted.

    Reusing the predicates rather than restating them is the point: a second copy of
    "is this already done?" is exactly the kind of thing that drifts out of agreement
    with the first, and the drift would be invisible.
    """
    if not action.will_process and action.verb is not planner.Verb.SKIP:
        return None

    destination = storage.plan_destination(action.path, cfg, scan_root)
    if action.verb is planner.Verb.SKIP:
        if mirrors_skipped_files(cfg) and _already_copied(action.path, destination.final):
            return f"{action.reason}; already present in the output tree"
        return None
    if _already_produced(action, destination, prober):
        return "already present in the output tree"
    return None


def _stat_pair(source: Path, final: Path) -> tuple[os.stat_result, os.stat_result] | None:
    """Both files' metadata, or ``None`` if either could not be read.

    The mtime comparisons built on this are truncated to whole seconds to match
    :func:`media_condenser.storage.commit`, which carries the source's mtime onto the output --
    the two have to agree on precision or every rerun re-encodes.
    """
    try:
        return source.stat(), final.stat()
    except OSError as exc:
        log.debug("cannot compare %s with %s", source, final, exc_info=exc)
        return None


def _already_copied(source: Path, final: Path) -> bool:
    stats = _stat_pair(source, final)
    if stats is None:
        return False
    source_stat, final_stat = stats
    return source_stat.st_size == final_stat.st_size and int(source_stat.st_mtime) == int(final_stat.st_mtime)


async def _execute_one(
    action: planner.Action,
    cfg: config.GlobalConfig,
    prober: probe.Prober,
    scan_root: Path | None,
    *,
    verify_outputs: bool = False,
) -> report.Record:
    destination = storage.plan_destination(action.path, cfg, scan_root)
    scratch: motion.ScratchPaths | None = None

    try:
        if await asyncio.to_thread(_already_produced, action, destination, prober):
            return _skipped_record(action, "already present in the output tree")

        storage.prepare(destination)
        storage.assert_same_filesystem(destination)

        if action.verb is planner.Verb.RESIZE_IMAGE:
            result = await image.resize(
                action.path,
                destination.tmp,
                action.candidate.rules.images,
                cfg,
                clear_motion_container=action.clear_motion_container,
            )
        elif action.verb is planner.Verb.TRANSCODE_VIDEO:
            assert action.video_info is not None
            # ffmpeg infers its output container from the extension, so it writes to
            # a suffixed temp path which is then moved to the committed temp name.
            encode_target = _with_video_suffix(destination.tmp, action.path)
            result = await video.transcode(
                action.path,
                encode_target,
                action.video_info,
                action.candidate.rules.videos,
                cfg,
            )
            if result.ok and encode_target != destination.tmp:
                encode_target.replace(destination.tmp)
        elif action.verb is planner.Verb.REBUILD_MOTION_PHOTO:
            assert action.layout is not None
            scratch = motion.ScratchPaths(
                base=storage.sibling_tmp(destination, "mp"),
                image_rules=action.candidate.rules.images,
                video_rules=action.candidate.rules.videos,
            )
            result = await motion.rebuild(
                action.layout,
                destination.tmp,
                prober,
                cfg,
                action.candidate.rules.motion_photos,
                scratch,
            )
        else:
            return _failed_record(action, f"unhandled action {action.verb}")

        if not result.ok:
            storage.discard(destination)
            return _failed_record(action, result.error)

        await _apply_inferred_date(action, destination.tmp, prober, cfg, result)

        if result.new_size >= action.candidate.size:
            storage.discard(destination)
            grew = f"output would be larger ({result.new_size} >= {action.candidate.size} bytes); kept the original"
            if mirrors_skipped_files(cfg):
                return await _copy_through(action, cfg, scan_root, reason=grew)
            return _skipped_record(action, grew)

        verification = None
        if verify_outputs:
            verification = await asyncio.to_thread(
                verify.verify_output, action.path, destination.tmp, action.kind, prober, cfg
            )
            if not verification.ok:
                storage.discard(destination)
                return _failed_record(
                    action,
                    f"verification failed, output discarded: {'; '.join(verification.problems)}",
                    verified=True,
                    verify_checks=len(verification.checks),
                    verify_problems=list(verification.problems),
                )
            log.debug("verified %s (%d checks)", action.path.name, len(verification.checks))

        committed = storage.commit(destination)
        notes = list(action.notes)
        for note in result.notes:
            if note not in notes:
                notes.append(note)

        return report.Record(
            path=action.path,
            kind=action.kind,
            outcome=report.Outcome.DOWNGRADED if notes else report.Outcome.SUCCEEDED,
            original_size=action.candidate.size,
            new_size=committed,
            detail="; ".join(notes),
            notes=notes,
            verified=verification is not None,
            verify_checks=len(verification.checks) if verification else 0,
        )

    except asyncio.CancelledError:
        log.debug("cancelled while working on %s", action.path)
        storage.discard(destination)
        raise
    except (storage.StorageError, OSError) as exc:
        log.debug("%s failed", action.path, exc_info=exc)
        storage.discard(destination)
        return _failed_record(action, str(exc))
    except Exception as exc:
        log.debug("unexpected failure on %s", action.path, exc_info=exc)
        storage.discard(destination)
        return _failed_record(action, f"unexpected {type(exc).__name__}: {exc}")
    finally:
        if scratch is not None:
            scratch.cleanup()


def _already_produced(action: planner.Action, destination: storage.Destination, prober: probe.Prober) -> bool:
    """Whether the destination already holds this action's finished output.

    Two conditions, both necessary. The timestamp must match, because ``commit``
    carries the source's mtime onto the output -- equality is therefore the mark of a
    file this tool wrote from *this* source, and a later edit to the source breaks it.
    And the output must still conform to the current rules, so tightening
    ``max_edge`` re-encodes rather than being mistaken for work already done.

    In-place runs are excluded: there the output *is* the source, and the planner
    already made this decision from the file's own measured properties.
    """
    if not _output_looks_current(action, destination):
        return False

    rules = action.candidate.rules
    try:
        if action.kind is planner.Kind.VIDEO:
            return prober.video_info(destination.final).short_edge <= rules.videos.max_short_edge
        return prober.image_info(destination.final).long_edge <= rules.images.max_edge
    except probe.ProbeError as exc:
        log.debug("cannot probe existing output %s; will re-encode", destination.final, exc_info=exc)
        return False


def _output_looks_current(action: planner.Action, destination: storage.Destination) -> bool:
    """The stat-only half of :func:`_already_produced`.

    Split out so the probe that follows it can be prefetched: it is the test for
    whether an output is even a candidate for reuse, and over a mostly-empty output
    tree it rejects nearly everything, which is what keeps
    :func:`prefetch_existing_outputs` from launching a probe per file that isn't there.
    """
    if destination.replaces_source:
        return False
    stats = _stat_pair(action.path, destination.final)
    return stats is not None and int(stats[0].st_mtime) == int(stats[1].st_mtime)


async def prefetch_existing_outputs(
    actions: list[planner.Action],
    cfg: config.GlobalConfig,
    prober: probe.Prober,
    scan_root: Path | None,
    *,
    concurrency: int,
) -> None:
    """Warm the probe caches for the outputs :func:`existing_output` is about to read.

    The same reasoning as :func:`media_condenser.planner.prefetch`, for the other serial per-file
    probe loop in the tool: a ``--dry-run`` over a populated ``--output-dir`` has to
    measure every output that is already there to say whether the run would skip it.
    Only outputs that pass the stat gate are probed, so a dry run against an empty
    output tree still costs nothing.
    """
    exif: list[Path] = []
    video: list[Path] = []
    for action in actions:
        if not action.will_process:
            continue
        destination = storage.plan_destination(action.path, cfg, scan_root)
        if not _output_looks_current(action, destination):
            continue
        if action.kind is planner.Kind.VIDEO:
            video.append(destination.final)
        else:
            exif.append(destination.final)

    if exif or video:
        await prober.prefetch(exif=exif, video=video, concurrency=concurrency)


def _with_video_suffix(tmp: Path, source: Path) -> Path:
    """ffmpeg infers its output container from the extension.

    The temp filename already ends in the original name (and therefore its
    extension) in the default layout, but an ``output_dir`` mirror or an unusual
    name can leave it without one.
    """
    if tmp.suffix.lower() in (".mp4", ".mov", ".m4v", ".mkv"):
        return tmp
    return tmp.with_name(tmp.name + (source.suffix or ".mp4"))


async def _apply_inferred_date(
    action: planner.Action,
    target: Path,
    prober: probe.Prober,
    cfg: config.GlobalConfig,
    result: handlers.HandlerResult,
) -> None:
    """Write a filename-inferred date, but only when metadata had none.

    Short-circuits on the filename first: if no pattern matches there is nothing to
    infer, so the metadata read can be skipped entirely. That keeps the common case
    free, since most files match no pattern or already carry a date.
    """
    if dates.from_filename(action.path.name) is None:
        return

    exif = await _read_dates_async(action.path, cfg)
    resolved = dates.resolve(action.path, exif)
    if resolved.source != "filename":
        return
    argv = dates.build_date_write_command(target, resolved, cfg.tools.exiftool)
    if argv is None:
        return
    code, _ = await video.run_command(argv)
    if code == 0:
        result.notes.append(f"creation date inferred from filename ({resolved.exif_value})")


async def _read_dates_async(path: Path, cfg: config.GlobalConfig) -> dict[str, str]:
    """Read just the date tags without blocking the event loop.

    :class:`~media_condenser.probe.Prober` is synchronous, which is fine during planning but
    would stall every other in-flight encode if called from inside a worker.

    The tags come from :data:`media_condenser.dates.METADATA_KEYS` -- the same list
    :func:`media_condenser.dates.resolve` reads back out -- so the two cannot drift.
    """
    argv = [cfg.tools.exiftool, "-json", *(f"-{key}" for key in dates.METADATA_KEYS), str(path)]
    code, stdout, stderr = await probe.run_async(argv)
    if code:
        log.debug("exiftool exit %d reading dates from %s\n%s", code, path, probe.decode(stderr))
    if not stdout.strip():
        return {}
    try:
        parsed = json.loads(stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        log.debug("unparseable exiftool date output for %s", path, exc_info=exc)
        return {}
    return parsed[0] if parsed else {}
