"""Command line interface."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import pydantic
import typer
from rich import markup

from media_condenser import (
    config,
    discovery,
    logging_setup,
    pipeline,
    planner,
    probe,
    progress,
    report,
    storage,
    verify,
)

app = typer.Typer(
    # Also what registers the shell completion classes: with this off, typer never
    # calls `completion_init()`, so `_MCON_COMPLETE` is dead too and no completion
    # works at all -- not just the --install-completion flag.
    add_completion=True,
    no_args_is_help=True,
    help="Scale images and videos down to save disk space, preserving metadata.",
)
log = logging.getLogger(__name__)

#: Exit code for an interrupted run, by the usual 128+SIGINT convention. Distinct
#: from 1 on purpose: a run that was stopped half-way is not a run that found
#: failures, and a wrapper script has to be able to tell those apart.
EXIT_INTERRUPTED = 130


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(importlib.metadata.version("media_condenser"))
        raise typer.Exit()


@app.command()
def main(
    # Flag names are only spelled out where they have to be: to add a short alias, or
    # where the name typer derives from the parameter is not the one we want
    # (`config_file` would give `--config-file`, `do_verify` would give `--do-verify`,
    # `show_progress` would give `--show-progress`). Everything else takes the derived
    # name, so there is one spelling to keep right instead of two.
    paths: Annotated[
        list[Path],
        typer.Argument(help="Files or directories to process.", exists=True),
    ],
    _version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the current version and exit.",
        ),
    ] = False,
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            # dir_okay=False both rejects a directory up front and tells the shell to
            # offer files here. The same goes for the other path options below.
            dir_okay=False,
            help="Global config file. Defaults to ~/.config/mcon/config.yaml.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Classify everything and print the plan without writing."),
    ] = False,
    do_verify: Annotated[
        # On by default, so a flag pair rather than a switch -- `--no-verify` is the
        # only way to express the other half, the same reason `--progress` is a pair.
        bool,
        typer.Option(
            "--verify/--no-verify",
            help="Check each output before committing it, discarding anything that does not pass.",
        ),
    ] = True,
    strategy: Annotated[
        config.Strategy | None,
        typer.Option(help="Override the storage strategy (copy or replace)."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", file_okay=False, help="Mirror output tree (copy strategy only)."),
    ] = None,
    jobs: Annotated[
        int | None,
        typer.Option("--jobs", "-j", min=1, help="Concurrent encode jobs."),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Only warnings and errors, and no progress bar."),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="-v for debug logs, -vv to include libraries."),
    ] = 0,
    log_level: Annotated[
        logging_setup.LogLevel | None,
        typer.Option(help="Explicit log level. Overrides --verbose and --quiet."),
    ] = None,
    log_format: Annotated[
        logging_setup.LogFormat | None,
        typer.Option(help="Defaults to rich on a terminal and plain when redirected."),
    ] = None,
    show_progress: Annotated[
        # A flag pair with no default, rather than a `--no-progress` switch: this is
        # the one boolean whose default is not simply False but auto-detected, so
        # "force it on" and "force it off" are both real requests and neither can be
        # expressed by a single flag. It also avoids a negatively-named parameter,
        # where `no_progress=False` has to be read twice to mean "show the bar".
        bool | None,
        typer.Option("--progress/--no-progress", help="Force the live progress bar on or off."),
    ] = None,
    summary: Annotated[
        report.SummaryFormat,
        typer.Option(help="Render the end-of-run results as tables, as JSON, or not at all."),
    ] = report.SummaryFormat.TABLE,
    report_json: Annotated[
        Path | None,
        typer.Option(dir_okay=False, help="Write the per-file results as JSON, for scripting."),
    ] = None,
) -> None:
    """Compress every image and video under PATHS."""
    # Logging comes up before anything can fail, so that a bad config file has
    # somewhere to be reported. The real configuration replaces it a few lines down,
    # once the file it is described in has actually been read.
    logging_setup.bootstrap()
    logging_setup.install_tracebacks()

    try:
        cfg = config.load_global_config(config_file)
    except (FileNotFoundError, ValueError) as exc:
        log.error("config error: %s", exc)
        raise typer.Exit(code=2) from exc

    overrides: dict[str, object] = {}
    if strategy is not None:
        overrides["strategy"] = strategy
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if jobs is not None:
        # Re-derive the encoder pool so jobs x pools stays within the core count.
        overrides["jobs"] = jobs
        overrides["encoder_pools"] = None
    if overrides:
        try:
            cfg = config.GlobalConfig.model_validate({**cfg.model_dump(), **overrides})
        except pydantic.ValidationError as exc:
            # Command-line flags go through the same schema as the config file, so
            # cross-field rules (such as --output-dir with --strategy replace) are
            # enforced once and reported the same way from either source.
            log.error("config error: %s", _first_error(exc))
            raise typer.Exit(code=2) from exc

    try:
        resolved_format = logging_setup.configure(
            cfg.logging,
            level=_resolve_level(log_level, verbose, quiet),
            # -vv opens up the root logger too, which is where anything not written
            # by this tool ends up.
            root_level=logging_setup.LogLevel.DEBUG if verbose >= 2 else None,
            log_format=log_format,
        )
    except Exception as exc:
        # dictConfig raises a bare ValueError for anything it dislikes, and an
        # unusable logging section should be reported like any other config error
        # rather than as a traceback.
        log.error("config error: unusable logging configuration: %s", exc)
        raise typer.Exit(code=2) from exc

    if cfg.strategy is config.Strategy.COPY and cfg.output_dir is None:
        log.error(
            "config error: the copy strategy has nowhere to write. Pass --output-dir/-o DIR "
            "(or set output_dir in the config file) to build a mirror tree, or pass "
            "--strategy replace to overwrite the originals in place."
        )
        raise typer.Exit(code=2)

    display = progress.RunProgress(
        enabled=logging_setup.progress_enabled(
            progress=show_progress,
            quiet=quiet,
            log_format=resolved_format,
        )
    )

    if cfg.strategy is config.Strategy.REPLACE and not dry_run:
        log.warning("replace strategy: originals will be overwritten (each via a temp file, committed only on success)")

    resolver = config.RulesResolver(cfg.rules, paths)
    prober = probe.Prober(cfg.tools)
    reporter = report.Reporter()
    started = time.monotonic()

    with display:
        actions = _scan_and_plan(paths, resolver, prober, display)

    if not actions:
        log.warning("no media files found under %s", ", ".join(str(p) for p in paths))
        # Still emitted, so that a batch consumer parsing stdout gets a document
        # saying zero files rather than nothing at all to parse.
        _emit_summary(reporter, [], summary)
        return

    # The same scan root must be used for the plan and the run, or an --output-dir
    # mirror run would preview and verify a path nothing was ever written to.
    scan_root = _mirror_root(paths)

    if dry_run:
        # Checked here rather than assumed: the destination state is what decides
        # whether the run does any work, and it costs a stat and a probe per file.
        with display:
            asyncio.run(
                pipeline.prefetch_existing_outputs(actions, cfg, prober, scan_root, concurrency=config.cpu_count())
            )
            display.phase("Checking outputs", total=len(actions))
            already_done: dict[Path, str] = {}
            for action in actions:
                reason = pipeline.existing_output(action, cfg, prober, scan_root)
                if reason is not None:
                    already_done[action.path] = reason
                display.advance()
        if already_done:
            log.info("%d of %d files are already present in the output tree", len(already_done), len(actions))
        _emit_plan(
            reporter,
            actions,
            summary,
            copies_skipped=pipeline.mirrors_skipped_files(cfg),
            already_done=already_done,
        )
        return

    removed = _clear_stale_temp_files(paths, cfg)
    if removed:
        # A warning rather than a note: these are half-written outputs from a run
        # that did not finish, which is worth knowing about even under --quiet.
        log.warning("removed %d leftover temp file(s) from an interrupted run", removed)

    with display:
        # The same warm-up the dry run does, for the same reason: `_already_produced`
        # probes every output that passes the stat gate, and doing that lazily from
        # inside the workers is one exiftool or ffprobe launch per file throttled to
        # `cfg.jobs`. A rerun over a populated output tree spends its whole wall clock
        # there. Against an empty one the stat gate rejects everything and this costs
        # nothing.
        asyncio.run(pipeline.prefetch_existing_outputs(actions, cfg, prober, scan_root, concurrency=config.cpu_count()))
        interrupted = asyncio.run(
            pipeline.execute_all(
                actions,
                cfg,
                prober,
                reporter,
                scan_root=scan_root,
                display=display,
                verify_outputs=do_verify,
            )
        )

    if do_verify:
        reporter.note_verification(verify.caveat(actions))
    verify_problems = reporter.verify_totals()["problems"]

    log.info("finished in %s", _duration(time.monotonic() - started))
    if report_json is not None:
        _write_report(report_json, reporter)

    # Every result is emitted here, after the last log line, so that nothing can
    # scroll it away. --verify in particular logs a line per file, and printing the
    # summary before it ran meant the table was gone by the time the run ended.
    _emit_summary(reporter, actions, summary)

    if interrupted:
        raise typer.Exit(code=EXIT_INTERRUPTED)

    # A verification failure is a failure of the run. Asking for --verify and getting
    # a zero exit back is what makes `mcon --verify ... && rm -rf originals` unsafe.
    failures = sum(1 for record in reporter.records if record.outcome is report.Outcome.FAILED)
    if failures or verify_problems:
        raise typer.Exit(code=1)


def _scan_and_plan(
    paths: list[Path],
    resolver: config.RulesResolver,
    prober: probe.Prober,
    display: progress.RunProgress,
) -> list[planner.Action]:
    """Scan the tree and classify every file in it."""
    display.phase("Scanning")
    candidates = discovery.walk(paths, resolver)
    if not candidates:
        return []
    log.info("found %d media files", len(candidates))

    # Every measured property planning needs, read in bulk before any of it is asked
    # for. This is the phase that used to be the whole cost of a scan: one ffprobe or
    # exiftool launch per file, serially, with the interpreter startup dominating the
    # actual reading. Classification itself then runs off the warm cache.
    display.phase("Probing", total=len(candidates))
    asyncio.run(
        planner.prefetch(
            candidates,
            prober,
            # Deliberately not `cfg.jobs`, which is calibrated for encodes: it is small
            # because each ffmpeg job then grabs several cores of its own. A probe reads
            # a header and exits, so the same ceiling would just leave most of the
            # machine idle during the phase that used to be the whole wait.
            concurrency=config.cpu_count(),
            on_progress=display.advance,
        )
    )

    # A loop rather than a comprehension so the bar can advance: on a large library the
    # static spinner this replaces made it look like the tool had hung. It moves fast
    # now that the probes are done, but anything the prefetch could not read is probed
    # here instead, so this is not guaranteed to be quick.
    display.phase("Planning", total=len(candidates))
    actions = []
    for candidate in candidates:
        actions.append(planner.plan(candidate, prober))
        display.advance()
    return actions


def _resolve_level(explicit: logging_setup.LogLevel | None, verbose: int, quiet: bool) -> logging_setup.LogLevel | None:
    """Turn the verbosity flags into a level, or ``None`` to leave the config alone.

    ``None`` rather than a default matters: it is what lets someone set a level in
    their config file and have it survive a run with no verbosity flags on it.
    """
    if explicit is not None:
        return explicit
    if verbose:
        return logging_setup.LogLevel.DEBUG
    if quiet:
        return logging_setup.LogLevel.WARNING
    return None


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def _emit_json(payload: dict[str, object]) -> None:
    """Write one JSON document to stdout, bypassing rich.

    Deliberately not ``Console.print_json``: rich renders JSON through ``Syntax``,
    which is width-aware, so a long file path can be wrapped or clipped and the
    output stops being parseable. A machine-readable stream must not depend on how
    wide the terminal happens to be.
    """
    print(json.dumps(payload, indent=2), file=logging_setup.RESULT_CONSOLE.file)


def _emit_plan(
    reporter: report.Reporter,
    actions: list[planner.Action],
    summary: report.SummaryFormat,
    *,
    copies_skipped: bool,
    already_done: Mapping[Path, str],
) -> None:
    """Render the ``--dry-run`` plan in the requested format."""
    if summary is report.SummaryFormat.NONE:
        return
    if summary is report.SummaryFormat.JSON:
        _emit_json(reporter.plan_as_dict(actions, copies_skipped=copies_skipped, already_done=already_done))
        return
    reporter.print_plan(actions, copies_skipped=copies_skipped, already_done=already_done)


def _emit_summary(reporter: report.Reporter, actions: list[planner.Action], summary: report.SummaryFormat) -> None:
    """Render the end-of-run results in the requested format.

    Includes the ``--verify`` tally, which is a result like any other: it belongs
    with the summary rather than buried in the log lines verification produces.
    """
    if summary is report.SummaryFormat.NONE:
        return
    if summary is report.SummaryFormat.JSON:
        _emit_json(reporter.as_dict())
        return

    reporter.print_summary()
    totals = reporter.verify_totals()
    if not totals["ran"]:
        return

    verified = totals["verified"]
    problems = totals["problems"]
    if verified == 0:
        logging_setup.RESULT_CONSOLE.print("  [dim]nothing to verify[/dim]")
        return
    style = "red" if problems else "green"
    logging_setup.RESULT_CONSOLE.print(
        f"\n  [{style}]{verified - problems}/{verified} outputs verified clean[/{style}]"
    )
    note = verify.caveat(actions)
    if note:
        logging_setup.RESULT_CONSOLE.print(f"  [yellow]Note:[/yellow] {markup.escape(note)}")


def _write_report(path: Path, reporter: report.Reporter) -> None:
    """Write the machine-readable run report."""
    try:
        path.write_text(json.dumps(reporter.as_dict(), indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        # Reported, not fatal: the run itself already happened, and failing it now
        # would misreport work that actually succeeded.
        log.error("could not write the run report to %s: %s", path, exc)
    else:
        log.info("wrote the run report to %s", path)


def _first_error(exc: pydantic.ValidationError) -> str:
    """The first validation message, without pydantic's multi-line framing."""
    for error in exc.errors():
        return str(error.get("msg", exc)).removeprefix("Value error, ")
    return str(exc)


def _mirror_root(paths: list[Path]) -> Path | None:
    """The directory an ``--output-dir`` mirror tree is built relative to.

    With one directory argument this is simply that directory. With several, it is
    their nearest common ancestor -- *not* nothing: falling back to each file's own
    parent (which is what a ``None`` root does) flattens every subtree into the output
    root, so ``2020/IMG_0001.jpg`` and ``2021/IMG_0001.jpg`` both commit to
    ``out/IMG_0001.jpg`` and the second overwrites the first while both are reported
    as succeeded. Anchoring on the common ancestor keeps the relative paths distinct,
    which is what makes the mirror faithful.

    File arguments contribute their parent directory, so naming a handful of files
    still writes them side by side rather than recreating their whole path.
    """
    if not paths:
        return None
    bases = [path.resolve() if path.is_dir() else path.resolve().parent for path in paths]
    if len(bases) == 1:
        return bases[0]
    try:
        return Path(os.path.commonpath(bases))
    except ValueError:
        # No shared ancestor at all (different Windows drives). The per-file fallback
        # in `plan_destination` applies, and `execute_all` refuses any collision it
        # produces rather than overwriting.
        return None


def _clear_stale_temp_files(paths: list[Path], cfg: config.GlobalConfig) -> int:
    """Remove leftover ``.mcon_tmp_*`` files before the run starts.

    An interrupted run (Ctrl-C, OOM, a killed encoder) leaves half-written temp files
    beside their destinations. They carry real JPEG/MP4 magic bytes, so the walk of a
    later run would otherwise treat them as library media -- resizing one and
    committing it in place under its temp name, or copying it into the mirror tree as
    junk, and reporting failures against filenames that match no real photo. The walk
    now refuses them by name as well; this is what stops them accumulating.
    """
    roots = [path if path.is_dir() else path.parent for path in paths]
    if cfg.output_dir is not None:
        roots.append(cfg.output_dir)
    removed = 0
    for root in roots:
        if root.is_dir():
            removed += storage.cleanup_stale(root)
    return removed


if __name__ == "__main__":
    app()
