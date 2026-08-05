"""The live progress display.

One ``Live`` region holding two stacked ``Progress`` tables: an overall bar for the
current phase, and one transient row per file currently being encoded. It draws on
:data:`~media_condenser.logging_setup.LOG_CONSOLE`, the same console the log handler writes to,
which is what makes log lines scroll *above* the block instead of through it.

Deliberately not part of the logging config: a ``dictConfig`` describes handlers,
formatters and levels for log records, and none of those describe a progress bar.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Self

from rich import console, live, text
from rich import progress as rich_progress

from media_condenser import logging_setup, report

NAME_WIDTH = 34
"""Basenames are truncated to this, so a long filename cannot push the size column
off the edge and reflow the whole block on every update."""


class UnitColumn(rich_progress.ProgressColumn):
    """``6.1/14.2 GB`` for byte-weighted phases, ``312/807`` for counted ones."""

    def render(self, task: rich_progress.Task) -> text.Text:
        if task.total is None:
            return text.Text("")
        if task.fields.get("unit") == "bytes":
            done, total = report.human_bytes(int(task.completed)), report.human_bytes(int(task.total))
        else:
            done, total = str(int(task.completed)), str(int(task.total))
        return text.Text(f"{done}/{total}", style="progress.download")


class SavedColumn(rich_progress.ProgressColumn):
    """Running total of the space reclaimed so far."""

    def render(self, task: rich_progress.Task) -> text.Text:
        saved = int(task.fields.get("saved") or 0)
        if not saved:
            return text.Text("")
        return text.Text(f"saved {report.human_bytes(saved)}", style="green")


def _truncate(name: str) -> str:
    if len(name) <= NAME_WIDTH:
        return name
    return f"{name[: NAME_WIDTH - 4]}...{name[-1:]}"


class RunProgress:
    """The live display for one run, or a no-op stand-in when it is disabled.

    Every method is safe to call when disabled, so callers never branch on it. That
    matters because the alternative -- ``if progress is not None`` at each of the
    eight call sites -- is exactly the shape the old inline bar had, and it is what
    let the per-file description go stale in the first place.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._overall = rich_progress.Progress(
            # The style goes in the argument rather than as a `[progress.description]`
            # tag in the text, because markup is off (see the note below) and the tag
            # would otherwise render as literal characters.
            rich_progress.TextColumn("{task.description}", style="progress.description", markup=False),
            rich_progress.BarColumn(bar_width=None),
            UnitColumn(),
            rich_progress.TimeElapsedColumn(),
            rich_progress.TextColumn("eta", style="dim"),
            rich_progress.TimeRemainingColumn(),
            SavedColumn(),
            console=logging_setup.LOG_CONSOLE,
        )
        # A second table because the columns differ: the overall bar and a per-file
        # row cannot share one layout.
        self._workers = rich_progress.Progress(
            rich_progress.SpinnerColumn(style="cyan"),
            rich_progress.TextColumn("{task.fields[verb]:<9}", style="cyan", markup=False),
            # markup=False throughout: these render user-supplied filenames, and
            # `my [bold] holiday.jpg` would otherwise be parsed as console markup.
            rich_progress.TextColumn("{task.description}", markup=False),
            rich_progress.TextColumn("{task.fields[size]:>10}", style="dim", markup=False),
            console=logging_setup.LOG_CONSOLE,
        )
        self._live = live.Live(
            console.Group(self._overall, self._workers),
            console=logging_setup.LOG_CONSOLE,
            transient=True,
            refresh_per_second=10,
        )
        self._task: rich_progress.TaskID | None = None
        self._saved = 0

    # -- lifetime ------------------------------------------------------

    def __enter__(self) -> Self:
        if self.enabled:
            self._live.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.enabled:
            self._live.stop()

    # -- phases --------------------------------------------------------

    def phase(self, title: str, *, total: float | None = None, unit: str = "files") -> None:
        """Start a new phase, replacing the previous one's bar.

        ``total=None`` leaves the bar indeterminate, for work whose size is not
        known up front (the initial directory walk).
        """
        if not self.enabled:
            return
        if self._task is not None:
            self._overall.remove_task(self._task)
        self._task = self._overall.add_task(title, total=total, unit=unit, saved=self._saved)

    def advance(self, amount: float = 1) -> None:
        if self.enabled and self._task is not None:
            self._overall.advance(self._task, amount)

    def add_saved(self, saved: int) -> None:
        """Fold one file's saving into the running total shown on the bar."""
        if saved <= 0:
            return
        self._saved += saved
        if self.enabled and self._task is not None:
            self._overall.update(self._task, saved=self._saved)

    # -- per-file rows -------------------------------------------------

    @contextmanager
    def file(self, verb: str, name: str, size: int) -> Iterator[None]:
        """Show a row for one in-flight file, for as long as it is being worked on.

        A row per worker rather than one shared description: with ``jobs`` files in
        flight, a single description only ever showed whichever worker updated it
        last, which read as though that one file had been stuck for minutes.
        """
        if not self.enabled:
            yield
            return
        task = self._workers.add_task(
            _truncate(name),
            total=None,
            verb=verb,
            size=report.human_bytes(size),
        )
        try:
            yield
        finally:
            self._workers.remove_task(task)
