"""The end-of-run summary, in both human and machine-readable form.

Everything here is a *result*, so it goes to stdout -- ``mcon photos/ -o out/ > report.txt``
has to capture it. Logs and the live progress bar go to stderr instead; see
:mod:`media_condenser.logging_setup`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from rich import console, markup, table

from media_condenser import logging_setup, planner


class SummaryFormat(StrEnum):
    """How the result stream (stdout) is rendered.

    Separate from ``--log-format`` because the two streams serve different readers:
    a batch job commonly wants machine-readable results and plain human logs, or
    results only and no logs at all.
    """

    TABLE = "table"
    """Rich tables. The default."""

    JSON = "json"
    """One JSON document, for a pipeline that parses stdout."""

    NONE = "none"
    """Nothing at all, for a job that only cares about the exit code."""


class Outcome(StrEnum):
    """Three buckets, not two.

    'Downgraded' exists because a motion photo that lost a stale video component is
    a success with a caveat, not a failure. Reporting them together makes it
    impossible to see at a glance whether a run actually went well.
    """

    SUCCEEDED = "succeeded"
    DOWNGRADED = "downgraded"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class Record:
    path: Path
    kind: planner.Kind
    outcome: Outcome
    original_size: int = 0
    new_size: int = 0
    detail: str = ""
    notes: list[str] = field(default_factory=list)

    verified: bool = False
    """Whether ``--verify`` actually checked this file's output."""

    verify_checks: int = 0
    verify_problems: list[str] = field(default_factory=list)
    """Non-empty means verification rejected the output, which is why the outcome is
    ``FAILED``: nothing was committed and the original is untouched."""

    @property
    def saved(self) -> int:
        if self.outcome in (Outcome.SUCCEEDED, Outcome.DOWNGRADED):
            return self.original_size - self.new_size
        return 0

    @property
    def size_after(self) -> int:
        """What this file occupies once the run is over.

        Not :attr:`new_size`, which is zero for a failure because nothing was ever
        committed. Summing that column raw would report a failed file as having
        compressed away to nothing -- so a run that failed outright would show the
        best compression rate it ever achieved. For anything not written, the
        original is still sitting there at its original size.
        """
        if self.outcome in (Outcome.SUCCEEDED, Outcome.DOWNGRADED):
            return self.new_size
        return self.original_size


def human_bytes(value: int) -> str:
    negative = value < 0
    size = float(abs(value))
    # The "TB" arm always returns, so the loop cannot fall through.
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            rendered = f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            return f"-{rendered}" if negative else rendered
        size /= 1024
    raise AssertionError("unreachable")


def compression_rate(before: int, after: int) -> str:
    """How much smaller the output is, as a percentage of the input.

    Rendered as a dash rather than ``0%`` when there was nothing to compress, so a
    category that never ran cannot be misread as one that ran and achieved nothing.
    """
    if before <= 0:
        return "-"
    return f"{(before - after) / before:.0%}"


class Reporter:
    """Collects records and renders the run's output."""

    def __init__(self, out: console.Console | None = None) -> None:
        self.console = out or logging_setup.RESULT_CONSOLE
        self.records: list[Record] = []
        self._verify_caveat: str | None = None
        self._verify_requested = False

    def note_verification(self, caveat: str | None) -> None:
        """Record that ``--verify`` ran, and what it could not prove.

        The caveat is the only part of the verify block that is not derivable from
        the records, so it is the only part stored -- the tally itself comes from
        :meth:`verify_totals` and therefore cannot drift from the outcomes.
        """
        self._verify_requested = True
        self._verify_caveat = caveat

    def add(self, record: Record) -> None:
        self.records.append(record)

    # -- dry run -------------------------------------------------------

    def print_plan(
        self,
        actions: list[planner.Action],
        *,
        copies_skipped: bool = False,
        already_done: Mapping[Path, str] | None = None,
    ) -> None:
        """Render the classification table for ``--dry-run``.

        ``copies_skipped`` reflects that a skipped file still gets copied into a
        mirror output tree. A dry run is meant to be an honest preview of what lands
        on disk, so a row saying only "skip" would understate it.

        ``already_done`` maps a source path to the reason its output is already in
        place, from :func:`media_condenser.pipeline.existing_output`. Those rows report the skip
        the run would perform rather than the classification the planner reached,
        which is what makes a rerun preview match the rerun.
        """
        already_done = already_done or {}
        grid = table.Table(title="Planned actions", title_style="bold", header_style="bold cyan")
        grid.add_column("File", overflow="fold")
        grid.add_column("Type")
        grid.add_column("Action")
        grid.add_column("Size", justify="right")
        grid.add_column("Reason", overflow="fold")

        for action in actions:
            verb, reason = _effective(action, already_done)
            if action.path not in already_done and copies_skipped and action.verb is planner.Verb.SKIP:
                reason = f"{reason}; will be copied unchanged"
            # Escaped because both are user data: a file really can be called
            # `my [bold] holiday.jpg`, and rich would either swallow the tag or
            # raise MarkupError on an unmatched closing one.
            grid.add_row(
                markup.escape(action.path.name),
                str(action.kind),
                _verb_markup(verb),
                human_bytes(action.candidate.size),
                markup.escape(reason) + _note_suffix(action.notes),
            )
        self.console.print(grid)

        counts: dict[planner.Verb, int] = defaultdict(int)
        for action in actions:
            counts[_effective(action, already_done)[0]] += 1
        parts = [f"{count} {verb}" for verb, count in sorted(counts.items())]
        self.console.print("[bold]Totals:[/bold] " + ", ".join(parts) if parts else "Nothing to do")

    # -- machine-readable ----------------------------------------------

    @staticmethod
    def plan_as_dict(
        actions: list[planner.Action],
        *,
        copies_skipped: bool = False,
        already_done: Mapping[Path, str] | None = None,
    ) -> dict[str, Any]:
        """The ``--dry-run`` plan, as the JSON counterpart of :meth:`print_plan`."""
        already_done = already_done or {}
        effective = [(action, *_effective(action, already_done)) for action in actions]
        counts: dict[str, int] = defaultdict(int)
        for _, verb, _reason in effective:
            counts[str(verb)] += 1
        return {
            "version": 1,
            "dry_run": True,
            "totals": {"files": len(actions), "by_action": dict(counts)},
            "files": [
                {
                    "path": str(action.path),
                    "kind": str(action.kind),
                    "action": str(verb),
                    "size": action.candidate.size,
                    "reason": reason,
                    "notes": action.notes,
                    "already_present": action.path in already_done,
                    "will_be_copied_unchanged": (
                        copies_skipped and action.verb is planner.Verb.SKIP and action.path not in already_done
                    ),
                }
                for action, verb, reason in effective
            ],
        }

    def as_dict(self) -> dict[str, Any]:
        """The run's results, for ``--report-json``.

        A stable contract for scripts, so it is built from :class:`Record` fields
        rather than by scraping the summary table -- the table exists to be read by
        a person and its columns will move.
        """
        totals = {outcome.value: 0 for outcome in Outcome}
        for record in self.records:
            totals[record.outcome.value] += 1
        # `size_after`, so that `original_bytes - new_bytes == saved_bytes` holds
        # even with failures in the run. Summing the raw `new_size` breaks that
        # identity, and a consumer computing its own rate from those two fields
        # would count every failure as a total saving.
        before = sum(record.original_size for record in self.records)
        after = sum(record.size_after for record in self.records)
        return {
            "version": 1,
            "totals": totals
            | {
                "original_bytes": before,
                "new_bytes": after,
                "saved_bytes": before - after,
                "compression_rate": round((before - after) / before, 4) if before else None,
            },
            "files": [
                {
                    "path": str(record.path),
                    "kind": str(record.kind),
                    "outcome": record.outcome.value,
                    "original_size": record.original_size,
                    "new_size": record.new_size,
                    "saved": record.saved,
                    "detail": record.detail,
                    "notes": record.notes,
                    "verified": record.verified,
                    "verify_problems": record.verify_problems,
                }
                for record in self.records
            ],
            # The caveat key appears only when verification was asked for: a run
            # without --verify has nothing to caveat, and emitting a null there
            # would read as "checked, nothing to report".
            "verify": self.verify_totals() | ({"caveat": self._verify_caveat} if self._verify_requested else {}),
        }

    def verify_totals(self) -> dict[str, Any]:
        """The ``--verify`` tally, derived from the records themselves.

        Derived rather than counted separately so it cannot drift from the outcomes:
        a file that failed verification is a failed *file*, and both numbers come
        from the same place.
        """
        verified = [record for record in self.records if record.verified]
        return {
            "ran": bool(verified),
            "verified": len(verified),
            "problems": sum(1 for record in verified if record.verify_problems),
        }

    # -- summary -------------------------------------------------------

    def print_summary(self) -> None:
        if not self.records:
            self.console.print("[yellow]No media files found.[/yellow]")
            return

        grouped: dict[planner.Kind, dict[Outcome, list[Record]]] = defaultdict(lambda: defaultdict(list))
        for record in self.records:
            grouped[record.kind][record.outcome].append(record)

        # Every number is no_wrap while `Type` is left free to wrap, so that on a
        # narrow terminal the squeeze lands on a label that stays readable split over
        # two lines, rather than truncating a size into something that reads as a
        # smaller number.
        grid = table.Table(title="Summary by media type", title_style="bold", header_style="bold cyan")
        grid.add_column("Type")
        grid.add_column("OK", justify="right", style="green", no_wrap=True)
        grid.add_column("Down", justify="right", style="yellow", no_wrap=True)
        grid.add_column("Skip", justify="right", style="blue", no_wrap=True)
        grid.add_column("Fail", justify="right", style="red", no_wrap=True)
        grid.add_column("Before", justify="right", no_wrap=True)
        grid.add_column("After", justify="right", no_wrap=True)
        grid.add_column("Rate", justify="right", style="green", no_wrap=True)

        for kind in sorted(grouped, key=str):
            buckets = grouped[kind]
            records = [record for bucket in buckets.values() for record in bucket]
            before = sum(record.original_size for record in records)
            after = sum(record.size_after for record in records)
            grid.add_row(
                str(kind),
                str(len(buckets[Outcome.SUCCEEDED])),
                str(len(buckets[Outcome.DOWNGRADED])),
                str(len(buckets[Outcome.SKIPPED])),
                str(len(buckets[Outcome.FAILED])),
                human_bytes(before),
                human_bytes(after),
                compression_rate(before, after),
            )

        totals = dict.fromkeys(Outcome, 0)
        for record in self.records:
            totals[record.outcome] += 1
        before = sum(record.original_size for record in self.records)
        after = sum(record.size_after for record in self.records)
        grid.add_section()
        grid.add_row(
            "[bold]total[/bold]",
            f"[bold]{totals[Outcome.SUCCEEDED]}[/bold]",
            f"[bold]{totals[Outcome.DOWNGRADED]}[/bold]",
            f"[bold]{totals[Outcome.SKIPPED]}[/bold]",
            f"[bold]{totals[Outcome.FAILED]}[/bold]",
            f"[bold]{human_bytes(before)}[/bold]",
            f"[bold]{human_bytes(after)}[/bold]",
            f"[bold]{compression_rate(before, after)}[/bold]",
        )
        self.console.print(grid)

        self._print_detail_list(
            Outcome.DOWNGRADED,
            "Downgraded (processed, with a component or step reduced)",
            "yellow",
        )
        self._print_detail_list(Outcome.FAILED, "Failed", "red")

    def _print_detail_list(self, outcome: Outcome, title: str, style: str) -> None:
        matching = [record for record in self.records if record.outcome is outcome]
        if not matching:
            return
        self.console.print(f"\n[bold {style}]{title}[/bold {style}]")
        for record in matching:
            detail = record.detail or "; ".join(record.notes)
            self.console.print(f"  [{style}]•[/{style}] {markup.escape(record.path.name)}: {markup.escape(detail)}")


_VERB_COLOURS = {
    planner.Verb.SKIP: "blue",
    planner.Verb.RESIZE_IMAGE: "green",
    planner.Verb.TRANSCODE_VIDEO: "green",
    planner.Verb.REBUILD_MOTION_PHOTO: "magenta",
    planner.Verb.FAIL: "red",
}


def _verb_markup(verb: planner.Verb) -> str:
    return f"[{_VERB_COLOURS[verb]}]{verb}[/]"


def _effective(action: planner.Action, already_done: Mapping[Path, str]) -> tuple[planner.Verb, str]:
    """What the run would actually do with this file, and why.

    An output already sitting in place turns any verb into a skip. Both the table
    and its JSON counterpart have to agree on that, so they read it from here.
    """
    done = already_done.get(action.path)
    if done is not None:
        return planner.Verb.SKIP, done
    return action.verb, action.reason


def _note_suffix(notes: list[str]) -> str:
    return f" [yellow]({markup.escape('; '.join(notes))})[/yellow]" if notes else ""
