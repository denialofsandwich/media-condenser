"""Per-media-type processing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HandlerResult:
    """Outcome of processing one file."""

    ok: bool
    original_size: int = 0
    new_size: int = 0
    error: str = ""
    notes: list[str] = field(default_factory=list)
    """Components that had to be degraded. A result with notes but ``ok=True`` is
    reported as 'downgraded' -- explicitly distinct from a failure."""


def tail(text: str, limit: int = 300) -> str:
    """The last meaningful line of a tool's stderr, for a one-line error message.

    Blank lines are dropped because ffmpeg and exiftool both end their output with
    one, and the last *line* is the useful one: the preceding context is already in
    the debug log that :func:`media_condenser.handlers.video.run_command` writes.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1][:limit] if lines else ""
