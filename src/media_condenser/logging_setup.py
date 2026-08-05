"""Logging setup: two consoles, a themed :class:`~rich.logging.RichHandler`, and
the ``logging.config.dictConfig`` document that wires them together.

Two consoles, not one, because a run produces two different kinds of output:

- **results** (the plan table, the summary, the ``--verify`` tally) go to stdout,
  so ``mcon photos/ -o out/ > report.txt`` captures them
- **logs and the progress bar** go to stderr, so ``2>/dev/null`` silences the
  noise without losing the results

The progress display draws on :data:`LOG_CONSOLE` too. That shared instance is the
whole mechanism behind "log lines above, progress bar pinned below": rich's ``Live``
intercepts writes to its own console and renders them above the live region. A
handler pointed at a *different* stderr console would tear straight through the bar.
"""

from __future__ import annotations

import json
import logging
import logging.config
from copy import deepcopy
from enum import StrEnum
from importlib import resources
from typing import Any, ClassVar, Final

import yaml
from rich import console, highlighter, theme
from rich import logging as rich_logging


class LogFormat(StrEnum):
    """How log records are rendered. Each name is also a handler in the dictConfig."""

    RICH = "rich"
    """Colour, aligned columns, and a live progress bar. The default on a terminal."""

    PLAIN = "plain"
    """Timestamped single lines. The default when stderr is not a terminal."""

    JSON = "json"
    """One JSON object per line, for log shipping."""


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


#: Handler names that carry a format. Exactly one is wired to the loggers at a time;
#: any other handler a user has configured is left alone. See :func:`_swap_format`.
LOG_FORMATS: Final = frozenset(fmt.value for fmt in LogFormat)

log = logging.getLogger(__name__)

DEFAULT_LOGGING_FILE: Final = "default_logging.yaml"

LOG_THEME: Final = theme.Theme(
    {
        "logging.level.debug": "dim cyan",
        "logging.level.info": "green",
        "logging.level.warning": "yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "reverse bold red",
        # Styles for LogHighlighter's named groups, minus the "log." prefix.
        "log.filename": "bold cyan",
        "log.size": "magenta",
        "log.dimensions": "magenta",
        "log.percent": "magenta",
        "log.tool": "dim",
    }
)


class LogHighlighter(highlighter.RegexHighlighter):
    """Colour for log messages, without console markup.

    Markup is deliberately *not* enabled on the handler: filenames are user data in
    a photo library, and rich would eat ``my [bold] holiday.jpg`` and raise
    ``MarkupError`` on ``a[/red]b.jpg``. Highlighting runs over already-rendered
    text instead, so no filename can change how a message is parsed.
    """

    base_style = "log."
    highlights: ClassVar[list[str]] = [
        r"(?P<filename>[^\s/\\]+\.(?i:jpe?g|png|heic|heif|avif|webp|gif|tiff?|dng|cr[23]|mp4|mov|m4v|mkv|avi|3gp))\b",
        r"(?P<size>\b\d+(?:\.\d+)?\s?(?:B|KB|MB|GB|TB)\b)",
        r"(?P<dimensions>\b\d{2,5}x\d{2,5}\b)",
        r"(?P<percent>\b\d+(?:\.\d+)?%)",
        r"^(?P<tool>\$ \S+)",
    ]


#: Extras that :class:`JsonFormatter` forwards. A whitelist rather than "everything
#: not on a stock LogRecord", so the batch output stays a stable contract instead of
#: quietly gaining fields whenever a call site passes something new.
JSON_EXTRA_FIELDS: Final = (
    "file",
    "kind",
    "outcome",
    "original_size",
    "new_size",
    "saved",
    "phase",
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log shipping and batch pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in JSON_EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


RESULT_CONSOLE: Final = console.Console()
"""stdout. Tables and verification results -- the output a pipe should capture."""

LOG_CONSOLE: Final = console.Console(stderr=True, theme=LOG_THEME)
"""stderr. Log records and the live progress display, which must share a console."""


def rich_handler() -> rich_logging.RichHandler:
    """Build the default handler. Referenced from the dictConfig ``()`` key."""
    return rich_logging.RichHandler(
        console=LOG_CONSOLE,
        highlighter=LogHighlighter(),
        markup=False,
        rich_tracebacks=True,
        show_path=False,
        omit_repeated_times=True,
        log_time_format="[%X]",
    )


def default_logging_config() -> dict[str, Any]:
    """A fresh copy of the packaged dictConfig document."""
    text = resources.files(__package__).joinpath(DEFAULT_LOGGING_FILE).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def resolve_format(explicit: LogFormat | None) -> LogFormat:
    """Pick the log format: explicit flag, else rich on a terminal and plain in a pipe."""
    if explicit is not None:
        return explicit
    return LogFormat.RICH if LOG_CONSOLE.is_terminal else LogFormat.PLAIN


def _swap_format(handlers: Any, chosen: LogFormat) -> Any:
    """Point the format-carrying handler at ``chosen``, leaving any others alone.

    Only the three names in :data:`LOG_FORMATS` are swapped, so a user who added
    their own file handler keeps it under every ``--log-format``. A handler list
    naming none of them is left untouched: that user has taken full control.
    """
    if not isinstance(handlers, list):
        return handlers
    swapped = [chosen.value if name in LOG_FORMATS else name for name in handlers]
    # dict.fromkeys rather than set(), to keep the declared order stable.
    return list(dict.fromkeys(swapped))


def configure(
    logging_config: dict[str, Any],
    *,
    level: LogLevel | None = None,
    root_level: LogLevel | None = None,
    log_format: LogFormat | None = None,
) -> LogFormat:
    """Apply ``logging_config``, with the command line layered on top.

    Returns the format that ended up in effect. Raises ``ValueError`` if the
    document is not a usable dictConfig, which the CLI reports as a config error.
    """
    resolved = deepcopy(logging_config)
    chosen = resolve_format(log_format)

    loggers = resolved.setdefault("loggers", {})
    package_logger = loggers.setdefault(__package__, {})
    package_logger["handlers"] = _swap_format(package_logger.get("handlers", [chosen.value]), chosen)
    if level is not None:
        package_logger["level"] = level.value

    root = resolved.setdefault("root", {})
    root["handlers"] = _swap_format(root.get("handlers", [chosen.value]), chosen)
    if root_level is not None:
        root["level"] = root_level.value

    logging.config.dictConfig(resolved)
    return chosen


def bootstrap() -> None:
    """Configure logging from the packaged defaults alone.

    Called before the config file is read, so that a failure to read it has
    somewhere to go. The real configuration replaces this once the file has loaded.
    """
    configure(default_logging_config())
    logging.captureWarnings(True)


def progress_enabled(*, progress: bool | None, quiet: bool, log_format: LogFormat) -> bool:
    """Whether to draw the live progress display.

    ``progress`` is the ``--progress/--no-progress`` pair: ``None`` means neither was
    given, so the decision is made here. Unattended, the bar is drawn only when stderr
    is a terminal rendering the rich format, because a display redrawing itself into a
    pipe or a CI log is noise rather than feedback.

    ``--progress`` overrides that -- it is how you get a bar alongside plain logs, or
    under ``-q`` -- but it cannot conjure a terminal. Two combinations are impossible
    rather than merely unusual, and both are refused out loud: a live region needs a
    terminal to redraw in, and interleaving one with JSON records would leave the
    stream unparseable. Saying so beats accepting the flag and doing nothing.

    Deliberately not a logging concern: a dictConfig describes handlers, formatters
    and levels for log *records*, and a progress bar is none of those.
    """
    if progress:
        if not LOG_CONSOLE.is_terminal:
            log.warning("ignoring --progress: a live display needs stderr to be a terminal")
            return False
        if log_format is LogFormat.JSON:
            log.warning("ignoring --progress: a live display would corrupt the JSON log stream")
            return False
        return True
    if progress is False or quiet:
        return False
    return log_format is LogFormat.RICH and LOG_CONSOLE.is_terminal


def install_tracebacks() -> None:
    """Render unexpected crashes through rich, on the log console.

    Suppresses typer's own frames (which vendor click as ``typer._click``): they sit
    between the entry point and every one of this tool's frames, and say nothing
    about what went wrong.
    """
    import typer
    from rich import traceback as rich_traceback

    rich_traceback.install(console=LOG_CONSOLE, show_locals=False, suppress=[typer])
