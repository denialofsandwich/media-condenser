"""Logging configuration and the log/result stream split.

The behaviours pinned here are the ones that are silent when they break: a config
fragment that quietly discards the packaged handlers, a filename that a console
markup parser eats, or a progress bar redrawing itself into a pipe.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import fixtures
import pytest
from typer import testing

from media_condenser import cli, config, logging_setup, planner, progress, report


@pytest.fixture(autouse=True)
def _restore_logging():
    """Each test reconfigures the root logger; put it back for the next one."""
    yield
    logging_setup.bootstrap()


def force_terminal(monkeypatch, value: bool) -> None:
    """Make the log console claim to be (or not be) a terminal.

    ``Console.is_terminal`` is a read-only property that consults
    ``_force_terminal`` first, which is the documented override.
    """
    monkeypatch.setattr(logging_setup.LOG_CONSOLE, "_force_terminal", value)


# ---------------------------------------------------------------------------
# The packaged dictConfig, and merging a user's fragment onto it
# ---------------------------------------------------------------------------


def test_the_packaged_config_is_a_usable_dictconfig() -> None:
    """It ships as package data, so a wheel that omits it must fail loudly here."""
    cfg = logging_setup.default_logging_config()
    assert set(cfg["handlers"]) == {"rich", "plain", "json"}
    assert logging_setup.configure(cfg) in tuple(logging_setup.LogFormat)


def test_a_fragment_is_merged_onto_the_defaults_not_substituted_for_them() -> None:
    """Otherwise setting one level means restating every formatter and handler.

    A fragment on its own is not a valid dictConfig at all -- it has no ``version``
    and no handlers -- so the obvious thing to write in a config file would fail
    outright if the merge were a replacement.
    """
    cfg = config.GlobalConfig.model_validate({"logging": {"loggers": {"media_condenser.probe": {"level": "DEBUG"}}}})

    assert cfg.logging["loggers"]["media_condenser.probe"]["level"] == "DEBUG"
    assert cfg.logging["version"] == 1
    assert set(cfg.logging["handlers"]) == {"rich", "plain", "json"}
    # The default entry for `media_condenser` itself survives alongside the added one.
    assert cfg.logging["loggers"]["media_condenser"]["handlers"] == ["rich"]


def test_a_users_own_handler_survives_every_log_format() -> None:
    """This is what stands in for a --log-file flag, so it has to hold.

    Only the format-carrying handler is swapped; anything else the user wired up is
    left alone.
    """
    cfg = config.GlobalConfig.model_validate(
        {
            "logging": {
                "handlers": {"audit": {"class": "logging.NullHandler"}},
                "loggers": {"media_condenser": {"handlers": ["rich", "audit"]}},
            }
        }
    )
    logging_setup.configure(cfg.logging, log_format=logging_setup.LogFormat.JSON)

    handlers = logging.getLogger("media_condenser").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)
    assert any(getattr(h, "formatter", None).__class__ is logging_setup.JsonFormatter for h in handlers)


def test_an_unusable_logging_section_is_a_config_error_not_a_traceback(tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("logging:\n  handlers:\n    rich:\n      class: no.such.Handler\n")
    workdir = tmp_path / "lib"
    workdir.mkdir()

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "-c", str(config_file)])

    assert result.exit_code == 2, result.output
    assert "logging configuration" in result.output


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------


def test_the_format_defaults_to_plain_when_stderr_is_not_a_terminal(monkeypatch) -> None:
    """Rich's live redraws are feedback on a terminal and noise in a log file."""
    force_terminal(monkeypatch, False)
    assert logging_setup.resolve_format(None) is logging_setup.LogFormat.PLAIN
    force_terminal(monkeypatch, True)
    assert logging_setup.resolve_format(None) is logging_setup.LogFormat.RICH


def test_json_records_are_one_object_per_line_with_the_documented_extras(capsys) -> None:
    logging_setup.configure(
        logging_setup.default_logging_config(),
        level=logging_setup.LogLevel.INFO,
        log_format=logging_setup.LogFormat.JSON,
    )
    logging.getLogger("media_condenser.test").info(
        "succeeded %s",
        "IMG_0001.jpg",
        extra={"file": "/lib/IMG_0001.jpg", "outcome": "succeeded", "saved": 4096},
    )

    line = capsys.readouterr().err.strip()
    payload = json.loads(line)
    assert payload["message"] == "succeeded IMG_0001.jpg"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "media_condenser.test"
    assert payload["outcome"] == "succeeded"
    assert payload["saved"] == 4096


def test_an_unlisted_extra_stays_out_of_the_json(capsys) -> None:
    """The batch contract is a whitelist, so it cannot drift as call sites change."""
    logging_setup.configure(logging_setup.default_logging_config(), log_format=logging_setup.LogFormat.JSON)
    logging.getLogger("media_condenser.test").warning("hello", extra={"internal_detail": "should not ship"})

    assert "internal_detail" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Filenames are user data, not console markup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "my [bold] holiday.jpg",  # rich would silently swallow the tag
        "/photos/[/red]b.jpg",  # and raise MarkupError on this one
        "IMG_[1].jpg",
    ],
)
def test_a_filename_is_never_parsed_as_console_markup(capsys, name: str) -> None:
    """A photo library is full of names rich would try to interpret.

    Markup is off on the handler for exactly this reason; colour comes from the
    highlighter, which runs over rendered text and cannot change how it parses.
    """
    logging_setup.configure(logging_setup.default_logging_config(), log_format=logging_setup.LogFormat.PLAIN)
    logging.getLogger("media_condenser.test").info("resized %s", name)

    assert name in capsys.readouterr().err


def test_the_summary_renders_a_markup_shaped_filename_intact(capsys) -> None:
    """The result tables interpolate names into markup strings, so they escape too."""
    reporter = report.Reporter()
    reporter.add(
        report.Record(
            path=Path("/lib/my [bold] holiday.jpg"),
            kind=planner.Kind.IMAGE,
            outcome=report.Outcome.FAILED,
            original_size=100,
            detail="something [/red] went wrong",
        )
    )
    reporter.print_summary()

    out = capsys.readouterr().out
    assert "my [bold] holiday.jpg" in out
    assert "[/red]" in out


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


def auto(
    monkeypatch, *, terminal: bool, quiet: bool = False, fmt: logging_setup.LogFormat = logging_setup.LogFormat.RICH
) -> bool:
    """What the display does when neither --progress nor --no-progress was given."""
    force_terminal(monkeypatch, terminal)
    return logging_setup.progress_enabled(progress=None, quiet=quiet, log_format=fmt)


def test_progress_is_off_unless_a_terminal_is_there_to_draw_on(monkeypatch) -> None:
    assert not auto(monkeypatch, terminal=False)
    assert auto(monkeypatch, terminal=True)
    assert not auto(monkeypatch, terminal=True, quiet=True)
    # Interleaving redraws with JSON lines would corrupt the JSON stream.
    assert not auto(monkeypatch, terminal=True, fmt=logging_setup.LogFormat.JSON)


def test_the_progress_flag_pair_overrides_the_auto_detection(monkeypatch) -> None:
    """Both directions are real requests, which is why it is a pair and not a switch.

    ``--progress`` forces the bar on where it would be off (a pipe, or under -q), and
    ``--no-progress`` forces it off on a terminal where it would be on.
    """
    force_terminal(monkeypatch, True)
    # On a terminal, forced on where the heuristic would have said no.
    assert logging_setup.progress_enabled(progress=True, quiet=True, log_format=logging_setup.LogFormat.RICH)
    assert logging_setup.progress_enabled(progress=True, quiet=False, log_format=logging_setup.LogFormat.PLAIN)
    # And forced off where it would have said yes.
    assert not logging_setup.progress_enabled(progress=False, quiet=False, log_format=logging_setup.LogFormat.RICH)


@pytest.mark.parametrize(
    ("terminal", "fmt", "expected"),
    [
        (False, logging_setup.LogFormat.PLAIN, "needs stderr to be a terminal"),
        (True, logging_setup.LogFormat.JSON, "corrupt the JSON log stream"),
    ],
)
def test_an_impossible_progress_request_is_refused_out_loud(
    monkeypatch, caplog, terminal: bool, fmt: logging_setup.LogFormat, expected: str
) -> None:
    """Rich cannot paint a live region into a pipe, and a bar would break JSON.

    Both were previously accepted and then quietly did nothing, which is the failure
    mode this codebase refuses elsewhere -- so they say so instead.
    """
    force_terminal(monkeypatch, terminal)
    with caplog.at_level("WARNING", logger="media_condenser.logging_setup"):
        assert not logging_setup.progress_enabled(progress=True, quiet=False, log_format=fmt)
    assert expected in caplog.text


def test_a_disabled_display_is_a_no_op_at_every_call_site() -> None:
    """Callers never branch on `enabled`, so every method has to tolerate it."""
    display = progress.RunProgress(enabled=False)
    with display:
        display.phase("Compressing", total=10, unit="bytes")
        display.advance(5)
        display.add_saved(1024)
        with display.file("resize", "IMG_0001.jpg", 2048):
            pass


def test_the_display_can_be_reopened_for_each_phase() -> None:
    """Scanning, compressing and verifying each enter the block separately.

    Rich allows only one live display at a time, so a leaked one from an earlier
    phase would make the next `with` fail rather than degrade.
    """
    display = progress.RunProgress(enabled=True)
    for phase in ("Scanning", "Compressing", "Verifying"):
        with display:
            display.phase(phase, total=1)
            display.advance()


# ---------------------------------------------------------------------------
# The stream split
# ---------------------------------------------------------------------------


def test_results_go_to_stdout_and_logs_to_stderr(tmp_path) -> None:
    """`mcon photos/ -o out/ > report.txt` has to capture the summary and nothing else."""
    workdir = tmp_path / "lib"
    workdir.mkdir()
    shutil.copy2(fixtures.IMAGES / fixtures.IMAGE_LANDSCAPE, workdir / fixtures.IMAGE_LANDSCAPE)

    result = testing.CliRunner().invoke(
        cli.app, [str(workdir), "-o", str(tmp_path / "out"), "--dry-run", "--log-format", "plain"]
    )

    assert result.exit_code == 0, result.output
    assert "Planned actions" in result.stdout
    assert "Planned actions" not in result.stderr
    assert "found 1 media files" in result.stderr
    assert "found 1 media files" not in result.stdout


def test_quiet_drops_the_narration_but_keeps_the_warnings(tmp_path) -> None:
    """-q is for turning down the noise, not for hiding the things that went wrong.

    Two invocations because the two halves need different runs to provoke: a dry run
    over real files narrates and warns about nothing, and the cheapest genuine
    warning -- an empty tree -- has no narration to suppress.
    """
    workdir = tmp_path / "lib"
    workdir.mkdir()
    for name in (fixtures.IMAGE_LANDSCAPE, fixtures.IMAGE_PORTRAIT):
        shutil.copy2(fixtures.IMAGES / name, workdir / name)
    out = str(tmp_path / "out")

    full = testing.CliRunner().invoke(cli.app, [str(workdir), "-o", out, "--dry-run", "-q"])
    assert full.exit_code == 0, full.output
    assert "found 2 media files" not in full.stderr  # INFO, suppressed
    assert "Planned actions" in full.stdout  # a result, never suppressed

    empty = tmp_path / "empty"
    empty.mkdir()
    warned = testing.CliRunner().invoke(cli.app, [str(empty), "-o", out, "--dry-run", "-q"])
    assert warned.exit_code == 0, warned.output
    assert "no media files found" in warned.stderr  # WARNING, kept


def test_verbosity_flags_resolve_in_the_documented_order() -> None:
    """None rather than a default level, so a config-file level survives a plain run."""
    assert cli._resolve_level(None, 0, False) is None
    assert cli._resolve_level(None, 1, False) is logging_setup.LogLevel.DEBUG
    assert cli._resolve_level(None, 0, True) is logging_setup.LogLevel.WARNING
    assert cli._resolve_level(logging_setup.LogLevel.ERROR, 2, True) is logging_setup.LogLevel.ERROR
