"""Command-line behaviour: the signals a script or a person actually acts on.

Exit codes and up-front refusals, rather than the classification and encoding logic
covered elsewhere. Anything here that runs a real encode is marked ``slow``.
"""

from __future__ import annotations

import json
import re
import shutil

import fixtures
import pytest
from typer import testing

from media_condenser import cli, logging_setup, storage, verify


def library(tmp_path, *names: str):
    workdir = tmp_path / "lib"
    workdir.mkdir()
    for name in names or (fixtures.IMAGE_LANDSCAPE,):
        source = fixtures.IMAGES / name if (fixtures.IMAGES / name).exists() else fixtures.VIDEOS / name
        shutil.copy2(source, workdir / name)
    return workdir


# ---------------------------------------------------------------------------
# Refusals, before anything is written
# ---------------------------------------------------------------------------


def test_output_dir_with_replace_is_refused(tmp_path) -> None:
    """Silently discarding the flag meant overwriting the only copies instead.

    ``plan_destination`` short-circuits on ``replace`` before it looks at
    ``output_dir``, so the mirror tree was never written and the originals were
    rewritten in place -- while the only warning printed talked about overwriting
    without mentioning that ``-o`` had been dropped.
    """
    workdir = library(tmp_path)
    before = (workdir / fixtures.IMAGE_LANDSCAPE).read_bytes()
    out = tmp_path / "out"

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "--strategy", "replace", "-o", str(out)])

    assert result.exit_code == 2, result.output
    assert "output_dir cannot be combined" in result.output
    assert (workdir / fixtures.IMAGE_LANDSCAPE).read_bytes() == before
    assert not out.exists()


def test_copy_without_an_output_dir_is_refused(tmp_path) -> None:
    """The default invocation used to write ``photo_compressed.jpg`` siblings.

    ``mcon ~/Photos`` -- no flags, the most obvious thing to type -- interleaved a
    second copy of an entire library among the originals, under a name no gallery
    or sync tool recognises. Nothing about the command said that would happen.
    """
    workdir = library(tmp_path)
    before = sorted(p.name for p in workdir.iterdir())

    result = testing.CliRunner().invoke(cli.app, [str(workdir)])

    assert result.exit_code == 2, result.output
    # Both ways out, because the refusal is useless if it only says "no".
    assert "--output-dir" in result.output
    assert "--strategy replace" in result.output
    assert sorted(p.name for p in workdir.iterdir()) == before


def test_copy_without_an_output_dir_is_refused_before_the_dry_run_plan(tmp_path) -> None:
    """--dry-run is refused too, rather than previewing unwritable paths.

    A plan is a promise about where files would land, and there is no answer to
    that here. Printing one anyway would make the preview disagree with every
    possible run.
    """
    workdir = library(tmp_path)

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "--dry-run"])

    assert result.exit_code == 2, result.output
    assert "Planned actions" not in result.output


def test_copy_with_an_output_dir_in_the_config_file_is_accepted(tmp_path) -> None:
    """The requirement is on the resolved config, not on the flag.

    Setting ``output_dir`` in the config file has to satisfy it, or the check
    would be refusing runs that are perfectly well specified.
    """
    workdir = library(tmp_path)
    out = tmp_path / "out"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"output_dir: {out}\n")

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "-c", str(config_file), "--dry-run"])

    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Exit code
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_verification_failure_sets_a_nonzero_exit_code(tmp_path, monkeypatch) -> None:
    """`mcon --verify ... && rm -rf originals` has to be safe to write.

    The run itself succeeds here -- only verification fails -- so this pins the one
    signal a script has to go on.
    """
    workdir = library(tmp_path)
    real_verify_output = verify.verify_output

    def losing_verify(*args, **kwargs):
        verification = real_verify_output(*args, **kwargs)
        verification.failed("simulated verification failure")
        return verification

    monkeypatch.setattr(cli.verify, "verify_output", losing_verify)
    result = testing.CliRunner().invoke(cli.app, [str(workdir), "--strategy", "replace", "--verify", "--quiet"])

    assert result.exit_code == 1, result.output
    assert "0/1 outputs verified clean" in result.output


@pytest.mark.slow
def test_clean_verification_still_exits_zero(tmp_path) -> None:
    workdir = library(tmp_path)
    result = testing.CliRunner().invoke(cli.app, [str(workdir), "--strategy", "replace", "--verify", "--quiet"])
    assert result.exit_code == 0, result.output
    assert "1/1 outputs verified clean" in result.output


# ---------------------------------------------------------------------------
# Leftovers from an interrupted run
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_leftover_temp_files_are_cleared_before_the_run(tmp_path) -> None:
    """They are this tool's own half-written output, not library media.

    ``cleanup_stale`` existed and was unit-tested but had no caller, so leftovers
    accumulated -- and being real JPEG bytes, the next run ingested them: resized and
    committed in place under the temp name, or copied into the mirror tree as junk.
    """
    workdir = library(tmp_path)
    leftover = workdir / f"{storage.TMP_PREFIX}31415_{fixtures.IMAGE_LANDSCAPE}"
    shutil.copy2(fixtures.IMAGES / fixtures.IMAGE_LANDSCAPE, leftover)

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "--strategy", "replace", "--quiet"])

    assert result.exit_code == 0, result.output
    assert not leftover.exists()
    assert "1 leftover temp file" in result.output
    assert sorted(p.name for p in workdir.iterdir()) == [fixtures.IMAGE_LANDSCAPE]


# ---------------------------------------------------------------------------
# Shell completion
# ---------------------------------------------------------------------------


def complete(monkeypatch, line: str, word: str = "") -> str:
    """Ask the CLI for zsh completions of `line`, as the shell hook would.

    ``prog_name`` matters: click derives the completion environment variable from it,
    so without it the runner looks for ``_MAIN_COMPLETE`` (after the function name)
    and the request is simply parsed as a normal invocation.
    """
    monkeypatch.setenv("_MCON_COMPLETE", "complete_zsh")
    monkeypatch.setenv("_TYPER_COMPLETE_ARGS", line)
    monkeypatch.setenv("_TYPER_COMPLETE_WORD_TO_COMPLETE", word)
    return testing.CliRunner().invoke(cli.app, [], prog_name="mcon").output


def test_completion_is_wired_up(monkeypatch) -> None:
    """`add_completion=False` does more than hide the install flag.

    It also stops typer calling `completion_init()`, so the shell classes are never
    registered and every completion request answers "Shell zsh not supported" --
    which is what this tool did before. Hence a test on real completion output
    rather than on the flag.
    """
    output = complete(monkeypatch, "mcon --", "--")

    assert "not supported" not in output
    assert "--output-dir" in output
    assert "--report-json" in output
    assert "--install-completion" in output


def test_choice_options_complete_their_values(monkeypatch) -> None:
    assert "replace" in complete(monkeypatch, "mcon --strategy ")
    assert "json" in complete(monkeypatch, "mcon --log-format ")
    assert "CRITICAL" in complete(monkeypatch, "mcon --log-level ")


def test_path_options_defer_to_the_shells_own_file_completion(monkeypatch) -> None:
    """Better than listing paths ourselves: the shell already does it properly."""
    assert complete(monkeypatch, "mcon --config ").strip() == "_files"
    assert complete(monkeypatch, "mcon ").strip() == "_files"


def test_a_directory_is_refused_where_a_file_is_meant(tmp_path) -> None:
    """The same annotation that steers completion also rejects the wrong kind."""
    result = testing.CliRunner().invoke(cli.app, [str(tmp_path), "-c", str(tmp_path)])

    assert result.exit_code == 2, result.output
    assert "is a directory" in _unbox(result.output)


def _unbox(text: str) -> str:
    """Message text with rich's box drawing, ANSI color and wrapping taken back out.

    Rich puts errors in a panel, so a long path pushes the message onto a second
    line with a border in the middle of the sentence. Colour is stripped too: with
    it left in, GitHub Actions (which forces color on) leaves escape codes sitting
    where the border used to be, which breaks the substring match just as badly.
    """
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return " ".join(re.sub(r"[│╭╮╰╯─]", " ", text).split())


# ---------------------------------------------------------------------------
# The result stream: format, and where it lands in the output
# ---------------------------------------------------------------------------


def test_summary_json_puts_one_parseable_document_on_stdout(tmp_path) -> None:
    """For a pipeline that reads stdout instead of passing --report-json a path."""
    workdir = library(tmp_path)

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "-o", str(tmp_path / "out"), "-n", "--summary", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["totals"]["files"] == 1
    assert payload["files"][0]["action"] == "resize image"
    assert "Planned actions" not in result.stdout


def test_summary_none_writes_nothing_to_stdout(tmp_path) -> None:
    workdir = library(tmp_path)

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "-o", str(tmp_path / "out"), "-n", "--summary", "none"])

    assert result.exit_code == 0, result.output
    assert result.stdout == ""


def test_summary_json_is_emitted_even_when_nothing_was_found(tmp_path) -> None:
    """A batch consumer parsing stdout needs a document saying zero, not silence."""
    empty = tmp_path / "empty"
    empty.mkdir()

    result = testing.CliRunner().invoke(cli.app, [str(empty), "-o", str(tmp_path / "out"), "--summary", "json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["totals"]["succeeded"] == 0


def test_json_results_do_not_depend_on_the_terminal_width(tmp_path, monkeypatch) -> None:
    """Rendering JSON through rich would wrap or clip a long path into garbage."""
    monkeypatch.setattr(logging_setup.RESULT_CONSOLE, "_width", 40)
    workdir = tmp_path / ("d" * 90)
    workdir.mkdir()
    shutil.copy2(fixtures.IMAGES / fixtures.IMAGE_LANDSCAPE, workdir / fixtures.IMAGE_LANDSCAPE)

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "-o", str(tmp_path / "out"), "-n", "--summary", "json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["files"][0]["path"] == str(workdir / fixtures.IMAGE_LANDSCAPE)


@pytest.mark.slow
def test_the_summary_is_printed_after_the_last_verify_log_line(tmp_path) -> None:
    """--verify logs a line per file, and the summary used to be printed before them.

    On a real library that is hundreds of lines, so the table the run exists to
    produce had scrolled off the screen by the time it finished.
    """
    workdir = library(tmp_path)

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "--strategy", "replace", "--verify", "-v"])

    assert result.exit_code == 0, result.output
    combined = result.output
    summary_at = combined.index("Summary by media type")
    # "checks)" only ever appears in a per-file verification log line, and
    # "finished in" is the very last log line of a run.
    assert combined.index("checks)") < summary_at
    assert combined.index("finished in") < summary_at
    # And the tally belongs with the summary, not stranded among the log lines.
    assert summary_at < combined.index("outputs verified clean")


@pytest.mark.slow
def test_verification_results_reach_the_json_summary(tmp_path) -> None:
    workdir = library(tmp_path)

    result = testing.CliRunner().invoke(
        cli.app, [str(workdir), "--strategy", "replace", "--verify", "--summary", "json"]
    )

    assert result.exit_code == 0, result.output
    verify_section = json.loads(result.stdout)["verify"]
    assert verify_section == {"ran": True, "verified": 1, "problems": 0, "caveat": None}


# ---------------------------------------------------------------------------
# --verify gates the commit
# ---------------------------------------------------------------------------


def failing_verify(monkeypatch) -> None:
    """Make every verification reject its output, however good the output is."""
    real = verify.verify_output

    def losing(*args, **kwargs):
        verification = real(*args, **kwargs)
        verification.failed("simulated verification failure")
        return verification

    monkeypatch.setattr(verify, "verify_output", losing)


@pytest.mark.slow
def test_a_verification_failure_counts_as_a_failed_file(tmp_path, monkeypatch) -> None:
    """It used to count as a success.

    Verification ran after the whole run, so the file had already been committed and
    reported as succeeded; only the exit code and the tally disagreed.
    """
    workdir = library(tmp_path)
    out = tmp_path / "out"
    failing_verify(monkeypatch)

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "-o", str(out), "--verify", "--summary", "json"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["totals"]["failed"] == 1
    assert payload["totals"]["succeeded"] == 0
    assert payload["files"][0]["verify_problems"] == ["simulated verification failure"]


@pytest.mark.slow
def test_a_rejected_output_is_never_committed_under_copy(tmp_path, monkeypatch) -> None:
    """Nothing should be left in the output tree for a file that did not verify."""
    workdir = library(tmp_path)
    out = tmp_path / "out"
    failing_verify(monkeypatch)

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "-o", str(out), "--verify"])

    assert result.exit_code == 1, result.output
    assert list(out.rglob("*")) == [], "a rejected output was committed anyway"
    assert not list(workdir.glob(f"{storage.TMP_PREFIX}*")), "the temp file was left behind"


@pytest.mark.slow
def test_a_rejected_output_leaves_the_original_intact_under_replace(tmp_path, monkeypatch) -> None:
    """The case that made verify-after-commit unfixable.

    `replace` overwrites the original, so by the time a post-run check ran there was
    nothing left to roll back to -- a bad output was simply the only copy. Verifying
    the temp file before the rename is what makes the check able to refuse.
    """
    workdir = library(tmp_path)
    original = (workdir / fixtures.IMAGE_LANDSCAPE).read_bytes()
    failing_verify(monkeypatch)

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "--strategy", "replace", "--verify"])

    assert result.exit_code == 1, result.output
    assert (workdir / fixtures.IMAGE_LANDSCAPE).read_bytes() == original
    assert sorted(p.name for p in workdir.iterdir()) == [fixtures.IMAGE_LANDSCAPE]


@pytest.mark.slow
def test_no_verify_skips_the_check_entirely(tmp_path) -> None:
    workdir = library(tmp_path)
    out = tmp_path / "out"

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "-o", str(out), "--no-verify", "--summary", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["verify"] == {"ran": False, "verified": 0, "problems": 0}
    assert payload["files"][0]["verified"] is False


@pytest.mark.slow
def test_outputs_are_verified_without_being_asked(tmp_path) -> None:
    """Checking is the default: the point of the check is to be the thing that stops a
    bad output reaching the library, and an opt-in guard only protects whoever
    remembered to ask for it."""
    workdir = library(tmp_path)
    out = tmp_path / "out"

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "-o", str(out), "--summary", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["verify"] == {"ran": True, "verified": 1, "problems": 0, "caveat": None}
    assert payload["files"][0]["verified"] is True


@pytest.mark.slow
def test_a_bad_output_is_refused_by_default(tmp_path, monkeypatch) -> None:
    """The whole point of the default: no flag, and the original still survives."""
    workdir = library(tmp_path)
    original = (workdir / fixtures.IMAGE_LANDSCAPE).read_bytes()
    failing_verify(monkeypatch)

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "--strategy", "replace"])

    assert result.exit_code == 1, result.output
    assert (workdir / fixtures.IMAGE_LANDSCAPE).read_bytes() == original


# ---------------------------------------------------------------------------
# --dry-run against an already-populated output tree
# ---------------------------------------------------------------------------


def plan_of(workdir, *extra: str) -> dict:
    """The --dry-run plan as JSON."""
    result = testing.CliRunner().invoke(cli.app, [str(workdir), "-n", "--summary", "json", *extra])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


@pytest.mark.slow
def test_a_dry_run_predicts_the_skips_a_rerun_would_make(tmp_path) -> None:
    """It used to preview a full re-encode of a tree it would then skip entirely.

    A dry run is only worth having if it matches the run, and this is the case where
    it silently did not: `_already_produced` is consulted by the run but was not
    consulted by the plan.
    """
    workdir = library(tmp_path)
    out = tmp_path / "out"

    fresh = plan_of(workdir, "-o", str(out))
    assert fresh["totals"]["by_action"] == {"resize image": 1}
    assert fresh["files"][0]["already_present"] is False

    run = testing.CliRunner().invoke(cli.app, [str(workdir), "-o", str(out), "-q", "--summary", "json"])
    assert run.exit_code == 0, run.output
    assert json.loads(run.stdout)["totals"]["succeeded"] == 1

    rerun = plan_of(workdir, "-o", str(out))
    assert rerun["totals"]["by_action"] == {"skip": 1}
    assert rerun["files"][0]["already_present"] is True
    assert "already present in the output tree" in rerun["files"][0]["reason"]


@pytest.mark.slow
def test_a_changed_source_is_planned_as_work_again(tmp_path) -> None:
    """The guard keys on the carried-over mtime, so editing the source undoes it."""
    workdir = library(tmp_path)
    out = tmp_path / "out"
    testing.CliRunner().invoke(cli.app, [str(workdir), "-o", str(out), "-q", "--summary", "none"])
    assert plan_of(workdir, "-o", str(out))["totals"]["by_action"] == {"skip": 1}

    (workdir / fixtures.IMAGE_LANDSCAPE).touch()

    assert plan_of(workdir, "-o", str(out))["totals"]["by_action"] == {"resize image": 1}


@pytest.mark.slow
def test_the_plan_table_reports_the_skip_too(tmp_path) -> None:
    """Not only the JSON: the table is what a person reads before committing."""
    workdir = library(tmp_path)
    out = tmp_path / "out"
    testing.CliRunner().invoke(cli.app, [str(workdir), "-o", str(out), "-q", "--summary", "none"])

    result = testing.CliRunner().invoke(cli.app, [str(workdir), "-n", "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert "Totals: 1 skip" in result.stdout
    assert "already present" in _unbox(result.stdout)


def test_replace_never_reports_an_output_as_already_present(tmp_path) -> None:
    """There the output *is* the source, so the planner's own measurement decides."""
    workdir = library(tmp_path)

    plan = plan_of(workdir, "--strategy", "replace")

    assert plan["totals"]["by_action"] == {"resize image": 1}
    assert plan["files"][0]["already_present"] is False
