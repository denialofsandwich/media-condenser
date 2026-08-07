"""Verification-logic tests."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import fixtures

from media_condenser import config, discovery, planner, probe, verify


def action(kind: planner.Kind, *, will_process: bool = True) -> planner.Action:
    """A real :class:`Action`, rather than a stand-in with the same attributes.

    Using the actual type keeps the test honest: if ``will_process`` ever stops being
    derived from the verb, or the verbs are renamed, this breaks instead of quietly
    testing a shape that no longer matches production.
    """
    candidate = discovery.Candidate(path=Path("fixture.jpg"), kind=probe.JPEG, rules=config.Rules(), size=1024)
    verb = planner.Verb.REBUILD_MOTION_PHOTO if will_process else planner.Verb.SKIP
    return planner.Action(candidate=candidate, kind=kind, verb=verb)


def test_gps_rounding_is_not_reported_as_a_loss() -> None:
    """A round-trip re-renders coordinates at different precision.

    "37.5021" vs "37.50209" is about a metre apart -- a formatting artefact, not
    lost data. A string comparison flags it and buries the real failures.
    """
    assert verify._values_match("GPSPosition", "37.5021 127.034", "37.50209 127.03400")


def test_genuinely_different_coordinates_still_fail() -> None:
    assert not verify._values_match("GPSPosition", "37.5021 127.034", "48.1372 11.5756")


def test_partially_lost_coordinates_fail() -> None:
    assert not verify._values_match("GPSPosition", "37.5021 127.034", "37.5021")


def test_non_gps_tags_are_compared_strictly() -> None:
    """A one-second timestamp drift is a real problem, not a rounding artefact."""
    assert not verify._values_match("CreateDate", "2020:01:04 03:00:00", "2020:01:04 03:00:01")
    assert verify._values_match("Model", "MCT TestCam", "MCT TestCam")


def test_caveat_is_raised_only_for_motion_photos() -> None:
    """The tool must never let green ticks imply HDR/motion were confirmed."""
    assert verify.caveat([action(planner.Kind.MOTION_PHOTO)]) == verify.MOTION_PHOTO_CAVEAT
    assert verify.caveat([action(planner.Kind.VIDEO), action(planner.Kind.IMAGE)]) is None
    # A skipped motion photo was not rebuilt, so there is nothing to caveat.
    assert verify.caveat([action(planner.Kind.MOTION_PHOTO, will_process=False)]) is None


# ---------------------------------------------------------------------------
# Verifying in place, where the output path *is* the source path
# ---------------------------------------------------------------------------


def strip_all_metadata(path: Path) -> None:
    """Stand in for a processing step that silently lost the metadata, in place."""
    subprocess.run(
        ["exiftool", "-all=", "-overwrite_original", str(path)],
        check=True,
        capture_output=True,
    )


def test_metadata_loss_under_replace_is_detected(tmp_path) -> None:
    """The one strategy where a missed loss is unrecoverable must be the strictest.

    Under ``replace`` the output path is the source path, and both the "before" and
    "after" reads previously came from the same per-path probe cache -- so every tag
    was compared against itself and no loss could ever be reported, on the one
    strategy where the original is already gone. The pre-run snapshot plus a forced
    re-read of the output is what makes this check mean anything.
    """
    cfg = config.GlobalConfig(strategy=config.Strategy.REPLACE)
    prober = probe.Prober(cfg.tools)
    target = tmp_path / fixtures.IMAGE_LANDSCAPE
    shutil.copy2(fixtures.IMAGES / fixtures.IMAGE_LANDSCAPE, target)

    # Exactly what the CLI captures before executing, cache included.
    before = prober.exif(target)
    assert before.get("Model"), "fixture is supposed to carry camera identity"

    strip_all_metadata(target)

    verification = verify.verify_output(target, target, planner.Kind.IMAGE, prober, cfg, before=before)
    assert not verification.ok
    assert any("Model was lost" in problem for problem in verification.problems), verification.problems


def test_replace_without_a_snapshot_reports_it_instead_of_passing(tmp_path) -> None:
    """No snapshot and no original left is an unanswerable question, not a pass."""
    cfg = config.GlobalConfig(strategy=config.Strategy.REPLACE)
    prober = probe.Prober(cfg.tools)
    target = tmp_path / fixtures.IMAGE_LANDSCAPE
    shutil.copy2(fixtures.IMAGES / fixtures.IMAGE_LANDSCAPE, target)

    verification = verify.verify_output(target, target, planner.Kind.IMAGE, prober, cfg)
    assert not verification.ok
    assert any("not comparable" in problem for problem in verification.problems), verification.problems
    assert not any("preserved" in check for check in verification.checks)


def test_output_dimensions_are_re_read_rather_than_recalled(tmp_path) -> None:
    """The image check must describe the finished file, not the planned one."""
    cfg = config.GlobalConfig(strategy=config.Strategy.REPLACE)
    prober = probe.Prober(cfg.tools)
    target = tmp_path / fixtures.IMAGE_LANDSCAPE
    shutil.copy2(fixtures.IMAGES / fixtures.IMAGE_LANDSCAPE, target)

    original = prober.image_info(target)
    before = prober.exif(target)
    subprocess.run(
        ["convert", str(target), "-resize", "64x64", str(target)],
        check=True,
        capture_output=True,
    )

    verification = verify.verify_output(target, target, planner.Kind.IMAGE, prober, cfg, before=before)
    assert any("64x" in check for check in verification.checks), verification.checks
    assert not any(f"{original.width}x{original.height}" in check for check in verification.checks)


def test_untouched_original_is_still_compared_without_a_snapshot(tmp_path) -> None:
    """Where the original survives, omitting the snapshot stays a valid call."""
    cfg = config.GlobalConfig()
    prober = probe.Prober(cfg.tools)
    source = fixtures.IMAGES / fixtures.IMAGE_LANDSCAPE
    output = tmp_path / fixtures.IMAGE_LANDSCAPE
    shutil.copy2(source, output)

    verification = verify.verify_output(source, output, planner.Kind.IMAGE, prober, cfg)
    assert verification.ok, verification.problems
    assert any("Model preserved" in check for check in verification.checks), verification.checks

    strip_all_metadata(output)
    prober.forget(output)
    assert not verify.verify_output(source, output, planner.Kind.IMAGE, prober, cfg).ok
