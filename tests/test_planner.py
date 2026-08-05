"""Classification tests: the skip-or-process gate that makes reruns idempotent."""

from __future__ import annotations

import shutil
from pathlib import Path

import fixtures
import pytest

from media_condenser import config, discovery, planner, probe, storage


@pytest.fixture(scope="module")
def actions() -> dict[str, object]:
    cfg = config.GlobalConfig()
    resolver = config.RulesResolver(cfg.rules, [fixtures.ROOT])
    prober = probe.Prober(cfg.tools)
    return {c.path.name: planner.plan(c, prober) for c in discovery.walk([fixtures.ROOT], resolver)}


def test_every_fixture_is_discovered(actions) -> None:
    assert len(actions) == fixtures.TOTAL_FILES


def test_nothing_fails_to_classify(actions) -> None:
    """Including the deliberately broken fixtures -- none may hard-error."""
    failed = {name: a.reason for name, a in actions.items() if a.verb is planner.Verb.FAIL}
    assert not failed, f"planning failures: {failed}"


# -- content-based classification -------------------------------------


def test_extension_is_never_trusted() -> None:
    """A ``.png`` fixture that is really a JPEG must be sniffed as one."""
    assert probe.sniff_kind(fixtures.IMAGES / fixtures.SCREENSHOT_MISLABELED) == probe.JPEG
    assert probe.sniff_kind(fixtures.IMAGES / fixtures.SCREENSHOT_PNG) == probe.PNG
    assert probe.sniff_kind(fixtures.VIDEOS / fixtures.VIDEO_ROTATED_PLUS90) == probe.VIDEO


def iso_file(path: Path, brand: bytes) -> Path:
    """A minimal ISO base media header carrying ``brand``."""
    path.write_bytes(b"\x00\x00\x00\x18ftyp" + brand + b"\x00\x00\x02\x00" + brand)
    return path


@pytest.mark.parametrize(
    ("brand", "expected"),
    [
        (b"isom", probe.VIDEO),
        (b"mp42", probe.VIDEO),
        (b"qt  ", probe.VIDEO),
        (b"heic", probe.HEIF),
        # Stills that are *not* video. Classifying these as video hands them to the
        # video planner, which transcodes them to H.265 under their original name.
        (b"avif", probe.ISO_OTHER),
        (b"crx ", probe.ISO_OTHER),
        # A brand nobody taught the tool about: reported, never guessed at.
        (b"zzzz", probe.ISO_OTHER),
    ],
)
def test_iso_brands_decide_whether_a_file_is_video(tmp_path: Path, brand: bytes, expected: str) -> None:
    assert probe.sniff_kind(iso_file(tmp_path / "sample.bin", brand)) == expected


def test_unsupported_media_is_reported_rather_than_dropped(tmp_path: Path) -> None:
    """A recognised type this tool cannot process must still reach the report.

    Dropping it during the walk makes it invisible: no record, no row in the summary,
    and -- the part that loses data -- no copy into an ``--output-dir`` mirror. "42
    skipped, 0 failed" over a tree that silently omitted every HEIC tells the user the
    mirror is complete when it is not.
    """
    iso_file(tmp_path / "IMG_0001.heic", b"heic")
    resolver = config.RulesResolver(config.Rules(), [tmp_path])
    candidates = discovery.walk([tmp_path], resolver)
    assert [c.path.name for c in candidates] == ["IMG_0001.heic"]

    action = planner.plan(candidates[0], probe.Prober(config.GlobalConfig().tools))
    assert action.kind is planner.Kind.OTHER
    assert action.verb is planner.Verb.SKIP
    assert action.reason == "unsupported type (heif)"


def test_files_that_are_not_media_at_all_stay_out_of_the_walk(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not media")
    assert probe.sniff_kind(tmp_path / "notes.txt") == probe.UNKNOWN
    assert discovery.walk([tmp_path], config.RulesResolver(config.Rules(), [tmp_path])) == []


def test_our_own_temp_files_are_never_ingested(tmp_path: Path) -> None:
    """A leftover from an interrupted run is not library media.

    It carries real JPEG magic, so content sniffing alone accepts it -- and then a
    ``replace`` run resizes it and commits it in place, permanently installing a
    ``.mcon_tmp_*`` file into the user's library.
    """
    import shutil

    shutil.copy2(
        fixtures.IMAGES / fixtures.IMAGE_LANDSCAPE, tmp_path / f"{storage.TMP_PREFIX}31415_{fixtures.IMAGE_LANDSCAPE}"
    )
    shutil.copy2(fixtures.IMAGES / fixtures.IMAGE_LANDSCAPE, tmp_path / fixtures.IMAGE_LANDSCAPE)

    found = [c.path.name for c in discovery.walk([tmp_path], config.RulesResolver(config.Rules(), [tmp_path]))]
    assert found == [fixtures.IMAGE_LANDSCAPE]


# -- the zero-op set ---------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [fixtures.VIDEO_ALREADY_SMALL, *fixtures.SKIPPED_IMAGES],
)
def test_already_conforming_files_are_skipped(actions, name: str) -> None:
    """These must never be resized, re-encoded or upscaled."""
    assert actions[name].verb is planner.Verb.SKIP


def test_screenshots_are_skipped_by_rule_not_by_size(actions) -> None:
    """Both screenshots exceed the size target, so only the exclusions can skip them.

    Worth asserting explicitly: the equivalent real fixture was *under* the target,
    so the PNG rule appeared to work while actually being untested.
    """
    for name in (fixtures.SCREENSHOT_PNG, fixtures.SCREENSHOT_MISLABELED):
        prober = probe.Prober(config.GlobalConfig().tools)
        assert prober.image_info(fixtures.IMAGES / name).long_edge > 2048
        assert actions[name].verb is planner.Verb.SKIP


def test_small_video_is_not_upscaled(actions) -> None:
    action = actions[fixtures.VIDEO_ALREADY_SMALL]
    assert action.verb is planner.Verb.SKIP
    assert action.video_info.short_edge == 296


# -- rotation awareness ------------------------------------------------


@pytest.mark.parametrize("name", [fixtures.VIDEO_ROTATED_MINUS90, fixtures.VIDEO_ROTATED_PLUS90])
def test_rotated_videos_report_portrait_display_size(actions, name: str) -> None:
    """Stored landscape but displayed portrait via the container matrix.

    Every size decision has to use the decoded orientation; reading the stored
    height would scale the wrong axis. Both rotation signs are covered.
    """
    info = actions[name].video_info
    assert (info.width, info.height) == fixtures.VIDEO_STORED_SIZE
    assert info.display_size == fixtures.VIDEO_DISPLAY_ROTATED
    assert actions[name].target_size == fixtures.VIDEO_TARGET_PORTRAIT


def test_square_video_stays_square(actions) -> None:
    """The filter's false branch must not distort a 1:1 frame."""
    action = actions[fixtures.VIDEO_SQUARE]
    assert action.video_info.display_size == fixtures.VIDEO_SQUARE_SIZE
    assert action.target_size == fixtures.VIDEO_TARGET_SQUARE


def test_landscape_video_targets_720p(actions) -> None:
    assert actions[fixtures.VIDEO_NO_AUDIO].target_size == fixtures.VIDEO_TARGET_LANDSCAPE


def test_audioless_video_is_detected(actions) -> None:
    """Mapping a nonexistent audio stream would be a hard error."""
    assert actions[fixtures.VIDEO_NO_AUDIO].video_info.has_audio is False


def test_other_videos_do_have_audio(actions) -> None:
    """Keeps the audio-existence check honest in both directions."""
    assert actions[fixtures.VIDEO_ROTATED_MINUS90].video_info.has_audio is True


def test_tilde_suffixed_file_is_classified_by_content(actions) -> None:
    """A '~2' duplicate-looking name that is genuinely full resolution."""
    action = actions[fixtures.VIDEO_TILDE_H264]
    assert action.verb is planner.Verb.TRANSCODE_VIDEO
    assert action.video_info.display_size == fixtures.VIDEO_STORED_SIZE


# -- motion photos -----------------------------------------------------


def test_all_motion_photos_are_recognized(actions) -> None:
    motion = {n for n, a in actions.items() if a.kind is planner.Kind.MOTION_PHOTO}
    assert motion == set(fixtures.ALL_MOTION_PHOTOS)


def test_bag_form_motion_photo_is_not_downgraded(actions) -> None:
    """The misdiagnosed serialization keeps its gain map and stays a motion photo.

    If the parser regressed to reading only the flattened ``DirectoryItem*`` tags,
    this file would fall back to the plain-image path and lose its HDR.
    """
    action = actions[fixtures.MP_BAG_FORM]
    assert action.kind is planner.Kind.MOTION_PHOTO
    assert action.verb is planner.Verb.REBUILD_MOTION_PHOTO
    assert action.layout.gain_map is not None


def test_corrupt_video_length_is_reported_as_a_note_not_a_failure(actions) -> None:
    action = actions[fixtures.MP_IMPOSSIBLE_VIDEO_LEN]
    assert action.verb is planner.Verb.REBUILD_MOTION_PHOTO
    assert action.layout.video is None
    assert action.layout.gain_map is not None
    assert any("exceeds file size" in note for note in action.notes)


def test_motion_photo_skip_gate_uses_the_primary_resolution(actions) -> None:
    action = actions[fixtures.MP_FULL_COMBO]
    assert (action.image_info.width, action.image_info.height) == fixtures.IMAGE_SIZE_PORTRAIT
    assert action.target_size == fixtures.IMAGE_TARGET_PORTRAIT


def stale_motion_photo(source: Path, target: Path) -> Path:
    """A motion photo that declares a video it cannot deliver.

    The XMP still says ``MotionPhoto`` with a real length, but the appended clip's
    ``ftyp`` box is overwritten, so ``resolve_layout`` refuses to confirm it and the
    file resolves to no usable attachments. Real files reach this state when an
    editor rewrites the primary and leaves the directory behind.
    """
    data = source.read_bytes()
    layout = fixtures.layout_of(source)
    assert layout.video is not None, "fixture is supposed to declare an embedded video"
    patched = bytearray(data)
    patched[layout.video.offset + 4 : layout.video.offset + 8] = b"XXXX"
    target.write_bytes(bytes(patched))
    return target


def plan_one(path: Path, rules: config.Rules) -> planner.Action:
    resolver = config.RulesResolver(rules, [path.parent])
    candidates = discovery.walk([path], resolver)
    assert len(candidates) == 1
    return planner.plan(candidates[0], probe.Prober(config.GlobalConfig().tools))


def test_skipping_a_motion_photo_does_not_read_the_container(tmp_path: Path, monkeypatch) -> None:
    """The resolution gate has to come before the whole-file read, not after it.

    Resolving the container costs a full read -- several megabytes on a real photo --
    and a library that is mostly motion photos already under ``max_edge`` paid it per
    file to reach a decision the cached dimensions had already made. A skip has no use
    for the result: only ``Verb.REBUILD_MOTION_PHOTO`` reads ``Action.layout``.
    """
    target = tmp_path / fixtures.MP_FULL_COMBO
    shutil.copy2(fixtures.MOTION / fixtures.MP_FULL_COMBO, target)

    reads: list[str] = []
    unpatched = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (reads.append(self.name), unpatched(self))[1])

    skipped = plan_one(target, config.Rules.model_validate({"images": {"max_edge": 4096}}))
    assert skipped.kind is planner.Kind.MOTION_PHOTO
    assert skipped.verb is planner.Verb.SKIP
    assert reads == []

    # The same file below the gate still resolves its container, so this pins an
    # ordering change rather than the read having been dropped altogether.
    rebuilt = plan_one(target, config.Rules.model_validate({"images": {"max_edge": 1024}}))
    assert rebuilt.verb is planner.Verb.REBUILD_MOTION_PHOTO
    assert rebuilt.layout is not None
    assert reads == [target.name]


def test_stale_motion_photo_clears_its_container_declaration(tmp_path: Path) -> None:
    """Falling back to a plain resize must not carry the declaration along.

    The resize path restores metadata with ``-tagsFromFile <original> -all:all``,
    which copies the original ``Container:Directory`` -- non-zero component lengths
    included -- onto an output holding nothing but the primary image. A reader
    trusting that declaration slices past the primary and hands back garbage as a
    video, which is exactly the hazard ``container.assemble`` was written to avoid.
    """
    target = stale_motion_photo(fixtures.MOTION / fixtures.MP_VIDEO_ONLY_LANDSCAPE, tmp_path / "stale.MP.jpg")
    action = plan_one(target, config.Rules())

    assert action.kind is planner.Kind.IMAGE
    assert action.verb is planner.Verb.RESIZE_IMAGE
    assert action.clear_motion_container is True
    assert any("no usable attached components" in note for note in action.notes)


def test_a_skipped_motion_photo_keeps_its_declaration(tmp_path: Path) -> None:
    """Nothing is rewritten, so nothing it declares has become untrue."""
    target = stale_motion_photo(fixtures.MOTION / fixtures.MP_VIDEO_ONLY_LANDSCAPE, tmp_path / "stale.MP.jpg")
    action = plan_one(target, config.Rules.model_validate({"images": {"max_edge": 4096}}))

    assert action.verb is planner.Verb.SKIP
    assert action.clear_motion_container is False


def test_disabling_motion_photos_still_clears_the_declaration(tmp_path: Path) -> None:
    """Disabling the feature means "resize the primary", not "lie about the rest".

    The resize drops the appended components either way, so the output must not go
    out declaring them -- and the loss is worth a note rather than silence.
    """
    target = tmp_path / fixtures.MP_FULL_COMBO
    shutil.copy2(fixtures.MOTION / fixtures.MP_FULL_COMBO, target)
    action = plan_one(target, config.Rules.model_validate({"motion_photos": {"enabled": False}}))

    assert action.kind is planner.Kind.IMAGE
    assert action.verb is planner.Verb.RESIZE_IMAGE
    assert action.clear_motion_container is True
    assert any("motion photo processing disabled" in note for note in action.notes)


# -- configurability ---------------------------------------------------


def test_disabling_a_media_type_skips_it() -> None:
    cfg = config.GlobalConfig()
    rules = config.Rules.model_validate({"videos": {"enabled": False}})
    resolver = config.RulesResolver(rules, [fixtures.VIDEOS])
    prober = probe.Prober(cfg.tools)
    actions = [planner.plan(c, prober) for c in discovery.walk([fixtures.VIDEOS], resolver)]
    assert actions
    assert all(a.verb is planner.Verb.SKIP for a in actions)


def test_png_exclusion_can_be_turned_off() -> None:
    """The screenshot skip is a default, not a hard rule."""
    rules = config.Rules.model_validate({"images": {"skip_png": False, "skip_name_patterns": []}})
    resolver = config.RulesResolver(rules, [fixtures.IMAGES])
    prober = probe.Prober(config.GlobalConfig().tools)
    actions = {c.path.name: planner.plan(c, prober) for c in discovery.walk([fixtures.IMAGES], resolver)}
    # Both screenshots exceed the long edge, so with exclusions off they now resize.
    assert actions[fixtures.SCREENSHOT_MISLABELED].verb is planner.Verb.RESIZE_IMAGE
    assert actions[fixtures.SCREENSHOT_PNG].verb is planner.Verb.RESIZE_IMAGE
