"""End-to-end tests over the fixture library.

These run real encodes, so they are the slowest tests here -- but they are the only
ones that can prove the properties that actually matter: that reruns are zero-op, that
conforming files come out byte-identical, and that every embedded component still
decodes after a container rebuild.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import fixtures
import pytest

from media_condenser import config, container, discovery, pipeline, planner, probe, report, verify
from media_condenser.handlers import image

pytestmark = pytest.mark.slow


def run_tool(paths: list[Path], cfg: config.GlobalConfig, scan_root: Path | None = None) -> report.Reporter:
    resolver = config.RulesResolver(cfg.rules, paths)
    prober = probe.Prober(cfg.tools)
    reporter = report.Reporter()
    actions = [planner.plan(candidate, prober) for candidate in discovery.walk(paths, resolver)]
    asyncio.run(pipeline.execute_all(actions, cfg, prober, reporter, scan_root=scan_root))
    return reporter


def by_name(reporter: report.Reporter) -> dict[str, report.Record]:
    return {record.path.name: record for record in reporter.records}


def probe_video(path: Path) -> dict[str, str]:
    # fmt: off
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name,codec_tag_string",
            "-of", "default=nw=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    # fmt: on
    return dict(line.split("=", 1) for line in out.strip().splitlines())


def exif_of(path: Path, *tags: str) -> str:
    return subprocess.run(
        ["exiftool", "-s3", *(f"-{tag}" for tag in tags), str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def image_run(tmp_path_factory) -> tuple[Path, report.Reporter]:
    output = tmp_path_factory.mktemp("images_out")
    cfg = config.GlobalConfig(output_dir=output, jobs=4)
    return output, run_tool([fixtures.IMAGES], cfg, scan_root=fixtures.IMAGES)


def test_conforming_images_are_never_re_encoded(image_run) -> None:
    """A skip must not touch the pixels.

    In a mirror tree the file is still copied across so the tree is complete, but it
    must arrive byte-for-byte identical -- never round-tripped through an encoder.
    """
    output, reporter = image_run
    records = by_name(reporter)
    for name in fixtures.SKIPPED_IMAGES:
        assert records[name].outcome is report.Outcome.SKIPPED
        assert (output / name).read_bytes() == (fixtures.IMAGES / name).read_bytes()


def test_resized_images_keep_their_metadata(image_run) -> None:
    output, _ = image_run
    for name in fixtures.RESIZED_IMAGES:
        assert (output / name).exists()
        before = exif_of(fixtures.IMAGES / name, "CreateDate", "Model", "Make")
        after = exif_of(output / name, "CreateDate", "Model", "Make")
        assert before and before == after, f"{name} lost metadata"


def test_resized_images_respect_the_long_edge(image_run) -> None:
    output, _ = image_run
    prober = probe.Prober(config.GlobalConfig().tools)
    expected = {
        fixtures.IMAGE_LANDSCAPE: fixtures.IMAGE_TARGET_LANDSCAPE,
        fixtures.IMAGE_PORTRAIT: fixtures.IMAGE_TARGET_PORTRAIT,
    }
    for name, (width, height) in expected.items():
        info = prober.image_info(output / name)
        assert (info.width, info.height) == (width, height)


def test_images_actually_get_smaller(image_run) -> None:
    _, reporter = image_run
    processed = [r for r in reporter.records if r.outcome is report.Outcome.SUCCEEDED]
    assert processed
    for record in processed:
        assert record.new_size < record.original_size


# ---------------------------------------------------------------------------
# Idempotency -- the property that makes reruns safe
# ---------------------------------------------------------------------------


def test_rerun_over_output_is_a_complete_no_op(tmp_path) -> None:
    """Processing the tool's own output must do nothing and change nothing.

    This is the guarantee that makes it safe to point the tool at a library
    repeatedly, and it only holds because the skip decision is made from measured
    properties before any work happens.
    """
    workdir = tmp_path / "lib"
    workdir.mkdir()
    for name in (fixtures.IMAGE_LANDSCAPE, fixtures.IMAGE_ALREADY_SMALL):
        shutil.copy2(fixtures.IMAGES / name, workdir / name)

    cfg = config.GlobalConfig(strategy=config.Strategy.REPLACE, jobs=2)
    run_tool([workdir], cfg, scan_root=workdir)

    fingerprint = {p.name: p.read_bytes() for p in sorted(workdir.iterdir())}

    second = run_tool([workdir], cfg, scan_root=workdir)
    assert all(r.outcome is report.Outcome.SKIPPED for r in second.records), (
        f"second run did work: {[(r.path.name, str(r.outcome)) for r in second.records]}"
    )
    assert {p.name: p.read_bytes() for p in sorted(workdir.iterdir())} == fingerprint


def test_rerun_over_a_populated_mirror_re_encodes_nothing(tmp_path) -> None:
    """The zero-op guarantee has to hold for mirror trees, not just in-place runs.

    A file that needs work is planned from the *source's* dimensions, which a previous
    run did not change -- so without a destination-side check every rerun transcodes
    the entire library again and overwrites outputs that were already correct. On a
    real library that is hours of x265 work to produce bytes that already existed.
    """
    lib = tmp_path / "lib"
    (lib / "2026").mkdir(parents=True)
    for name in (fixtures.IMAGE_LANDSCAPE, fixtures.IMAGE_ALREADY_SMALL):
        shutil.copy2(fixtures.IMAGES / name, lib / "2026" / name)
    shutil.copy2(fixtures.VIDEOS / fixtures.VIDEO_NO_AUDIO, lib / "2026" / fixtures.VIDEO_NO_AUDIO)

    output = tmp_path / "out"
    cfg = config.GlobalConfig(output_dir=output, jobs=4)

    first = run_tool([lib], cfg, scan_root=lib)
    assert any(r.outcome is report.Outcome.SUCCEEDED for r in first.records)
    fingerprint = {p.name: p.read_bytes() for p in sorted((output / "2026").iterdir())}

    second = run_tool([lib], cfg, scan_root=lib)
    assert all(r.outcome is report.Outcome.SKIPPED for r in second.records), (
        f"second run did work: {[(r.path.name, str(r.outcome), r.detail) for r in second.records]}"
    )
    assert {p.name: p.read_bytes() for p in sorted((output / "2026").iterdir())} == fingerprint


def test_a_tightened_rule_still_re_encodes_an_existing_output(tmp_path) -> None:
    """Keeps the rerun check from becoming "an output exists, therefore done"."""
    lib = tmp_path / "lib"
    lib.mkdir()
    shutil.copy2(fixtures.IMAGES / fixtures.IMAGE_LANDSCAPE, lib / fixtures.IMAGE_LANDSCAPE)
    output = tmp_path / "out"

    run_tool([lib], config.GlobalConfig(output_dir=output, jobs=2), scan_root=lib)
    prober = probe.Prober(config.GlobalConfig().tools)
    assert prober.image_info(output / fixtures.IMAGE_LANDSCAPE).long_edge == 2048

    tighter = config.GlobalConfig.model_validate(
        {"output_dir": str(output), "jobs": 2, "rules": {"images": {"max_edge": 800}}}
    )
    again = run_tool([lib], tighter, scan_root=lib)
    assert [r.outcome for r in again.records] == [report.Outcome.SUCCEEDED]
    assert probe.Prober(tighter.tools).image_info(output / fixtures.IMAGE_LANDSCAPE).long_edge == 800


def test_one_file_erroring_unexpectedly_does_not_end_the_run(tmp_path, monkeypatch) -> None:
    """A non-OSError escaping a worker used to abort the whole command.

    ``container.assemble`` raises ``ContainerRebuildError`` -- a ``RuntimeError`` --
    which the per-file handler did not catch and ``asyncio.gather`` did not contain. The
    first such photo in a 10,000-file run took the run down with a traceback, so the
    summary of everything that had already succeeded was never printed.
    """

    lib = tmp_path / "lib"
    lib.mkdir()
    for name in (fixtures.IMAGE_LANDSCAPE, fixtures.IMAGE_PORTRAIT):
        shutil.copy2(fixtures.IMAGES / name, lib / name)

    real_resize = image.resize

    async def exploding_resize(source, target, *args, **kwargs):
        if source.name == fixtures.IMAGE_LANDSCAPE:
            raise RuntimeError("simulated container rebuild failure")
        return await real_resize(source, target, *args, **kwargs)

    monkeypatch.setattr(image, "resize", exploding_resize)
    reporter = run_tool([lib], config.GlobalConfig(output_dir=tmp_path / "out", jobs=2), scan_root=lib)

    records = by_name(reporter)
    assert records[fixtures.IMAGE_LANDSCAPE].outcome is report.Outcome.FAILED
    assert "RuntimeError" in records[fixtures.IMAGE_LANDSCAPE].detail
    assert records[fixtures.IMAGE_PORTRAIT].outcome is report.Outcome.SUCCEEDED


def test_replace_strategy_leaves_no_temp_files(tmp_path) -> None:
    workdir = tmp_path / "lib"
    workdir.mkdir()
    shutil.copy2(fixtures.IMAGES / fixtures.IMAGE_PORTRAIT, workdir)
    run_tool([workdir], config.GlobalConfig(strategy=config.Strategy.REPLACE, jobs=2), scan_root=workdir)
    assert [p.name for p in workdir.iterdir()] == [fixtures.IMAGE_PORTRAIT]


def test_unsupported_media_reaches_the_mirror_tree(tmp_path) -> None:
    """A type this tool cannot process is still part of the library.

    HEIC files were recognised and then dropped from the walk, so they produced no
    record, appeared in no bucket of the summary, and -- the part that loses data --
    were never copied into the mirror. A user reading "0 failed", seeing a
    complete-looking tree and deleting the source loses every one of them.
    """
    lib = tmp_path / "lib"
    lib.mkdir()
    heic = lib / "IMG_0001.heic"
    heic.write_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x02\x00heicmif1")
    shutil.copy2(fixtures.IMAGES / fixtures.IMAGE_LANDSCAPE, lib / fixtures.IMAGE_LANDSCAPE)

    output = tmp_path / "out"
    reporter = run_tool([lib], config.GlobalConfig(output_dir=output, jobs=2), scan_root=lib)

    record = by_name(reporter)["IMG_0001.heic"]
    assert record.outcome is report.Outcome.SKIPPED
    assert "unsupported type (heif)" in record.detail
    assert (output / "IMG_0001.heic").read_bytes() == heic.read_bytes()


def test_skipped_file_under_replace_strategy_is_byte_identical(tmp_path) -> None:
    """The zero-op set must not be rewritten even when replacing in place."""
    workdir = tmp_path / "lib"
    workdir.mkdir()
    source = fixtures.IMAGES / fixtures.IMAGE_ALREADY_SMALL
    target = workdir / source.name
    shutil.copy2(source, target)
    run_tool([workdir], config.GlobalConfig(strategy=config.Strategy.REPLACE, jobs=2), scan_root=workdir)
    assert target.read_bytes() == source.read_bytes()


# ---------------------------------------------------------------------------
# Motion photos
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def motion_run(tmp_path_factory) -> tuple[Path, report.Reporter]:
    output = tmp_path_factory.mktemp("motion_out")
    cfg = config.GlobalConfig(output_dir=output, jobs=4)
    return output, run_tool([fixtures.MOTION], cfg, scan_root=fixtures.MOTION)


def test_no_motion_photo_hard_fails(motion_run) -> None:
    """Including every deliberately damaged container."""
    _, reporter = motion_run
    failed = {r.path.name: r.detail for r in reporter.records if r.outcome is report.Outcome.FAILED}
    assert not failed, f"motion photo failures: {failed}"


@pytest.mark.parametrize("name", fixtures.MOTION_PHOTOS_DOWNGRADED)
def test_edge_cases_are_reported_as_downgraded_not_failed(motion_run, name: str) -> None:
    _, reporter = motion_run
    record = by_name(reporter)[name]
    assert record.outcome is report.Outcome.DOWNGRADED
    assert record.notes


def test_intact_motion_photos_are_not_reported_as_downgraded(motion_run) -> None:
    """Keeps the downgrade bucket meaningful in both directions."""
    _, reporter = motion_run
    records = by_name(reporter)
    intact = set(fixtures.ALL_MOTION_PHOTOS) - set(fixtures.MOTION_PHOTOS_DOWNGRADED)
    for name in intact:
        assert records[name].outcome is report.Outcome.SUCCEEDED, records[name].detail


def test_every_rebuilt_component_still_decodes(motion_run) -> None:
    output, reporter = motion_run
    cfg = config.GlobalConfig()
    prober = probe.Prober(cfg.tools)

    checked = 0
    for record in reporter.records:
        if record.outcome not in (report.Outcome.SUCCEEDED, report.Outcome.DOWNGRADED):
            continue
        verification = verify.verify_output(
            fixtures.MOTION / record.path.name,
            output / record.path.name,
            planner.Kind.MOTION_PHOTO,
            prober,
            cfg,
        )
        assert verification.ok, f"{record.path.name}: {verification.problems}"
        checked += 1
    assert checked == len(fixtures.ALL_MOTION_PHOTOS)


@pytest.mark.parametrize("name", fixtures.MOTION_PHOTOS_WITH_GAIN_MAP)
def test_gain_maps_survive_the_rebuild_byte_identical(motion_run, name: str) -> None:
    """Every fixture that arrived with a gain map must leave with the same bytes.

    The ``-bearbeitet`` files are the point of this test: their container uses a
    serialization that both exiftool's flattened tags and its tag-copying get wrong,
    so a regression there loses HDR without any error being raised.

    Expected bytes are sliced from the input rather than hardcoded, so regenerating
    the library cannot silently invalidate the assertion.
    """
    output, _ = motion_run
    expected = fixtures.gain_map_bytes_of(fixtures.MOTION / name)
    assert expected is not None, "fixture is supposed to have a gain map"

    layout = fixtures.layout_of(output / name)
    assert layout.gain_map is not None, f"{name} lost its gain map"
    data = (output / name).read_bytes()
    assert data[layout.gain_map.offset : layout.gain_map.end] == expected


def test_gain_map_hdr_parameters_survive(motion_run) -> None:
    """A gain map without its ``hdrgm`` parameters cannot be applied by a renderer."""
    output, _ = motion_run
    layout = fixtures.layout_of(output / fixtures.MP_FULL_COMBO)
    data = (output / fixtures.MP_FULL_COMBO).read_bytes()
    blob = data[layout.gain_map.offset : layout.gain_map.end]

    extracted = output / "gm_extracted.jpg"
    extracted.write_bytes(blob)
    tags = subprocess.run(
        ["exiftool", "-s", "-XMP-hdrgm:all", str(extracted)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for tag in ("GainMapMin", "GainMapMax", "HDRCapacityMax"):
        assert tag in tags


def test_stale_video_is_not_declared_in_the_output(motion_run) -> None:
    """A dropped component must not be left declared in the directory.

    Otherwise a reader trusts the length and slices unrelated bytes as a video.
    """
    output, _ = motion_run
    data = (output / fixtures.MP_IMPOSSIBLE_VIDEO_LEN).read_bytes()
    semantics = {i.semantic for i in container.parse_container_items(container.extract_xmp_text(data))}
    assert "MotionPhoto" not in semantics
    assert "GainMap" in semantics


def test_a_plain_resize_does_not_inherit_a_container_declaration(tmp_path) -> None:
    """The same rule on the path that never reaches ``assemble``.

    Resized as a plain image, the primary comes out alone -- but the metadata restore
    is ``-tagsFromFile <original> -all:all``, which copies the original
    ``Container:Directory`` and ``MotionPhoto`` flag onto it. The output then declares
    a gain map and a multi-megabyte video inside a few hundred KB of JPEG, and a reader
    trusting that slices whatever follows and calls it a video. Only ``assemble()``
    rewrites the directory, and this path never gets there.
    """
    lib = tmp_path / "lib"
    lib.mkdir()
    shutil.copy2(fixtures.MOTION / fixtures.MP_FULL_COMBO, lib / fixtures.MP_FULL_COMBO)
    output = tmp_path / "out"
    cfg = config.GlobalConfig.model_validate(
        {"output_dir": str(output), "jobs": 2, "rules": {"motion_photos": {"enabled": False}}}
    )

    reporter = run_tool([lib], cfg, scan_root=lib)
    record = by_name(reporter)[fixtures.MP_FULL_COMBO]
    assert record.outcome is report.Outcome.DOWNGRADED, record.detail
    assert any("motion photo processing disabled" in note for note in record.notes)

    data = (output / fixtures.MP_FULL_COMBO).read_bytes()
    assert container.parse_container_items(container.extract_xmp_text(data)) == []
    assert exif_of(output / fixtures.MP_FULL_COMBO, "MotionPhoto", "MPImageStart") == ""


def test_embedded_video_within_target_is_passed_through_untouched(motion_run) -> None:
    """An embedded clip already at/below 720p must not be re-encoded.

    Re-encoding it would cost a generation of quality for no size win, so the bytes
    must come through identical to the input's.
    """
    output, _ = motion_run
    source_layout = fixtures.layout_of(fixtures.MOTION / fixtures.MP_SMALL_EMBEDDED_VIDEO)
    assert source_layout.video is not None
    assert source_layout.video.length > 0
    source_bytes = (fixtures.MOTION / fixtures.MP_SMALL_EMBEDDED_VIDEO).read_bytes()
    expected = source_bytes[source_layout.video.offset : source_layout.video.end]

    out_layout = fixtures.layout_of(output / fixtures.MP_SMALL_EMBEDDED_VIDEO)
    assert out_layout.video is not None
    out_bytes = (output / fixtures.MP_SMALL_EMBEDDED_VIDEO).read_bytes()
    assert out_bytes[out_layout.video.offset : out_layout.video.end] == expected


def test_oversized_embedded_video_is_re_encoded_to_720p(motion_run) -> None:
    """The counterpart: a clip above the target really is shrunk, and rotated
    correctly while doing it."""
    output, _ = motion_run
    layout = fixtures.layout_of(output / fixtures.MP_FULL_COMBO)
    assert layout.video is not None
    data = (output / fixtures.MP_FULL_COMBO).read_bytes()

    clip = output / "embedded.mp4"
    clip.write_bytes(data[layout.video.offset : layout.video.end])
    streams = probe_video(clip)
    # The source clip is stored landscape with a -90 matrix, so the orientation-aware
    # filter must produce a portrait result.
    assert (int(streams["width"]), int(streams["height"])) == fixtures.VIDEO_TARGET_PORTRAIT


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def video_run(tmp_path_factory) -> tuple[Path, report.Reporter]:
    output = tmp_path_factory.mktemp("video_out")
    cfg = config.GlobalConfig(output_dir=output, jobs=4)
    return output, run_tool([fixtures.VIDEOS], cfg, scan_root=fixtures.VIDEOS)


def test_no_video_fails(video_run) -> None:
    _, reporter = video_run
    failed = {r.path.name: r.detail for r in reporter.records if r.outcome is report.Outcome.FAILED}
    assert not failed, f"video failures: {failed}"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (fixtures.VIDEO_ROTATED_MINUS90, fixtures.VIDEO_TARGET_PORTRAIT),
        (fixtures.VIDEO_ROTATED_PLUS90, fixtures.VIDEO_TARGET_PORTRAIT),
        (fixtures.VIDEO_SQUARE, fixtures.VIDEO_TARGET_SQUARE),
        (fixtures.VIDEO_NO_AUDIO, fixtures.VIDEO_TARGET_LANDSCAPE),
        (fixtures.VIDEO_WITH_GPS, fixtures.VIDEO_TARGET_LANDSCAPE),
        (fixtures.VIDEO_TILDE_H264, fixtures.VIDEO_TARGET_LANDSCAPE),
    ],
)
def test_output_geometry_is_orientation_aware(video_run, name: str, expected) -> None:
    output, _ = video_run
    streams = probe_video(output / name)
    assert (int(streams["width"]), int(streams["height"])) == expected


def test_rotation_is_baked_into_the_pixels(video_run) -> None:
    """Output must not carry a leftover display matrix.

    The frames were physically rotated by the filter, so a surviving rotation tag
    would make a player turn the picture a second time.
    """
    output, _ = video_run
    for name in (fixtures.VIDEO_ROTATED_MINUS90, fixtures.VIDEO_ROTATED_PLUS90):
        # fmt: off
        rotation = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "side_data=rotation", "-of", "default=nw=1:nk=1",
             str(output / name)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        # fmt: on
        assert rotation in ("", "0")


def test_outputs_are_hevc_tagged_hvc1(video_run) -> None:
    """libx265 defaults to 'hev1', which some players reject."""
    output, reporter = video_run
    checked = 0
    for record in reporter.records:
        if record.outcome is not report.Outcome.SUCCEEDED:
            continue
        streams = probe_video(output / record.path.name)
        assert streams["codec_name"] == "hevc"
        assert streams["codec_tag_string"] == "hvc1"
        checked += 1
    assert checked


def test_small_video_is_copied_not_re_encoded(video_run) -> None:
    """Already under 720p: it reaches the output tree without touching an encoder."""
    output, reporter = video_run
    record = by_name(reporter)[fixtures.VIDEO_ALREADY_SMALL]
    assert record.outcome is report.Outcome.SKIPPED
    assert (output / fixtures.VIDEO_ALREADY_SMALL).read_bytes() == (
        fixtures.VIDEOS / fixtures.VIDEO_ALREADY_SMALL
    ).read_bytes()


def test_audioless_video_transcodes_without_error(video_run) -> None:
    output, reporter = video_run
    record = by_name(reporter)[fixtures.VIDEO_NO_AUDIO]
    assert record.outcome is report.Outcome.SUCCEEDED
    # fmt: off
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0",
         str(output / record.path.name)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    # fmt: on
    assert streams == "", "no audio stream should have been invented"


def test_audio_is_copied_untouched(video_run) -> None:
    output, _ = video_run
    # fmt: off
    args = ["ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels", "-of", "csv=p=0"]
    # fmt: on
    before = subprocess.run(
        [*args, str(fixtures.VIDEOS / fixtures.VIDEO_ROTATED_MINUS90)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    after = subprocess.run(
        [*args, str(output / fixtures.VIDEO_ROTATED_MINUS90)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert before.strip()
    assert before == after


def test_video_location_metadata_survives(video_run) -> None:
    """Regression guard for GPS loss.

    ``-map_metadata`` alone dropped the coordinates from the equivalent real file,
    which is only visible by diffing the full tag set -- nothing errors.
    """
    output, _ = video_run
    before = exif_of(fixtures.VIDEOS / fixtures.VIDEO_WITH_GPS, "GPSCoordinates")
    after = exif_of(output / fixtures.VIDEO_WITH_GPS, "GPSCoordinates")
    assert before, "fixture is supposed to carry GPS"
    assert after, "GPS coordinates were lost"
    # Compared numerically: a round-trip re-renders them at different precision.

    assert verify._values_match("GPSCoordinates", before, after)


def test_video_creation_metadata_survives(video_run) -> None:
    """ffmpeg drops these whenever a video filter is applied."""
    output, _ = video_run
    tags = ("CreateDate", "AndroidModel")
    before = exif_of(fixtures.VIDEOS / fixtures.VIDEO_ROTATED_MINUS90, *tags)
    after = exif_of(output / fixtures.VIDEO_ROTATED_MINUS90, *tags)
    assert "MCT TestCam" in before
    assert before == after


# ---------------------------------------------------------------------------
# Skipped files in a mirror output tree
# ---------------------------------------------------------------------------


def test_skipped_files_are_copied_into_the_mirror_tree(tmp_path) -> None:
    """A mirror tree must be complete, not full of holes where skips happened."""
    output = tmp_path / "out"
    reporter = run_tool([fixtures.IMAGES], config.GlobalConfig(output_dir=output, jobs=4), scan_root=fixtures.IMAGES)

    skipped = [r for r in reporter.records if r.outcome is report.Outcome.SKIPPED]
    assert len(skipped) == len(fixtures.SKIPPED_IMAGES)
    for record in skipped:
        copied = output / record.path.name
        assert copied.exists(), f"{record.path.name} missing from the output tree"
        assert copied.read_bytes() == record.path.read_bytes()
        assert "copied unchanged" in record.detail

    inputs = {p.name for p in fixtures.IMAGES.iterdir() if p.is_file()}
    assert {p.name for p in output.iterdir()} == inputs


def test_copied_skips_are_still_counted_as_skipped(tmp_path) -> None:
    """Copying is not compressing -- it must not inflate the success count."""
    output = tmp_path / "out"
    reporter = run_tool([fixtures.IMAGES], config.GlobalConfig(output_dir=output, jobs=4), scan_root=fixtures.IMAGES)
    record = by_name(reporter)[fixtures.IMAGE_ALREADY_SMALL]
    assert record.outcome is report.Outcome.SKIPPED
    assert record.saved == 0


def test_second_run_does_not_recopy_skipped_files(tmp_path) -> None:
    """Reruns over a populated tree must not rewrite every unchanged file."""
    output = tmp_path / "out"
    cfg = config.GlobalConfig(output_dir=output, jobs=4)

    run_tool([fixtures.IMAGES], cfg, scan_root=fixtures.IMAGES)
    copied = output / fixtures.IMAGE_ALREADY_SMALL
    first_inode = copied.stat().st_ino

    second = run_tool([fixtures.IMAGES], cfg, scan_root=fixtures.IMAGES)
    assert copied.stat().st_ino == first_inode, "file was rewritten instead of left alone"
    assert "already present" in by_name(second)[fixtures.IMAGE_ALREADY_SMALL].detail


def test_replace_mode_does_not_copy_skipped_files(tmp_path) -> None:
    """In replace mode a skipped file is already where it belongs."""
    workdir = tmp_path / "lib"
    workdir.mkdir()
    shutil.copy2(fixtures.IMAGES / fixtures.IMAGE_ALREADY_SMALL, workdir)

    reporter = run_tool([workdir], config.GlobalConfig(strategy=config.Strategy.REPLACE, jobs=2), scan_root=workdir)
    assert [p.name for p in workdir.iterdir()] == [fixtures.IMAGE_ALREADY_SMALL]
    assert "copied" not in by_name(reporter)[fixtures.IMAGE_ALREADY_SMALL].detail
