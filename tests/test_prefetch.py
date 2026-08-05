"""Bulk probing tests: it must be a pure optimisation, never a second opinion.

Planning used to spend most of a large run's wall clock launching one exiftool or
ffprobe per file -- ~95 ms of interpreter startup each, for a few measured numbers.
:func:`media_condenser.planner.prefetch` reads them in batches up front instead, which leaves one
thing to prove over and over here: that a warm cache and a cold one produce exactly
the same plan, including for the files that cannot be read at all.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import fixtures
import pytest

from media_condenser import config, container, discovery, planner, probe


def fingerprint(action: planner.Action) -> tuple:
    """Everything about an action that a later phase can act on."""
    return (
        str(action.path),
        str(action.kind),
        str(action.verb),
        action.reason,
        action.target_size,
        tuple(action.notes),
        action.clear_motion_container,
        None if action.image_info is None else (action.image_info.width, action.image_info.height),
        None if action.video_info is None else action.video_info.display_size,
        None if action.layout is None else (action.layout.is_motion_photo, action.layout.has_attachments),
    )


def plan_all(candidates: list[discovery.Candidate], *, warm: bool) -> list[tuple]:
    """Plan every candidate, with the caches either prefilled in bulk or cold."""
    prober = probe.Prober(config.ToolPaths())
    if warm:
        asyncio.run(planner.prefetch(candidates, prober, concurrency=8))
    return [fingerprint(planner.plan(candidate, prober)) for candidate in candidates]


@pytest.fixture(scope="module")
def candidates() -> list[discovery.Candidate]:
    cfg = config.GlobalConfig()
    return discovery.walk([fixtures.ROOT], config.RulesResolver(cfg.rules, [fixtures.ROOT]))


def test_prefetching_does_not_change_the_plan(candidates) -> None:
    """The whole point: same decisions, same reasons, same measured numbers.

    Over the real fixture library, so it covers the awkward ones too -- the mislabeled
    ``.png`` that is a JPEG, the motion photos whose declared components do not fit in
    the file, the rotated videos whose display size differs from their stored size.
    """
    assert plan_all(candidates, warm=True) == plan_all(candidates, warm=False)


def test_prefetch_fills_the_caches_it_is_given(candidates) -> None:
    """Otherwise the tests above would pass by simply not prefetching anything."""
    prober = probe.Prober(config.ToolPaths())
    asyncio.run(planner.prefetch(candidates, prober, concurrency=8))

    images = [c for c in candidates if c.kind in (probe.JPEG, probe.PNG)]
    videos = [c for c in candidates if c.kind == probe.VIDEO]
    assert images and videos, "fixture library should hold both"
    # Read back with the tools removed, so anything not already cached cannot be
    # probed lazily and has to fail instead of quietly succeeding.
    blinded = probe.Prober(config.ToolPaths(exiftool="/nonexistent", ffprobe="/nonexistent"))
    blinded._exif_cache = prober._exif_cache
    blinded._ffprobe_cache = prober._ffprobe_cache
    for candidate in images:
        assert blinded.image_info(candidate.path).long_edge > 0
    for candidate in videos:
        assert blinded.video_info(candidate.path).short_edge > 0


def test_progress_is_reported_once_per_candidate(candidates) -> None:
    """Including the kinds that are never probed at all.

    The caller sizes its progress bar from the candidate count, so a HEIC -- which is
    classified as unsupported without being probed -- still has to be counted or the
    bar stops short of its total on any library containing one.
    """
    prober = probe.Prober(config.ToolPaths())
    counted = 0

    def advance(amount: int) -> None:
        nonlocal counted
        counted += amount

    asyncio.run(planner.prefetch(candidates, prober, concurrency=8, on_progress=advance))
    assert counted == len(candidates)


def test_already_cached_files_are_still_counted(candidates) -> None:
    prober = probe.Prober(config.ToolPaths())
    asyncio.run(planner.prefetch(candidates, prober, concurrency=8))

    counted = 0

    def advance(amount: int) -> None:
        nonlocal counted
        counted += amount

    asyncio.run(planner.prefetch(candidates, prober, concurrency=8, on_progress=advance))
    assert counted == len(candidates)


# -- failures inside a batch ------------------------------------------


def test_one_unreadable_file_does_not_spoil_its_batch(tmp_path: Path) -> None:
    """exiftool exits non-zero if any file in a batch failed, and still reports the rest.

    Treating that exit status as fatal would throw away a few hundred good reads over
    one bad photo -- and then re-do them all lazily.
    """
    good = tmp_path / "good.jpg"
    good.write_bytes((fixtures.IMAGES / fixtures.IMAGE_LANDSCAPE).read_bytes())
    missing = tmp_path / "not_there.jpg"

    prober = probe.Prober(config.ToolPaths())
    asyncio.run(prober.prefetch(exif=[good, missing]))

    assert good in prober._exif_cache
    assert missing not in prober._exif_cache


def test_a_file_the_batch_missed_is_probed_lazily(tmp_path: Path) -> None:
    """The fallback that makes every failure above harmless rather than silent."""
    source = tmp_path / "photo.jpg"
    source.write_bytes((fixtures.IMAGES / fixtures.IMAGE_LANDSCAPE).read_bytes())

    prober = probe.Prober(config.ToolPaths())
    asyncio.run(prober.prefetch(exif=[tmp_path / "absent.jpg"]))
    assert source not in prober._exif_cache
    # Never batched, never cached -- and still measurable.
    assert prober.image_info(source).long_edge == max(fixtures.IMAGE_SIZE_LANDSCAPE)


def test_an_unprobeable_video_still_fails_with_its_own_message(tmp_path: Path) -> None:
    """A prefetch miss must not turn into a silent skip.

    The prefetch leaves an unreadable file uncached precisely so the lazy call runs and
    raises, which is what puts the file in the summary as failed with a real reason.
    """
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\xff" * 64)

    prober = probe.Prober(config.ToolPaths())
    asyncio.run(prober.prefetch(video=[broken]))
    assert broken not in prober._ffprobe_cache

    candidate = discovery.Candidate(path=broken, kind=probe.VIDEO, rules=config.Rules(), size=broken.stat().st_size)
    action = planner.plan(candidate, prober)
    assert action.verb is planner.Verb.FAIL
    assert action.reason


# -- names an argument file cannot express ----------------------------


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("new\nline.jpg", id="newline"),
        pytest.param("trailing space .jpg ", id="trailing-space"),
    ],
)
def test_names_a_batch_cannot_express_fall_back(tmp_path: Path, name: str) -> None:
    """Batch arguments go in line by line, stripped, so some names cannot be batched.

    Those must be excluded and probed individually. Passing them anyway would have
    exiftool read a *different* path than the one asked about -- silently measuring
    the wrong file, which is the one outcome worth being slow to avoid.
    """
    photo = tmp_path / name
    photo.write_bytes((fixtures.IMAGES / fixtures.IMAGE_LANDSCAPE).read_bytes())

    prober = probe.Prober(config.ToolPaths())
    asyncio.run(prober.prefetch(exif=[photo]))
    assert photo not in prober._exif_cache
    assert prober.image_info(photo).long_edge == max(fixtures.IMAGE_SIZE_LANDSCAPE)


# -- the XMP packet, now read alongside every other tag ---------------


@pytest.mark.parametrize("name", fixtures.ALL_MOTION_PHOTOS)
def test_xmp_packet_matches_the_dedicated_read(name: str) -> None:
    """It used to be its own ``exiftool -b -XMP`` launch per JPEG, for one boolean.

    Folding ``-XMP`` into the shared tag list is what halved the per-image cost, so
    what it returns has to stay byte-identical to what the separate call produced.
    """
    path = fixtures.MOTION / name
    expected = subprocess.run(
        ["exiftool", "-b", "-XMP", str(path)],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8", "replace")

    assert probe.Prober(config.ToolPaths()).xmp_packet(path) == expected
    assert container.is_motion_photo(expected), "fixture should be flagged as a motion photo"


def test_xmp_packet_is_empty_for_a_file_without_one(tmp_path: Path) -> None:
    """A missing key must read as "no XMP", not as a failure."""
    plain = tmp_path / "plain.jpg"
    plain.write_bytes((fixtures.IMAGES / fixtures.IMAGE_ALREADY_SMALL).read_bytes())
    packet = probe.Prober(config.ToolPaths()).xmp_packet(plain)
    assert not container.is_motion_photo(packet)


def test_binary_tags_are_decoded_when_exiftool_base64s_them() -> None:
    """A packet that is not valid text comes back base64-encoded under ``-b``.

    Reading that literally would make a motion photo look like a plain image and
    drop its embedded components, so the encoding is undone rather than trusted.
    """
    assert probe._decode_binary_tag("base64:aGVsbG8=") == "hello"
    assert probe._decode_binary_tag("<?xpacket begin=") == "<?xpacket begin="
    assert probe._decode_binary_tag("base64:@@not@@base64@@") == ""
    assert probe._decode_binary_tag(None) == ""
