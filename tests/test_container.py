"""Container parsing tests, driven by the real fixture files.

Each fixture exists to exercise one specific failure mode, so the assertions are
written per file rather than as a generic loop -- when one breaks, the name of the
test says which real-world case regressed.
"""

from __future__ import annotations

import fixtures
import pytest

from media_condenser import container


def layout_for(name: str):
    return fixtures.layout_of(fixtures.MOTION / name)


@pytest.mark.parametrize("name", fixtures.ALL_MOTION_PHOTOS)
def test_every_fixture_is_detected_as_a_motion_photo(name: str) -> None:
    """Detection is content-based, so it must work across all naming variants."""
    assert container.is_motion_photo(container.extract_xmp_text((fixtures.MOTION / name).read_bytes()))


@pytest.mark.parametrize("name", fixtures.ALL_MOTION_PHOTOS)
def test_declared_items_are_parsed_for_every_serialization(name: str) -> None:
    """Both RDF forms must yield items; an empty list means the parser missed one.

    This is the regression guard for the bug where only ``rdf:Seq``/attribute-style
    containers parsed, which made valid ``rdf:Bag`` files look like they had no
    attachments at all.
    """
    items = container.parse_container_items(container.extract_xmp_text((fixtures.MOTION / name).read_bytes()))
    assert items, "no container items parsed"
    assert items[0].semantic == container.Semantic.PRIMARY


@pytest.mark.parametrize("name", fixtures.ALL_MOTION_PHOTOS)
def test_resolved_components_stay_inside_the_file(name: str) -> None:
    layout = layout_for(name)
    for component in (layout.primary, layout.gain_map, layout.video):
        if component is not None:
            assert component.offset >= 0
            assert component.end <= layout.file_size


def test_full_combo_resolves_all_three_components() -> None:
    """Portrait primary + Ultra HDR gain map + rotated video, all at once."""
    layout = layout_for(fixtures.MP_FULL_COMBO)
    assert layout.gain_map is not None
    assert layout.video is not None
    # The components must tile the file exactly, with nothing left over.
    assert layout.primary.length + layout.gain_map.length + layout.video.length == layout.file_size


@pytest.mark.parametrize("name", [fixtures.MP_VIDEO_ONLY_LANDSCAPE, fixtures.MP_VIDEO_ONLY_PORTRAIT])
def test_video_only_file_has_no_gain_map(name: str) -> None:
    """A two-item container that is video, not gain map.

    Positional parsing would confuse this with the gain-map-only file, which also
    declares exactly two items -- which is why items are matched by label.
    """
    layout = layout_for(name)
    assert layout.gain_map is None
    assert layout.video is not None
    assert layout.primary.length + layout.video.length == layout.file_size


def test_gain_map_only_file_has_no_video_and_ignores_orphaned_bytes() -> None:
    """The other two-item shape, plus unreferenced trailing data.

    Locating the gain map by subtracting lengths from the end of the file would land
    inside the orphaned blob.
    """
    layout = layout_for(fixtures.MP_GAINMAP_ORPHANED)
    assert layout.video is None
    assert layout.gain_map is not None
    orphaned = layout.file_size - layout.gain_map.end
    assert orphaned > layout.gain_map.length, "orphan must be big enough to mislead"
    assert any("orphaned" in note for note in layout.notes)


def test_mpf_offset_is_what_defeats_the_orphaned_bytes() -> None:
    """The absolute MPF offset is load-bearing, not merely redundant.

    Re-resolving the same file without the MPF anchor must lose the gain map: that is
    the failure end-subtraction produces, and the reason MPF is preferred when present.
    """
    path = fixtures.MOTION / fixtures.MP_GAINMAP_ORPHANED
    data = path.read_bytes()
    offset, length = fixtures.mpf_tags(path)

    with_mpf = container.resolve_layout(path, data, mpf_offset=offset, mpf_length=length)
    without_mpf = container.resolve_layout(path, data, mpf_offset=None, mpf_length=None)

    assert with_mpf.gain_map is not None
    assert without_mpf.gain_map is None


def test_impossible_declared_video_length_is_rejected_not_fatal() -> None:
    """Declared video length exceeds the whole file; the gain map is still fine."""
    layout = layout_for(fixtures.MP_IMPOSSIBLE_VIDEO_LEN)
    assert layout.video is None, "an impossible length must not resolve"
    assert layout.gain_map is not None, "the valid gain map must survive"
    assert any("exceeds file size" in note for note in layout.notes)


def test_bag_serialized_file_keeps_its_gain_map() -> None:
    """Regression guard for the misdiagnosed 'stale flag' case.

    ExifTool flattens this file's ``rdf:Bag`` form to ``DirectorySemantic``/
    ``DirectoryLength`` instead of ``DirectoryItemSemantic``/``DirectoryItemLength``,
    which makes a complete container look empty. The real file this was modelled on
    holds a valid gain map ending exactly at EOF; treating it as stale silently
    discards working HDR.
    """
    layout = layout_for(fixtures.MP_BAG_FORM)
    assert layout.gain_map is not None
    assert layout.gain_map.end == layout.file_size


def test_gain_map_offset_points_at_a_jpeg_signature() -> None:
    for name in fixtures.ALL_MOTION_PHOTOS:
        layout = layout_for(name)
        if layout.gain_map is None:
            continue
        data = (fixtures.MOTION / name).read_bytes()
        assert data[layout.gain_map.offset : layout.gain_map.offset + 3] == b"\xff\xd8\xff"


def test_video_offset_points_at_an_iso_ftyp_box() -> None:
    for name in fixtures.ALL_MOTION_PHOTOS:
        layout = layout_for(name)
        if layout.video is None:
            continue
        data = (fixtures.MOTION / name).read_bytes()
        assert data[layout.video.offset + 4 : layout.video.offset + 8] == b"ftyp"


def test_both_serializations_are_represented_in_the_library() -> None:
    """Guards the fixtures themselves.

    If regeneration ever emitted a single form, every parser test above would still
    pass while silently covering half of what it claims to.
    """
    forms = set()
    for name in fixtures.ALL_MOTION_PHOTOS:
        xmp = container.extract_xmp_text((fixtures.MOTION / name).read_bytes())
        forms.add("bag" if "rdf:Bag" in xmp else "seq")
    assert forms == {"bag", "seq"}


# ---------------------------------------------------------------------------
# Directory writing
# ---------------------------------------------------------------------------


def test_written_directory_round_trips_through_the_parser() -> None:
    items = [
        container.ContainerItem(semantic=container.Semantic.PRIMARY, length=0, mime="image/jpeg"),
        container.ContainerItem(semantic=container.Semantic.GAIN_MAP, length=1234, mime="image/jpeg"),
        container.ContainerItem(semantic=container.Semantic.MOTION_PHOTO, length=5678, mime="video/mp4"),
    ]
    xml = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description rdf:about="">{container.build_directory_xml(items)}</rdf:Description>'
        "</rdf:RDF></x:xmpmeta>"
    )
    parsed = container.parse_container_items(xml)
    assert [(item.semantic, item.length) for item in parsed] == [
        ("Primary", 0),
        ("GainMap", 1234),
        ("MotionPhoto", 5678),
    ]


@pytest.mark.parametrize("name", [fixtures.MP_FULL_COMBO, fixtures.MP_BAG_FORM])
def test_directory_replacement_works_on_both_input_forms(name: str) -> None:
    """Writing must overwrite the directory whatever form it arrived in."""
    original = container.extract_xmp_text((fixtures.MOTION / name).read_bytes())
    items = [
        container.ContainerItem(semantic=container.Semantic.PRIMARY, length=0, mime="image/jpeg"),
        container.ContainerItem(semantic=container.Semantic.GAIN_MAP, length=42, mime="image/jpeg"),
    ]
    updated = container.set_container_directory(original, items)
    assert [(i.semantic, i.length) for i in container.parse_container_items(updated)] == [
        ("Primary", 0),
        ("GainMap", 42),
    ]
    # Unrelated metadata must survive the splice.
    assert "GCamera:MotionPhoto" in updated


def test_directory_replacement_survives_exiftool_prefix_renaming() -> None:
    """exiftool re-serializes the namespace as ``GContainer:``/``GItem:``.

    The files this tool writes therefore come back with different prefixes than the
    ones it read, so prefix handling must be driven by the xmlns declarations. No
    fixture is in this form -- it only ever arrives from our own output.
    """
    xmp = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        "<rdf:Description rdf:about=''"
        " xmlns:GContainer='http://ns.google.com/photos/1.0/container/'"
        " xmlns:GItem='http://ns.google.com/photos/1.0/container/item/'>"
        "<GContainer:Directory><rdf:Seq><rdf:li rdf:parseType='Resource'>"
        "<GContainer:Item rdf:parseType='Resource'>"
        "<GItem:Length>999</GItem:Length><GItem:Semantic>Primary</GItem:Semantic>"
        "</GContainer:Item></rdf:li></rdf:Seq></GContainer:Directory>"
        "</rdf:Description></rdf:RDF></x:xmpmeta>"
    )
    assert [(i.semantic, i.length) for i in container.parse_container_items(xmp)] == [("Primary", 999)]
    updated = container.set_container_directory(
        xmp, [container.ContainerItem(semantic=container.Semantic.PRIMARY, length=0, mime="image/jpeg")]
    )
    assert [(i.semantic, i.length) for i in container.parse_container_items(updated)] == [("Primary", 0)]
