"""Motion photo container parsing and rebuilding.

A Google motion photo is 2-3 files concatenated back to back inside what still
looks like a plain JPEG::

    [ primary JPEG ][ Ultra HDR gain map JPEG ][ MP4 video ]

The gain map and the video are both optional and *independently* so. The layout is
described by an XMP ``Container:Directory``, and the gain map's byte offset is
additionally recorded in the MPF (APP2) block.

Everything in this module is written against real-file behaviour that contradicts
the obvious implementation. In particular:

- The directory appears in **two different RDF serializations** (see
  :func:`parse_container_items`), which exiftool flattens to *different tag names*.
  Reading the flattened tags makes a perfectly valid file look empty.
- Items must be matched **by label**, never by position. A gain-map-only export and
  a video-only file both have two items and mean different things.
- Declared lengths can be larger than the whole file, and files can carry orphaned
  trailing data no longer referenced by anything. Every offset is range-checked and
  content-checked; failures degrade to "component absent" rather than raising.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

# XMP namespaces used by the container.
NS_CONTAINER = "http://ns.google.com/photos/1.0/container/"
NS_ITEM = "http://ns.google.com/photos/1.0/container/item/"
NS_CAMERA = "http://ns.google.com/photos/1.0/camera/"
NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

_XMP_STD_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
_XMP_EXT_HEADER = b"http://ns.adobe.com/xmp/extension/\x00"

_APP1 = 0xE1
_SOS = 0xDA
_EOI = 0xD9

#: Largest payload a single JPEG APP segment can hold (2-byte length includes itself).
_MAX_SEGMENT_PAYLOAD = 65533


class Semantic(StrEnum):
    PRIMARY = "Primary"
    GAIN_MAP = "GainMap"
    MOTION_PHOTO = "MotionPhoto"


@dataclass(frozen=True)
class ContainerItem:
    """One entry of the XMP ``Container:Directory``, as declared."""

    semantic: str
    length: int
    mime: str = ""
    padding: int = 0


@dataclass(frozen=True)
class Component:
    """A component whose byte range has been *resolved and validated*."""

    semantic: str
    offset: int
    length: int
    mime: str = ""

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass
class MotionPhotoLayout:
    """The resolved structure of a motion photo file."""

    path: Path
    file_size: int
    is_motion_photo: bool
    items: list[ContainerItem] = field(default_factory=list)
    primary: Component | None = None
    gain_map: Component | None = None
    video: Component | None = None
    notes: list[str] = field(default_factory=list)
    """Human-readable reasons a declared component was dropped. These drive the
    'downgraded' bucket in the summary rather than being treated as failures."""

    @property
    def has_attachments(self) -> bool:
        return self.gain_map is not None or self.video is not None


# ---------------------------------------------------------------------------
# JPEG segment walking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    marker: int
    offset: int
    """Offset of the 0xFF marker prefix."""
    seg_length: int
    """Value of the 2-byte length field (covers itself plus the payload)."""

    @property
    def payload_offset(self) -> int:
        return self.offset + 4

    @property
    def payload_length(self) -> int:
        return self.seg_length - 2

    @property
    def end(self) -> int:
        return self.offset + 2 + self.seg_length


_SOI = b"\xff\xd8"
"""JPEG start-of-image marker."""

_JPEG_SIGNATURE = b"\xff\xd8\xff"
"""SOI plus the first marker byte. What a resolved gain map has to start with."""


def _gain_map_at(data: bytes, offset: int, length: int, mime: str) -> Component | None:
    """A gain map component at ``offset``, or ``None`` if nothing is there.

    The signature test is the one that matters: both callers derive their offset
    arithmetically, and a computed position that happens to land inside orphaned
    trailing data would otherwise be handed on as HDR.
    """
    if data[offset : offset + len(_JPEG_SIGNATURE)] != _JPEG_SIGNATURE:
        return None
    return Component(semantic=Semantic.GAIN_MAP, offset=offset, length=length, mime=mime or "image/jpeg")


def iter_segments(data: bytes):
    """Yield JPEG marker segments up to (not including) the first SOS.

    Stops at SOS deliberately: past that point the byte stream is entropy-coded
    scan data where a scan for 0xFFD9 finds false positives. Real files proved
    this -- walking for an EOI marker landed 15 KB early on two fixtures, inside
    appended debug data. Nothing here needs to know where the scan ends, so the
    walker refuses to guess.
    """
    if not data.startswith(_SOI):
        return
    i = 2
    n = len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker in (_SOS, _EOI):
            return
        if i + 4 > n:
            return
        seg_length = int.from_bytes(data[i + 2 : i + 4], "big")
        if seg_length < 2 or i + 2 + seg_length > n:
            return
        yield Segment(marker=marker, offset=i, seg_length=seg_length)
        i += 2 + seg_length


def find_standard_xmp(data: bytes) -> Segment | None:
    """Locate the *standard* XMP APP1 segment.

    Deliberately skips extended-XMP segments (``HasExtendedXMP``), which hold the
    multi-kilobyte ``HDRPlusMakerNote`` payload. The container directory always
    lives in the standard packet -- around 1.5 KB in practice, so patching it stays
    well inside the 64 KB per-segment ceiling.
    """
    for segment in iter_segments(data):
        if segment.marker != _APP1:
            continue
        payload = data[segment.payload_offset : segment.payload_offset + segment.payload_length]
        if payload.startswith(_XMP_STD_HEADER):
            return segment
    return None


def extract_xmp_text(data: bytes) -> str:
    """The standard XMP packet as text, or ``""`` when absent."""
    segment = find_standard_xmp(data)
    if segment is None:
        return ""
    start = segment.payload_offset + len(_XMP_STD_HEADER)
    end = segment.payload_offset + segment.payload_length
    return data[start:end].decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# XMP container directory
# ---------------------------------------------------------------------------


def is_motion_photo(xmp_text: str) -> bool:
    """Content-based motion photo detection.

    Checks ``GCamera:MotionPhoto`` in the XMP and nothing else. Filename patterns
    (``.MP.jpg``, ``~2``, ``-bearbeitet``) are not reliable in either direction:
    real files exist without the pattern that are motion photos, and with it that
    are not.
    """
    if not xmp_text:
        return False
    flag = _find_gcamera_flag(xmp_text, "MotionPhoto")
    return flag == "1"


def _find_gcamera_flag(xmp_text: str, name: str) -> str | None:
    attr = re.search(rf'GCamera:{name}\s*=\s*"([^"]*)"', xmp_text)
    if attr:
        return attr.group(1).strip()
    element = re.search(rf"<GCamera:{name}>([^<]*)</GCamera:{name}>", xmp_text)
    if element:
        return element.group(1).strip()
    return None


def parse_container_items(xmp_text: str) -> list[ContainerItem]:
    """Parse ``Container:Directory`` into an ordered list of declared items.

    Handles both serializations found in real files -- this is the single most
    important detail in the module:

    ``rdf:Seq`` holding a ``Container:Item`` whose fields are **XML attributes**
    (5 of 7 fixtures)::

        <Container:Directory><rdf:Seq>
          <rdf:li rdf:parseType="Resource">
            <Container:Item Item:Semantic="GainMap" Item:Length="9119"/>

    ``rdf:Bag`` holding an ``rdf:Description`` whose fields are **child elements**
    (both ``-bearbeitet`` fixtures)::

        <Container:Directory><rdf:Bag>
          <rdf:li><rdf:Description>
            <Item:Semantic>GainMap</Item:Semantic>
            <Item:Length>59087</Item:Length>

    ExifTool flattens the first to ``DirectoryItemLength`` and the second to
    ``DirectoryLength``. Reading only the former makes the second form look like a
    file with no attachments at all, which is exactly the trap that would silently
    discard a working HDR gain map.
    """
    if not xmp_text:
        return []
    try:
        root = ET.fromstring(_strip_xpacket(xmp_text))
    except ET.ParseError:
        return []

    directory = root.find(f".//{{{NS_CONTAINER}}}Directory")
    if directory is None:
        return []

    items: list[ContainerItem] = []
    for li in directory.iter(f"{{{NS_RDF}}}li"):
        # The item fields turn up at three different depths across real files:
        # on the <rdf:li> itself, on a nested <Container:Item> (form A), or on a
        # nested <rdf:Description> (form B). Rather than enumerate the shapes,
        # scan the whole <rdf:li> subtree and take the first value found for each
        # field -- the container spec allows only one item per entry, so there is
        # nothing to disambiguate.
        semantic = mime = ""
        length = padding = 0
        for holder in li.iter():
            semantic = semantic or _field_value(holder, "Semantic")
            mime = mime or _field_value(holder, "Mime")
            length = length or _int_field(holder, "Length")
            padding = padding or _int_field(holder, "Padding")
        if semantic:
            items.append(ContainerItem(semantic=semantic, length=length, mime=mime, padding=padding))
    return items


def _field_value(holder: ET.Element, name: str) -> str:
    value = holder.get(f"{{{NS_ITEM}}}{name}")
    if value is not None:
        return value.strip()
    child = holder.find(f"{{{NS_ITEM}}}{name}")
    if child is not None and child.text:
        return child.text.strip()
    return ""


def _int_field(holder: ET.Element, name: str) -> int:
    raw = _field_value(holder, name)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _strip_xpacket(xmp_text: str) -> str:
    """Drop the ``<?xpacket ...?>`` processing instructions and trailing padding."""
    text = re.sub(r"<\?xpacket[^>]*\?>", "", xmp_text)
    return text.strip().strip("\x00")


# ---------------------------------------------------------------------------
# Layout resolution
# ---------------------------------------------------------------------------


def resolve_layout(
    path: Path,
    data: bytes,
    *,
    mpf_offset: int | None,
    mpf_length: int | None,
) -> MotionPhotoLayout:
    """Resolve declared items into validated byte ranges.

    ``mpf_offset``/``mpf_length`` come from the MPF ``MPImageStart``/``MPImageLength``
    tags and are the authoritative gain map position.
    """
    size = len(data)
    xmp_text = extract_xmp_text(data)
    layout = MotionPhotoLayout(
        path=path,
        file_size=size,
        is_motion_photo=is_motion_photo(xmp_text),
        items=parse_container_items(xmp_text),
    )

    by_semantic = {item.semantic: item for item in layout.items}

    gain_item = by_semantic.get(Semantic.GAIN_MAP)
    video_item = by_semantic.get(Semantic.MOTION_PHOTO)

    # -- gain map, via MPF --------------------------------------------
    # MPF is the preferred anchor because it is an absolute offset and therefore
    # immune to orphaned trailing data: one fixture carries 2.38 MB of leftover MP4
    # no longer referenced by anything, and subtracting lengths off the end of the
    # file lands inside it.
    if gain_item is not None and mpf_offset is not None:
        length = mpf_length or gain_item.length
        if length <= 0:
            layout.notes.append("gain map declared with zero length; treating as absent")
        elif mpf_offset + length > size:
            layout.notes.append(f"gain map range {mpf_offset}+{length} exceeds file size {size}; treating as absent")
        elif (found := _gain_map_at(data, mpf_offset, length, gain_item.mime)) is None:
            layout.notes.append(f"no JPEG signature at gain map offset {mpf_offset}; treating as absent")
        else:
            layout.gain_map = found

    # -- video ---------------------------------------------------------
    if video_item is not None:
        if video_item.length <= 0:
            layout.notes.append("video declared with zero length; treating as absent")
        else:
            # With a gain map anchored by MPF the video starts right after it.
            # Otherwise fall back to end-subtraction, accepted only if an ISO
            # base-media 'ftyp' box really sits at the computed offset.
            anchored = layout.gain_map is not None
            offset = layout.gain_map.end if anchored else size - video_item.length
            if offset < 0 or offset + video_item.length > size:
                layout.notes.append(
                    f"video range {offset}+{video_item.length} exceeds file size {size}; treating as absent"
                )
            elif data[offset + 4 : offset + 8] != b"ftyp":
                layout.notes.append(f"no ISO 'ftyp' box at video offset {offset}; treating as absent")
            else:
                layout.video = Component(
                    semantic=Semantic.MOTION_PHOTO,
                    offset=offset,
                    length=video_item.length,
                    mime=video_item.mime or "video/mp4",
                )

    # -- gain map, without MPF ----------------------------------------
    # Files this tool writes have no MPF block: it cannot be regenerated by the
    # metadata copy, and the XMP directory is authoritative anyway. So when MPF is
    # missing, derive the position from the declared lengths instead of giving up --
    # otherwise a rebuilt file would lose its HDR on the next read, which is
    # exactly what a round-trip test caught.
    if layout.gain_map is None and gain_item is not None and gain_item.length > 0:
        anchor = layout.video.offset if layout.video is not None else size
        offset = anchor - gain_item.length
        if offset < 0:
            layout.notes.append(
                f"gain map length {gain_item.length} does not fit before offset {anchor}; treating as absent"
            )
        elif (found := _gain_map_at(data, offset, gain_item.length, gain_item.mime)) is None:
            layout.notes.append(f"no JPEG signature at derived gain map offset {offset}; treating as absent")
        else:
            layout.gain_map = found

    # -- primary -------------------------------------------------------
    # The primary is everything before the first resolved attachment. Declared
    # length is always 0 ("to the start of the next item"), so it is derived.
    primary_end = size
    if layout.gain_map is not None:
        primary_end = layout.gain_map.offset
    elif layout.video is not None:
        primary_end = layout.video.offset
    layout.primary = Component(semantic=Semantic.PRIMARY, offset=0, length=primary_end, mime="image/jpeg")

    trailing = size - max(
        (c.end for c in (layout.primary, layout.gain_map, layout.video) if c is not None),
        default=size,
    )
    if trailing > 0:
        layout.notes.append(f"{trailing} orphaned trailing bytes ignored")

    return layout


# ---------------------------------------------------------------------------
# Rebuilding
# ---------------------------------------------------------------------------


def build_directory_xml(items: list[ContainerItem]) -> str:
    """Serialize a container directory in one canonical form.

    Output is always ``rdf:Seq`` + ``Container:Item`` with the fields as attributes,
    matching what the camera itself writes. The namespaces are declared on the
    element so the fragment is valid wherever it is spliced in.
    """
    lines = [
        f'<Container:Directory xmlns:Container="{NS_CONTAINER}" xmlns:Item="{NS_ITEM}">',
        "<rdf:Seq>",
    ]
    for item in items:
        attributes = [
            f'Item:Mime="{item.mime}"',
            f'Item:Semantic="{item.semantic}"',
            f'Item:Length="{item.length}"',
            f'Item:Padding="{item.padding}"',
        ]
        lines.append('<rdf:li rdf:parseType="Resource">')
        lines.append("<Container:Item " + " ".join(attributes) + "/>")
        lines.append("</rdf:li>")
    lines += ["</rdf:Seq>", "</Container:Directory>"]
    return "".join(lines)


def set_container_directory(xmp_text: str, items: list[ContainerItem]) -> str:
    """Replace the container directory with a freshly generated one.

    Writing the directory outright, rather than patching lengths inside whatever
    structure happens to be there, is a deliberate choice forced by two findings:

    - ``exiftool -tagsFromFile`` **corrupts** the ``rdf:Bag`` form. Copying tags
      from a ``-bearbeitet`` fixture produced a directory of ``<GDepth:Mime>``
      entries with the semantics and lengths silently dropped. Anything that edits
      the copied structure in place inherits that damage.
    - exiftool also renames the prefixes (``Container:``/``Item:`` become
      ``GContainer:``/``GItem:``), so the structure to be edited is not the one
      that was read.

    Reading stays permissive across all observed forms; writing is narrowed to one.
    """
    if not xmp_text:
        return xmp_text

    directory = build_directory_xml(items)
    region = _directory_region(xmp_text)
    if region is not None:
        start, end = region
        return xmp_text[:start] + directory + xmp_text[end:]

    # No directory present: attach a new rdf:Description carrying just this one.
    closing = xmp_text.rfind("</rdf:RDF>")
    if closing == -1:
        return xmp_text
    description = '<rdf:Description rdf:about="">' + directory + "</rdf:Description>"
    return xmp_text[:closing] + description + xmp_text[closing:]


def _prefix_pattern(xmp_text: str, namespace: str) -> str:
    """A regex alternation of every prefix bound to ``namespace``.

    Prefixes are always resolved from the ``xmlns`` declarations rather than
    assumed, because the same namespace appears under different prefixes depending
    on who wrote the file. Falls back to a generic matcher when no declaration is
    present.
    """
    found = re.findall(
        rf"""xmlns:([\w.\-]+)\s*=\s*["']{re.escape(namespace)}["']""",
        xmp_text,
    )
    if not found:
        return r"[\w.\-]+"
    return "(?:" + "|".join(re.escape(p) for p in dict.fromkeys(found)) + ")"


def _directory_region(xmp_text: str) -> tuple[int, int] | None:
    """Character range of the ``Directory`` element, whatever prefix it uses."""
    prefix = _prefix_pattern(xmp_text, NS_CONTAINER)
    open_match = re.search(rf"<{prefix}:Directory\b", xmp_text)
    if open_match is None:
        return None
    close_match = re.search(rf"</{prefix}:Directory>", xmp_text[open_match.end() :])
    if close_match is None:
        return None
    return open_match.start(), open_match.end() + close_match.end()


class ContainerRebuildError(RuntimeError):
    """The rebuilt container could not be assembled."""


def replace_standard_xmp(data: bytes, new_xmp: str) -> bytes:
    """Return ``data`` with its standard XMP APP1 payload replaced.

    The 2-byte segment length is recomputed, so the replacement may differ in size
    from the original.
    """
    segment = find_standard_xmp(data)
    if segment is None:
        raise ContainerRebuildError("no standard XMP segment to replace")

    payload = _XMP_STD_HEADER + new_xmp.encode("utf-8")
    if len(payload) > _MAX_SEGMENT_PAYLOAD:
        raise ContainerRebuildError(f"patched XMP is {len(payload)} bytes, over the {_MAX_SEGMENT_PAYLOAD} APP1 limit")
    seg_length = len(payload) + 2
    rebuilt = (
        data[: segment.offset] + bytes([0xFF, _APP1]) + seg_length.to_bytes(2, "big") + payload + data[segment.end :]
    )
    return rebuilt


def assemble(primary: bytes, gain_map: bytes | None, video: bytes | None) -> bytes:
    """Concatenate the components and write a directory describing exactly them.

    The directory is generated from the bytes actually being written, so a component
    that got dropped during the rebuild simply has no entry. That matters: had it
    kept its original non-zero length, a reader trusting the declaration would slice
    whatever bytes happened to follow and hand back garbage as a video.
    ``Primary`` stays 0 by spec -- it means "up to the next item".

    MPF is deliberately **not** patched. Its ``MPImageStart`` offset duplicates the
    gain map position and is now stale, but relying on the XMP directory alone was
    validated by hand on four real device tests (Google Photos still rendered both
    HDR and motion). If some other app turns out to need it, the fallback is to
    binary-patch the MPF offset here rather than to re-serialize the block.
    """
    items = [ContainerItem(semantic=Semantic.PRIMARY, length=0, mime="image/jpeg")]
    if gain_map is not None:
        items.append(ContainerItem(semantic=Semantic.GAIN_MAP, length=len(gain_map), mime="image/jpeg"))
    if video is not None:
        items.append(ContainerItem(semantic=Semantic.MOTION_PHOTO, length=len(video), mime="video/mp4"))

    xmp_text = extract_xmp_text(primary)
    if xmp_text:
        primary = replace_standard_xmp(primary, set_container_directory(xmp_text, items))

    out = bytearray(primary)
    if gain_map is not None:
        out += gain_map
    if video is not None:
        out += video
    return bytes(out)
