#!/usr/bin/env python3
"""Generate the test fixture library from scratch.

    uv run python tests/make_fixtures.py [--out tests/data]

Every fixture is synthesised from ffmpeg/ImageMagick test patterns, so the tree
contains no photographic content, no real timestamps and no location data. It is
also tiny: the whole library is a couple of megabytes, against ~340 MB when the
same cases were covered by real camera files.

**Why generated rather than downloaded.** The fixtures have to exhibit specific
byte-level container defects -- a declared video length larger than the file, 2.3 MB
of orphaned trailing data, a gain map declared in an alternate RDF serialization.
No public sample has those properties; they only occur in files a particular camera
or editor happened to damage. They must be constructed either way, so constructing
all of it keeps the library reproducible and license-free.

**What is faithful to real files.** The structural facts were measured from real
device output before it was removed, and are reproduced exactly:

- both XMP container serializations (``rdf:Seq`` with attribute-style ``Item:*``,
  and ``rdf:Bag`` with element-style ``Item:*``)
- a real MPF (APP2) block, hand-assembled, because the orphaned-trailing-bytes case
  is only handled correctly when the absolute MPF offset beats end-subtraction
- ``hdrgm:*`` gain map parameters on the gain map images
- container-level display-matrix rotation, of both signs
- videos with and without an audio stream

**Sizing.** Sources are deliberately encoded at higher quality than the tool's own
targets, so that the 720p CRF-23 re-encode and the 2048 px q85 resize genuinely come
out smaller. Otherwise the tool would correctly refuse to write a larger "compressed"
file, and every geometry assertion downstream would fail on a file that was skipped.
``gradients``/``gradient:`` patterns were chosen after measuring alternatives: flat
patterns like ``smptebars`` are smaller but *grow* when re-encoded, while noisy ones
like ``testsrc2`` shrink well but cost hundreds of KB per clip.

**Synthetic identity.** Metadata is stamped with ``Make=MCT``, ``Model=MCT TestCam``,
dates in January 2020, and GPS at the Royal Observatory Greenwich -- a public landmark,
chosen so the coordinate-preservation test has something real-shaped to assert on while
revealing nothing about anyone.

**Expected outcomes**, for a quick sanity check of a run over this library:

- The 4 already-conforming files come out **untouched**: the 296x640 clip, the
  1280x1280 image, and both screenshots. With ``--output-dir`` they are copied across
  byte-for-byte so the mirror tree is complete; they never reach an encoder.
- The 2 damaged containers and the impossible-length file must **not fail** -- they
  process via a fallback and are reported as *downgraded*.
- Every gain map survives **byte-identical**, since gain maps are passed through
  verbatim unless they exceed the target.
- Every output still reports its original ``CreateDate`` and camera model, and the
  drone clip still reports its GPS coordinates.
- A second run over the output reports 100% skipped.

Each fixture is documented inline in :func:`build`, next to the call that creates it.
:func:`verify` then asserts the properties the tests depend on, because the tools this
script drives fail quietly -- exiftool once declined a GPS spelling, printed "Nothing
to do" and exited 0, shipping a fixture with no coordinates and leaving the GPS
regression test asserting on nothing.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from media_condenser import config, container, probe  # noqa: E402

NS_HDRGM = "http://ns.adobe.com/hdr-gain-map/1.0/"
NS_RDF_URI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

#: Synthetic camera identity. Written so the metadata-preservation tests have
#: something to assert on without naming a real device or owner.
CAMERA_MAKE = "MCT"
CAMERA_MODEL = "MCT TestCam"

#: Royal Observatory Greenwich -- a public landmark, used so the GPS-preservation
#: regression test has coordinates that reveal nothing about anyone.
GPS_LAT = 51.4779
GPS_LON = -0.0015

_APP1 = 0xE1
_APP2 = 0xE2


def run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(argv, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"{argv[0]} failed: {result.stderr.strip()[-500:]}")
    return result


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def make_image(
    path: Path,
    width: int,
    height: int,
    *,
    gradient: str,
    quality: int = 90,
    png: bool = False,
    exif: bool = True,
    date: str = "2020:01:01 12:00:00",
) -> None:
    """Render a gradient image, optionally stamping synthetic camera metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / ("x.png" if png else "x.jpg")
        argv = ["magick", "-size", f"{width}x{height}", f"gradient:{gradient}"]
        if not png:
            argv += ["-quality", str(quality), "-sampling-factor", "4:4:4"]
        argv.append(str(raw))
        run(argv)
        shutil.copy2(raw, path)

    if exif:
        run(
            [
                "exiftool",
                "-q",
                "-overwrite_original",
                f"-Make={CAMERA_MAKE}",
                f"-Model={CAMERA_MODEL}",
                f"-CreateDate={date}",
                f"-DateTimeOriginal={date}",
                str(path),
            ]
        )


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------


def make_video(
    path: Path,
    width: int,
    height: int,
    *,
    rotation: int = 0,
    codec: str = "libx264",
    audio: bool = True,
    fps: int = 6,
    duration: float = 1.0,
    crf: int = 20,
    gps: bool = False,
    android_tags: bool = True,
    date: str = "2020:01:01 12:00:00",
) -> None:
    """Encode a short gradient clip.

    ``rotation`` writes a container-level display matrix via ``-display_rotation``,
    which is what makes a portrait clip get *stored* as landscape pixels -- the case
    a naive ``scale=-2:720`` gets wrong.
    """
    with tempfile.TemporaryDirectory() as tmp:
        plain = Path(tmp) / "plain.mp4"
        # fmt: off
        argv = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"gradients=size={width}x{height}:rate={fps}:n=4",
        ]
        if audio:
            argv += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100"]
        argv += ["-t", str(duration), "-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p"]
        # fmt: on
        if audio:
            argv += ["-c:a", "aac", "-b:a", "32k", "-shortest"]
        if codec == "libx265":
            argv += ["-tag:v", "hvc1", "-x265-params", "log-level=error"]
        argv.append(str(plain))
        run(argv)

        if rotation:
            rotated = Path(tmp) / "rot.mp4"
            # fmt: off
            run([
                "ffmpeg", "-y", "-v", "error",
                "-display_rotation", str(rotation),
                "-i", str(plain), "-c", "copy", str(rotated),
            ])
            # fmt: on
            plain = rotated

        shutil.copy2(plain, path)

    tags = ["-q", "-overwrite_original", f"-CreateDate={date}"]
    if android_tags:
        # Keys-atom metadata, standing in for the com.android.* tags that ffmpeg
        # silently drops whenever a video filter is applied.
        tags += [f"-Keys:AndroidModel={CAMERA_MODEL}", "-Keys:AndroidVersion=16"]
    if gps:
        # Comma-separated, and explicitly in the UserData group. The obvious
        # space-separated spelling is silently ignored: exiftool prints "Nothing to
        # do" and still exits 0, so the fixture would ship without coordinates and
        # the GPS regression test would be asserting on nothing. verify() catches it.
        tags += [f"-UserData:GPSCoordinates={GPS_LAT}, {GPS_LON}"]
    run(["exiftool", *tags, str(path)])


def video_bytes(width: int, height: int, **kwargs) -> bytes:
    """A clip as raw bytes, for embedding inside a motion photo."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "clip.mp4"
        make_video(target, width, height, android_tags=False, **kwargs)
        return target.read_bytes()


# ---------------------------------------------------------------------------
# Gain map
# ---------------------------------------------------------------------------


def gain_map_bytes(width: int, height: int) -> bytes:
    """An Ultra HDR gain map: a small grayscale JPEG carrying ``hdrgm:*`` params.

    The parameters matter -- a renderer needs them to apply the map -- so the
    fixtures carry real ones and the tests can assert they survive.
    """
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "gm.jpg"
        # fmt: off
        run([
            "magick", "-size", f"{width}x{height}",
            "gradient:gray10-gray90", "-colorspace", "Gray",
            "-quality", "90", str(target),
        ])
        # fmt: on
        run(
            [
                "exiftool",
                "-q",
                "-overwrite_original",
                "-XMP-hdrgm:Version=1.0",
                "-XMP-hdrgm:GainMapMin=0.0",
                "-XMP-hdrgm:GainMapMax=0.283056",
                "-XMP-hdrgm:HDRCapacityMin=0.0",
                "-XMP-hdrgm:HDRCapacityMax=0.283056",
                "-XMP-hdrgm:OffsetSDR=0.0",
                "-XMP-hdrgm:OffsetHDR=0.0",
                str(target),
            ]
        )
        return target.read_bytes()


# ---------------------------------------------------------------------------
# XMP container directory, in both real-world serializations
# ---------------------------------------------------------------------------


@dataclass
class Item:
    semantic: str
    mime: str
    length: int = 0


def _directory_seq(items: list[Item]) -> str:
    """``rdf:Seq`` + ``Container:Item`` with the fields as attributes.

    What Pixel camera output uses, and what 5 of the 7 real motion photos had.
    """
    lines = ["      <Container:Directory>", "        <rdf:Seq>"]
    for item in items:
        attrs = f'Item:Mime="{item.mime}" Item:Semantic="{item.semantic}"'
        if item.length:
            attrs += f' Item:Length="{item.length}"'
        lines += [
            '          <rdf:li rdf:parseType="Resource">',
            f"            <Container:Item {attrs}/>",
            "          </rdf:li>",
        ]
    lines += ["        </rdf:Seq>", "      </Container:Directory>"]
    return "\n".join(lines)


def _directory_bag(items: list[Item]) -> str:
    """``rdf:Bag`` + ``rdf:Description`` with the fields as child elements.

    The serialization both ``-bearbeitet`` fixtures used. ExifTool flattens this to
    ``DirectoryLength``/``DirectorySemantic`` rather than ``DirectoryItemLength``/
    ``DirectoryItemSemantic``, which is what made a complete container look empty and
    nearly cost a working HDR gain map.
    """
    lines = ["      <Container:Directory>", "        <rdf:Bag>"]
    for item in items:
        lines += [
            "          <rdf:li>",
            "            <rdf:Description>",
            f"              <Item:Length>{item.length}</Item:Length>",
            f"              <Item:Mime>{item.mime}</Item:Mime>",
            "              <Item:Padding>0</Item:Padding>",
            f"              <Item:Semantic>{item.semantic}</Item:Semantic>",
            "            </rdf:Description>",
            "          </rdf:li>",
        ]
    lines += ["        </rdf:Bag>", "      </Container:Directory>"]
    return "\n".join(lines)


def build_xmp(items: list[Item], *, form: str, hdr: bool) -> str:
    directory = _directory_bag(items) if form == "bag" else _directory_seq(items)
    hdr_attr = '\n      hdrgm:Version="1.0"' if hdr else ""
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="MCT fixture generator">\n'
        f'  <rdf:RDF xmlns:rdf="{NS_RDF_URI}">\n'
        '    <rdf:Description rdf:about=""\n'
        f'        xmlns:GCamera="{container.NS_CAMERA}"\n'
        f'        xmlns:Container="{container.NS_CONTAINER}"\n'
        f'        xmlns:Item="{container.NS_ITEM}"\n'
        f'        xmlns:hdrgm="{NS_HDRGM}"'
        f"{hdr_attr}\n"
        '      GCamera:MotionPhoto="1"\n'
        '      GCamera:MotionPhotoVersion="1"\n'
        '      GCamera:MotionPhotoPresentationTimestampUs="500000">\n'
        f"{directory}\n"
        "    </rdf:Description>\n"
        "  </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    )


# ---------------------------------------------------------------------------
# JPEG segment surgery
# ---------------------------------------------------------------------------

_XMP_STD_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


def set_xmp_packet(data: bytes, packet: str) -> bytes:
    """Replace the standard XMP APP1 payload with ``packet``."""
    segment = container.find_standard_xmp(data)
    if segment is None:
        raise RuntimeError("no standard XMP segment to replace; write an XMP tag first")
    payload = _XMP_STD_HEADER + packet.encode("utf-8")
    seg_length = len(payload) + 2
    return data[: segment.offset] + bytes([0xFF, _APP1]) + seg_length.to_bytes(2, "big") + payload + data[segment.end :]


def _last_app_end(data: bytes) -> int:
    """Byte offset just past the final APPn segment, where MPF can be inserted."""
    end = 2
    for segment in container.iter_segments(data):
        if 0xE0 <= segment.marker <= 0xEF:
            end = segment.end
    return end


def build_mpf_segment(image_sizes: list[int], offsets: list[int]) -> bytes:
    """Assemble an MPF (APP2) block declaring ``len(image_sizes)`` images.

    MPF duplicates the gain map's position as an *absolute* offset, which is exactly
    why it is worth reproducing: one fixture carries orphaned trailing data, and
    locating the gain map by subtracting declared lengths from the end of the file
    lands inside that junk. Only the absolute offset gets it right.

    Offsets are stored relative to the start of the MPF TIFF header. ``offsets[0]``
    is 0 by spec -- the primary image is the file itself.
    """
    count = len(image_sizes)
    entries = b""
    for index, (size, offset) in enumerate(zip(image_sizes, offsets, strict=True)):
        # Attribute word: representative + "Baseline MP Primary Image" for the first
        # entry, undefined type for the rest (matching what real files carried).
        attribute = 0x20000003 if index == 0 else 0x00000000
        entries += (
            attribute.to_bytes(4, "little")
            + size.to_bytes(4, "little")
            + offset.to_bytes(4, "little")
            + (0).to_bytes(2, "little")
            + (0).to_bytes(2, "little")
        )

    # MP Index IFD: 3 entries, then the next-IFD pointer, then the MPEntry blob.
    ifd_offset = 8
    ifd_size = 2 + 3 * 12 + 4
    entries_offset = ifd_offset + ifd_size

    ifd = (3).to_bytes(2, "little")
    ifd += b"\x00\xb0" + (7).to_bytes(2, "little") + (4).to_bytes(4, "little") + b"0100"
    ifd += b"\x01\xb0" + (4).to_bytes(2, "little") + (1).to_bytes(4, "little") + count.to_bytes(4, "little")
    ifd += (
        b"\x02\xb0"
        + (7).to_bytes(2, "little")
        + (16 * count).to_bytes(4, "little")
        + entries_offset.to_bytes(4, "little")
    )
    ifd += (0).to_bytes(4, "little")

    tiff = b"II" + (0x2A).to_bytes(2, "little") + ifd_offset.to_bytes(4, "little") + ifd + entries
    payload = b"MPF\x00" + tiff
    return bytes([0xFF, _APP2]) + (len(payload) + 2).to_bytes(2, "big") + payload


def mpf_entry_patch(data: bytes, size: int, absolute_offset: int) -> bytes:
    """Rewrite the second MPF entry's size and offset once the layout is known.

    Inserting the MPF block shifts every later byte, so the offsets it must record
    cannot be known until after insertion. The block is therefore written at its
    final size with zeroed values, and patched here -- byte count unchanged, so the
    layout stays put.
    """
    marker = b"\xff\xe2"
    index = data.find(marker + b"\x00")
    while index != -1:
        seg_len = int.from_bytes(data[index + 2 : index + 4], "big")
        payload = data[index + 4 : index + 2 + seg_len]
        if payload.startswith(b"MPF\x00"):
            tiff_base = index + 4 + 4
            entries_at = tiff_base + 8 + (2 + 3 * 12 + 4)
            second = entries_at + 16
            patched = bytearray(data)
            patched[second + 4 : second + 8] = size.to_bytes(4, "little")
            patched[second + 8 : second + 12] = (absolute_offset - tiff_base).to_bytes(4, "little")
            return bytes(patched)
        index = data.find(marker, index + 1)
    raise RuntimeError("no MPF segment found to patch")


# ---------------------------------------------------------------------------
# Motion photos
# ---------------------------------------------------------------------------


def make_motion_photo(
    path: Path,
    *,
    width: int,
    height: int,
    gradient: str,
    form: str = "seq",
    gain_map: bytes | None = None,
    video: bytes | None = None,
    declared_video_length: int | None = None,
    orphan: bytes = b"",
    date: str = "2020:01:01 12:00:00",
) -> None:
    """Assemble a motion photo container.

    ``declared_video_length`` overrides the length written into the XMP without
    changing the bytes appended, which is how the "declared length exceeds the file"
    fixture is produced. ``orphan`` appends unreferenced trailing bytes.
    """
    # 1. Primary image with synthetic camera metadata, plus a throwaway XMP tag so
    #    that a standard XMP packet exists for us to overwrite.
    make_image(path, width, height, gradient=gradient, quality=92, date=date)
    run(
        [
            "exiftool",
            "-q",
            "-overwrite_original",
            "-XMP-GCamera:MotionPhoto=1",
            "-XMP-GCamera:MotionPhotoVersion=1",
            str(path),
        ]
    )

    items = [Item("Primary", "image/jpeg", 0)]
    if gain_map is not None:
        items.append(Item("GainMap", "image/jpeg", len(gain_map)))
    if video is not None or declared_video_length is not None:
        declared = declared_video_length if declared_video_length is not None else len(video or b"")
        items.append(Item("MotionPhoto", "video/mp4", declared))

    data = set_xmp_packet(path.read_bytes(), build_xmp(items, form=form, hdr=gain_map is not None))

    # 2. Insert the MPF block only when there is a gain map to point at -- the
    #    video-only fixtures had no MPF at all, and that difference is load-bearing:
    #    it forces the resolver's end-subtraction fallback to be exercised too.
    if gain_map is not None:
        insert_at = _last_app_end(data)
        placeholder = build_mpf_segment([0, 0], [0, 0])
        data = data[:insert_at] + placeholder + data[insert_at:]
        data = mpf_entry_patch(data, len(gain_map), len(data))

    out = bytearray(data)
    if gain_map is not None:
        out += gain_map
    if video is not None:
        out += video
    out += orphan
    path.write_bytes(bytes(out))


# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------


def build(out: Path) -> None:
    for name in ("images", "videos", "motion_photos"):
        target = out / name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

    images, videos, motion = out / "images", out / "videos", out / "motion_photos"

    # =================================================================
    # images/
    #
    # 2560x1440 and 1440x2560 are both 16:9, so they resize to exactly 2048x1152
    # and 1152x2048 -- the same expected outputs the real 4032x2268 files produced,
    # at 2.5x fewer pixels.
    # =================================================================
    print("images...")

    # Baseline landscape: the ordinary resize path.
    make_image(images / "PXL_20200107_120000000.jpg", 2560, 1440, gradient="navy-orange")

    # Baseline portrait: same path, other orientation. Proves aspect ratio is kept
    # rather than the long edge being applied to a fixed axis.
    make_image(images / "PXL_20200108_120000000.jpg", 1440, 2560, gradient="darkgreen-yellow")

    # Already inside the 2048 px target: must be skipped, and never upscaled.
    make_image(
        images / "IMG_20200109_120000_000.jpg",
        1280,
        1280,
        gradient="purple-white",
        date="2020:01:09 12:00:00",
    )

    # A genuine PNG. Deliberately *over* the size target, so the PNG exclusion is the
    # only rule that can skip it. The real fixture was 1440x1045 -- under the target --
    # so the size rule was skipping it and the PNG rule was never actually exercised.
    make_image(
        images / "Screenshot_20200110-120000-bearbeitet.png",
        2560,
        1440,
        gradient="gray20-gray80",
        png=True,
        exif=False,
    )

    # JPEG bytes behind a .png name, also over the target. Content sniffing alone
    # would resize this, which is why the screenshot exclusion matches a real-PNG
    # check *or* a configurable name pattern. The `~2` suffix is incidental: it also
    # confirms a "duplicate-looking" name is not treated as one.
    make_image(
        images / "Screenshot_20200111-120000~2.png",
        1440,
        2560,
        gradient="black-skyblue",
        exif=False,
    )

    # =================================================================
    # videos/
    #
    # 1600x900 scales to exactly 1280x720; rotated, to exactly 720x1280.
    # =================================================================
    print("videos...")

    # Portrait stored as landscape pixels with a -90 display matrix. A plain
    # `scale=-2:720` scales the wrong axis on these; the filter has to test the
    # *decoded* aspect ratio. H.265 with audio.
    make_video(videos / "PXL_20200101_100000000.mp4", 1600, 900, rotation=-90, codec="libx265")

    # The same trick with the opposite rotation sign, as iPhone-style files carry.
    # ExifTool and ffprobe disagree on the sign here (270 vs 90), which is why nothing
    # hand-rolls the transform and ffmpeg's autorotate is trusted instead.
    # Also the one filename matching no date pattern, so inference must decline.
    make_video(videos / "IMG_5174.MP4", 1600, 900, rotation=90, codec="libx264")

    # No audio stream at all, as timelapses have. Mapping a nonexistent audio stream
    # is a hard error, so the mapping needs a per-file existence check.
    make_video(videos / "PXL_20200102_100000000.mp4", 1600, 900, codec="libx265", audio=False)

    # A tilde-suffixed "duplicate" name that is genuinely full-resolution H.264.
    # Classification has to come from measured resolution and codec, not the name.
    make_video(videos / "PXL_20200103_100000000~2.mp4", 1600, 900, codec="libx264")

    # Non-Pixel source: a mixed-source library is normal, so nothing may assume
    # Pixel-native containers. The only clip carrying GPS, which `-map_metadata`
    # silently drops -- a loss only visible by diffing the full tag set.
    #
    # Its metadata date deliberately disagrees with its filename by 9 hours, modelling
    # the real drone file whose name was local time and whose metadata was UTC. That
    # mismatch is why metadata wins and filename inference is only a fallback:
    # trusting the name here would overwrite a correct timestamp with a wrong one.
    make_video(
        videos / "DJI_20200104_120000_1_null_video.mp4",
        1600,
        900,
        codec="libx264",
        fps=120,
        duration=0.25,
        gps=True,
        android_tags=False,
        date="2020:01:04 03:00:00",
    )

    # Square. The scale filter's `gt(iw,ih)`-false branch must yield 720x720 rather
    # than distorting the frame.
    make_video(videos / "VID-20200105-WA0001.mp4", 768, 768, codec="libx264", android_tags=False)

    # Already below the 720p target: must be skipped, never upscaled. This is the
    # fixture that keeps the zero-op guarantee honest for video.
    make_video(videos / "VID_20200106_120000_000.mp4", 296, 640, codec="libx264")

    # =================================================================
    # motion_photos/
    #
    # Primaries are 16:9, so they resize to exactly 2048x1152 / 1152x2048.
    #
    # The two XMP serializations below are the single most important detail in the
    # library. Form A (rdf:Seq -> Container:Item, fields as attributes) is what the
    # camera writes and covers 5 fixtures. Form B (rdf:Bag -> rdf:Description, fields
    # as child elements) covers the two `-bearbeitet` fixtures. ExifTool flattens B to
    # `DirectoryLength`/`DirectorySemantic` instead of `DirectoryItemLength`/
    # `DirectoryItemSemantic`, so reading only the latter makes a complete container
    # look empty -- the misdiagnosis that would silently destroy a working HDR gain
    # map. A third form exists but is never read from a fixture: exiftool
    # re-serializes the same namespaces as `GContainer:`/`GItem:`, so files the tool
    # writes itself come back with different prefixes than the ones it reads.
    # =================================================================
    print("motion photos...")

    # Embedded clips. 1600x900 (short edge 900) is over the target and must be
    # re-encoded; 640x360 is already inside it and must be passed through byte-for-byte
    # rather than losing a generation of quality for no size win.
    big_clip = video_bytes(1600, 900, codec="libx265", audio=False)
    big_clip_rot = video_bytes(1600, 900, codec="libx265", audio=False, rotation=-90)
    small_clip = video_bytes(640, 360, codec="libx264", audio=False)
    gm_landscape = gain_map_bytes(400, 225)
    gm_portrait = gain_map_bytes(225, 400)

    # Landscape, video only, no gain map -- the simplest real case. Note this declares
    # exactly two items, as the gain-map-only fixture below also does: they mean
    # different things, which is why items are matched by label and never by position.
    # No MPF either, so the resolver's end-subtraction fallback gets exercised.
    make_motion_photo(
        motion / "PXL_20200112_120000000.MP.jpg",
        width=2560,
        height=1440,
        gradient="maroon-gold",
        video=big_clip,
    )

    # Portrait, video only. The embedded clip carries its own -90 display matrix, so it
    # needs the same orientation-aware filter as a standalone video -- the rotation
    # problem is not limited to top-level files.
    make_motion_photo(
        motion / "PXL_20200113_120000000.MP.jpg",
        width=1440,
        height=2560,
        gradient="indigo-pink",
        video=big_clip_rot,
    )

    # Gain map only, no video, form B, plus orphaned trailing MP4 data: a leftover blob
    # no longer referenced by anything. Locating the gain map by subtracting declared
    # lengths from EOF lands inside that junk; only the absolute MPF offset gets it
    # right. verify() asserts the orphan is big enough that end-subtraction really does
    # fail, so the fixture cannot quietly stop proving its point.
    #
    # The orphan only has to exceed the gain map's length, since end-subtraction
    # computes `size - gain_map_length`. Real files carried 2.3 MB; ~55 KB reproduces
    # the failure mode without the bulk.
    make_motion_photo(
        motion / "PXL_20200114_120000000.MP-bearbeitet.jpg",
        width=2560,
        height=1440,
        gradient="teal-white",
        form="bag",
        gain_map=gm_landscape,
        orphan=big_clip * 4,
    )

    # Full container whose embedded clip is *already* within target, so the video must
    # be passed through untouched while the primary is still resized.
    make_motion_photo(
        motion / "PXL_20200115_120000000.MP.jpg",
        width=2560,
        height=1440,
        gradient="darkred-cyan",
        gain_map=gm_landscape,
        video=small_clip,
    )

    # Declared video length larger than the entire file, with no video bytes present.
    # Must detect the impossible value, keep the valid gain map, and report a downgrade
    # rather than crashing or slicing garbage.
    make_motion_photo(
        motion / "PXL_20200116_120000000.MP.jpg",
        width=1440,
        height=2560,
        gradient="olive-lavender",
        gain_map=gm_portrait,
        declared_video_length=99_000_000,
    )

    # Form B again, this time with the gain map ending exactly at EOF and no orphan.
    # This is the shape that was misdiagnosed as a stale MotionPhoto flag: the
    # container is complete and the gain map works, so anything treating it as stale
    # throws away HDR with no error raised.
    make_motion_photo(
        motion / "PXL_20200117_120000000.MP-bearbeitet.jpg",
        width=1440,
        height=2560,
        gradient="brown-azure",
        form="bag",
        gain_map=gm_portrait,
    )

    # Everything at once: portrait primary + gain map + rotated embedded video.
    make_motion_photo(
        motion / "PXL_20200118_120000000.MP.jpg",
        width=1440,
        height=2560,
        gradient="black-white",
        gain_map=gm_portrait,
        video=big_clip_rot,
    )

    verify(out)

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\ndone: {total / 1024 / 1024:.1f} MB across {len(list(out.rglob('*.*')))} files")


def verify(out: Path) -> None:
    """Assert every property the fixtures are supposed to have.

    The generator does not get to assume its own tools worked. exiftool in particular
    will decline to write a tag it does not like, print "Nothing to do" and exit 0 --
    which once shipped a GPS fixture with no coordinates, leaving the regression test
    that depends on it quietly asserting nothing. Anything the tests rely on is
    checked here, at the point where a failure is obvious.
    """
    prober = probe.Prober(config.GlobalConfig().tools)
    problems: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    videos = out / "videos"
    gps_clip = videos / "DJI_20200104_120000_1_null_video.mp4"
    coords = run(["exiftool", "-s3", "-GPSCoordinates", str(gps_clip)]).stdout.strip()
    check(bool(coords), f"{gps_clip.name}: GPS coordinates were not written")

    rotations = {}
    for path in sorted(videos.iterdir()):
        info = prober.video_info(path)
        rotations[path.name] = info.rotation
        if path.name.startswith("PXL_20200102"):
            check(not info.has_audio, f"{path.name}: expected no audio stream")
        elif path.name.startswith("VID_20200106"):
            check(info.short_edge < 720, f"{path.name}: expected to be under the 720p target")
        else:
            check(info.short_edge > 720, f"{path.name}: expected to be over the 720p target")
    check(-90 in rotations.values(), "no video carries a -90 display matrix")
    check(90 in rotations.values(), "no video carries a +90 display matrix")

    forms: set[str] = set()
    gain_maps = videos_seen = 0
    for path in sorted((out / "motion_photos").iterdir()):
        data = path.read_bytes()
        xmp = container.extract_xmp_text(data)
        forms.add("bag" if "rdf:Bag" in xmp else "seq")
        check("GCamera:MotionPhoto" in xmp, f"{path.name}: missing the MotionPhoto flag")

        start, length = None, None
        exif = prober.exif(path)
        start, length = exif.get("MPImageStart"), exif.get("MPImageLength")
        layout = container.resolve_layout(path, data, mpf_offset=start, mpf_length=length)
        if layout.gain_map is not None:
            gain_maps += 1
            check(
                data[layout.gain_map.offset : layout.gain_map.offset + 3] == b"\xff\xd8\xff",
                f"{path.name}: gain map offset does not point at a JPEG",
            )
            check(start is not None, f"{path.name}: gain map present but no MPF offset")
        if layout.video is not None:
            videos_seen += 1
            check(
                data[layout.video.offset + 4 : layout.video.offset + 8] == b"ftyp",
                f"{path.name}: video offset does not point at an ISO container",
            )
    check(forms == {"bag", "seq"}, f"expected both RDF serializations, got {forms}")
    check(gain_maps >= 4, f"expected several gain maps, got {gain_maps}")
    check(videos_seen >= 3, f"expected several embedded videos, got {videos_seen}")

    # The orphan fixture is only meaningful if end-subtraction really would fail.
    orphan = out / "motion_photos" / "PXL_20200114_120000000.MP-bearbeitet.jpg"
    data = orphan.read_bytes()
    exif = prober.exif(orphan)
    with_mpf = container.resolve_layout(
        orphan, data, mpf_offset=exif.get("MPImageStart"), mpf_length=exif.get("MPImageLength")
    )
    without_mpf = container.resolve_layout(orphan, data, mpf_offset=None, mpf_length=None)
    check(with_mpf.gain_map is not None, f"{orphan.name}: gain map should resolve via MPF")
    check(
        without_mpf.gain_map is None,
        f"{orphan.name}: orphan is too small to defeat end-subtraction, so the fixture "
        "no longer proves the MPF offset is load-bearing",
    )

    images = out / "images"
    for name in ("Screenshot_20200110-120000-bearbeitet.png", "Screenshot_20200111-120000~2.png"):
        info = prober.image_info(images / name)
        check(
            info.long_edge > 2048,
            f"{name}: must exceed the size target so the exclusion rule is what skips it",
        )

    if problems:
        raise SystemExit("fixture verification failed:\n  " + "\n  ".join(problems))
    print("verified: all structural properties present")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
