"""Names of the generated fixtures, by the behaviour each one exercises.

Tests refer to these constants rather than filenames so that regenerating or
renaming the library is a one-file change.

The media itself is generated and therefore not committed. ``make_fixtures.py`` is
both the builder and the documentation: what each fixture is for is written inline
next to the call that creates it. ``conftest.py`` refuses to run the suite if the media
is missing, rather than letting every test fail with a confusing "file not found".
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from media_condenser import container

ROOT = Path(__file__).resolve().parent / "data"
IMAGES = ROOT / "images"
VIDEOS = ROOT / "videos"
MOTION = ROOT / "motion_photos"

# -- videos ----------------------------------------------------------------
#: Portrait stored as landscape pixels, -90 display matrix, H.265, has audio.
VIDEO_ROTATED_MINUS90 = "PXL_20200101_100000000.mp4"
#: The same trick with the opposite rotation sign. Also the only filename that
#: matches no date pattern.
VIDEO_ROTATED_PLUS90 = "IMG_5174.MP4"
#: No audio stream at all.
VIDEO_NO_AUDIO = "PXL_20200102_100000000.mp4"
#: Tilde "duplicate" name that is genuinely full resolution.
VIDEO_TILDE_H264 = "PXL_20200103_100000000~2.mp4"
#: The only clip carrying GPS coordinates.
VIDEO_WITH_GPS = "DJI_20200104_120000_1_null_video.mp4"
#: Square 768x768.
VIDEO_SQUARE = "VID-20200105-WA0001.mp4"
#: Already below the 720p target.
VIDEO_ALREADY_SMALL = "VID_20200106_120000_000.mp4"

ALL_VIDEOS = [
    VIDEO_ROTATED_MINUS90,
    VIDEO_ROTATED_PLUS90,
    VIDEO_NO_AUDIO,
    VIDEO_TILDE_H264,
    VIDEO_WITH_GPS,
    VIDEO_SQUARE,
    VIDEO_ALREADY_SMALL,
]

# -- images ----------------------------------------------------------------
IMAGE_LANDSCAPE = "PXL_20200107_120000000.jpg"
IMAGE_PORTRAIT = "PXL_20200108_120000000.jpg"
#: Already below the 2048 px target.
IMAGE_ALREADY_SMALL = "IMG_20200109_120000_000.jpg"
#: A genuine PNG, deliberately over the size target.
SCREENSHOT_PNG = "Screenshot_20200110-120000-bearbeitet.png"
#: JPEG bytes behind a .png name.
SCREENSHOT_MISLABELED = "Screenshot_20200111-120000~2.png"

RESIZED_IMAGES = [IMAGE_LANDSCAPE, IMAGE_PORTRAIT]
SKIPPED_IMAGES = [IMAGE_ALREADY_SMALL, SCREENSHOT_PNG, SCREENSHOT_MISLABELED]

# -- motion photos ---------------------------------------------------------
#: Landscape, video only, no gain map.
MP_VIDEO_ONLY_LANDSCAPE = "PXL_20200112_120000000.MP.jpg"
#: Portrait, video only; the embedded clip is itself rotated.
MP_VIDEO_ONLY_PORTRAIT = "PXL_20200113_120000000.MP.jpg"
#: Gain map only, rdf:Bag form, plus orphaned trailing bytes.
MP_GAINMAP_ORPHANED = "PXL_20200114_120000000.MP-bearbeitet.jpg"
#: Embedded clip already within target, so it must pass through untouched.
MP_SMALL_EMBEDDED_VIDEO = "PXL_20200115_120000000.MP.jpg"
#: Declared video length exceeds the whole file.
MP_IMPOSSIBLE_VIDEO_LEN = "PXL_20200116_120000000.MP.jpg"
#: rdf:Bag form with the gain map ending exactly at EOF.
MP_BAG_FORM = "PXL_20200117_120000000.MP-bearbeitet.jpg"
#: Portrait primary + gain map + rotated embedded video.
MP_FULL_COMBO = "PXL_20200118_120000000.MP.jpg"

ALL_MOTION_PHOTOS = [
    MP_VIDEO_ONLY_LANDSCAPE,
    MP_VIDEO_ONLY_PORTRAIT,
    MP_GAINMAP_ORPHANED,
    MP_SMALL_EMBEDDED_VIDEO,
    MP_IMPOSSIBLE_VIDEO_LEN,
    MP_BAG_FORM,
    MP_FULL_COMBO,
]

#: Fixtures that arrive with a gain map, which must therefore leave with one.
MOTION_PHOTOS_WITH_GAIN_MAP = [
    MP_GAINMAP_ORPHANED,
    MP_SMALL_EMBEDDED_VIDEO,
    MP_IMPOSSIBLE_VIDEO_LEN,
    MP_BAG_FORM,
    MP_FULL_COMBO,
]

#: Fixtures whose container is deliberately damaged, so they must be reported as
#: downgraded rather than failed.
MOTION_PHOTOS_DOWNGRADED = [MP_GAINMAP_ORPHANED, MP_IMPOSSIBLE_VIDEO_LEN]

#: Total number of media files in the library.
TOTAL_FILES = len(ALL_VIDEOS) + len(RESIZED_IMAGES) + len(SKIPPED_IMAGES) + len(ALL_MOTION_PHOTOS)

# -- expected geometry -----------------------------------------------------
#: Source clips are 16:9 at 1600x900, so they land on exactly these.
VIDEO_TARGET_LANDSCAPE = (1280, 720)
VIDEO_TARGET_PORTRAIT = (720, 1280)
VIDEO_TARGET_SQUARE = (720, 720)
VIDEO_STORED_SIZE = (1600, 900)
VIDEO_DISPLAY_ROTATED = (900, 1600)
VIDEO_SQUARE_SIZE = (768, 768)

#: Images and motion photo primaries are 16:9, so they land on exactly these.
IMAGE_SIZE_LANDSCAPE = (2560, 1440)
IMAGE_SIZE_PORTRAIT = (1440, 2560)
IMAGE_TARGET_LANDSCAPE = (2048, 1152)
IMAGE_TARGET_PORTRAIT = (1152, 2048)


def mpf_tags(path: Path) -> tuple[int | None, int | None]:
    """``MPImageStart``/``MPImageLength``, or ``(None, None)`` when absent."""
    out = subprocess.run(
        ["exiftool", "-json", "-n", "-MPImageStart", "-MPImageLength", str(path)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    data = json.loads(out)[0] if out.strip() else {}
    return data.get("MPImageStart"), data.get("MPImageLength")


def layout_of(path: Path):
    """Resolve a motion photo's layout the way the tool does."""
    offset, length = mpf_tags(path)
    return container.resolve_layout(path, path.read_bytes(), mpf_offset=offset, mpf_length=length)


def gain_map_bytes_of(path: Path) -> bytes | None:
    """The gain map sliced out of a fixture, for comparing against an output.

    Tests compare against this rather than a hardcoded byte count, so regenerating
    the library cannot silently invalidate an assertion.
    """
    layout = layout_of(path)
    if layout.gain_map is None:
        return None
    data = path.read_bytes()
    return data[layout.gain_map.offset : layout.gain_map.end]
