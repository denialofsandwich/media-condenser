"""Deciding what to do with each file, from measured properties only.

This module is pure and does no I/O beyond probing: it turns a
:class:`~media_condenser.discovery.Candidate` into an :class:`Action`. Keeping the decision
separate from the execution is what makes ``--dry-run`` an honest preview rather
than a partial run, and it puts the whole skip-or-process rule in one readable
place.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from media_condenser import container, discovery, probe


class Kind(StrEnum):
    """Reporting category. Motion photos are their own group, not a kind of image."""

    IMAGE = "image"
    VIDEO = "video"
    MOTION_PHOTO = "motion photo"
    OTHER = "other"


#: The sniffed kind each reporting category is reached from. The one place the
#: mapping is stated: :func:`prefetch` picks a probe from it, :func:`_plan_inner`
#: picks a planner, and both stay in step with :func:`_kind_guess` by construction.
_KIND_OF_SNIFF = {
    probe.VIDEO: Kind.VIDEO,
    probe.JPEG: Kind.IMAGE,
    probe.PNG: Kind.IMAGE,
}


class Verb(StrEnum):
    SKIP = "skip"
    RESIZE_IMAGE = "resize image"
    TRANSCODE_VIDEO = "transcode video"
    REBUILD_MOTION_PHOTO = "rebuild motion photo"
    FAIL = "fail"


@dataclass
class Action:
    """What to do with one file, and why."""

    candidate: discovery.Candidate
    kind: Kind
    verb: Verb
    reason: str = ""
    target_size: tuple[int, int] | None = None
    """Intended output dimensions, for the dry-run table. ``None`` when unchanged
    or when ffmpeg computes the exact even-numbered result itself."""

    image_info: probe.ImageInfo | None = None
    video_info: probe.VideoInfo | None = None
    layout: container.MotionPhotoLayout | None = None
    clear_motion_container: bool = False
    """Set when the source declares motion photo components that the output will not
    contain, so the declarations have to be deleted rather than copied across."""

    notes: list[str] = field(default_factory=list)
    """Reasons a component was degraded. Non-empty means the result belongs in the
    'downgraded' bucket, which is reported separately from failures."""

    @property
    def path(self) -> Path:
        return self.candidate.path

    @property
    def will_process(self) -> bool:
        return self.verb not in (Verb.SKIP, Verb.FAIL)


async def prefetch(
    candidates: Sequence[discovery.Candidate],
    prober: probe.Prober,
    *,
    concurrency: int,
    on_progress: Callable[[int], None] | None = None,
) -> None:
    """Warm the probe caches for everything :func:`plan` is about to classify.

    Purely an optimisation: :func:`plan` is unchanged by it and produces the same
    actions either way, only without a subprocess launch of its own per file. See
    :meth:`media_condenser.probe.Prober.prefetch` for the measured effect.

    The split lives here rather than in :mod:`media_condenser.probe` because which tool answers
    for which kind is a *planning* decision. It reads :data:`_KIND_OF_SNIFF`, the
    same mapping :func:`_plan_inner` dispatches on, so the prefetch cannot drift
    into probing the wrong half of a library.
    """
    exif: list[Path] = []
    video: list[Path] = []
    unprobed = 0
    for candidate in candidates:
        match _kind_guess(candidate):
            case Kind.VIDEO:
                video.append(candidate.path)
            case Kind.IMAGE:
                exif.append(candidate.path)
            case _:
                # HEIF and unrecognised ISO brands are classified as unsupported
                # without being probed at all. Counted so the caller's progress total
                # can stay the honest one -- every candidate -- rather than only the
                # probed ones.
                unprobed += 1

    if on_progress and unprobed:
        on_progress(unprobed)
    await prober.prefetch(exif=exif, video=video, concurrency=concurrency, on_progress=on_progress)


def plan(candidate: discovery.Candidate, prober: probe.Prober) -> Action:
    """Classify one candidate."""
    try:
        return _plan_inner(candidate, prober)
    except probe.ProbeError as exc:
        return Action(
            candidate=candidate,
            kind=_kind_guess(candidate),
            verb=Verb.FAIL,
            reason=str(exc),
        )


def _kind_guess(candidate: discovery.Candidate) -> Kind:
    """The reporting category a candidate's sniffed kind belongs to.

    A motion photo still reads as :attr:`Kind.IMAGE` here -- telling one from a
    plain JPEG means looking inside, which only :func:`_plan_image_like` does.
    """
    return _KIND_OF_SNIFF.get(candidate.kind, Kind.OTHER)


def _plan_inner(candidate: discovery.Candidate, prober: probe.Prober) -> Action:
    match _kind_guess(candidate):
        case Kind.VIDEO:
            return _plan_video(candidate, prober)
        case Kind.IMAGE:
            return _plan_image_like(candidate, prober)
        case _:
            # Named rather than generic: "unsupported type (heif)" is what tells a
            # user reading the summary that their HEIC library was recognised and
            # left alone, rather than leaving them to wonder whether it was seen.
            return Action(
                candidate=candidate,
                kind=Kind.OTHER,
                verb=Verb.SKIP,
                reason=f"unsupported type ({candidate.kind})",
            )


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


def _plan_video(candidate: discovery.Candidate, prober: probe.Prober) -> Action:
    rules = candidate.rules.videos
    if not rules.enabled:
        return Action(candidate=candidate, kind=Kind.VIDEO, verb=Verb.SKIP, reason="videos disabled")

    info = prober.video_info(candidate.path)

    # Gate on the *displayed* short edge, i.e. after autorotation. Deciding from
    # the stored height would skip portrait files stored as landscape pixels.
    if info.short_edge <= rules.max_short_edge:
        return Action(
            candidate=candidate,
            kind=Kind.VIDEO,
            verb=Verb.SKIP,
            reason=f"already {info.display_size[0]}x{info.display_size[1]} (short edge <= {rules.max_short_edge})",
            video_info=info,
        )

    return Action(
        candidate=candidate,
        kind=Kind.VIDEO,
        verb=Verb.TRANSCODE_VIDEO,
        reason=f"{info.display_size[0]}x{info.display_size[1]} -> short edge {rules.max_short_edge}",
        target_size=_video_target(info.display_size, rules.max_short_edge),
        video_info=info,
    )


def _video_target(display: tuple[int, int], max_short_edge: int) -> tuple[int, int]:
    """Predicted output size, for display purposes.

    ffmpeg computes the real value (rounded to even), so this is an estimate used
    only in the dry-run table.
    """
    width, height = display
    if width >= height:
        scale = max_short_edge / height
        return _even(round(width * scale)), max_short_edge
    scale = max_short_edge / width
    return max_short_edge, _even(round(height * scale))


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


# ---------------------------------------------------------------------------
# Images and motion photos
# ---------------------------------------------------------------------------


def _plan_image_like(candidate: discovery.Candidate, prober: probe.Prober) -> Action:
    """Route a JPEG/PNG to the image or motion photo path.

    A motion photo is a JPEG, so this has to look inside before deciding.
    """
    rules = candidate.rules.images

    # Exclusions first -- cheapest, and they apply regardless of what is inside.
    if rules.skip_png and candidate.kind == probe.PNG:
        return Action(
            candidate=candidate,
            kind=Kind.IMAGE,
            verb=Verb.SKIP,
            reason="PNG (downscaling costs legibility for little gain)",
        )
    if discovery.matches_any(candidate.name, rules.skip_name_patterns):
        return Action(
            candidate=candidate,
            kind=Kind.IMAGE,
            verb=Verb.SKIP,
            reason="matches an excluded name pattern",
        )
    if not rules.enabled:
        return Action(candidate=candidate, kind=Kind.IMAGE, verb=Verb.SKIP, reason="images disabled")

    # The motion photo flag is read whether or not motion photos are enabled. With
    # them disabled the file is still resized as a plain image, and the resize drops
    # the appended components -- so the output must not go out carrying the source's
    # declaration of them. Knowing it *was* a motion photo is what allows that.
    is_mp = candidate.kind == probe.JPEG and container.is_motion_photo(prober.xmp_packet(candidate.path))
    if is_mp and candidate.rules.motion_photos.enabled:
        return _plan_motion_photo(candidate, prober)

    notes = ["motion photo processing disabled; embedded components dropped"] if is_mp else None
    return _plan_plain_image(candidate, prober, notes=notes, was_motion_photo=is_mp)


def _plan_plain_image(
    candidate: discovery.Candidate,
    prober: probe.Prober,
    *,
    notes: list[str] | None = None,
    was_motion_photo: bool = False,
) -> Action:
    rules = candidate.rules.images
    info = prober.image_info(candidate.path)

    if info.long_edge <= rules.max_edge:
        # Untouched output: whatever the file declares is still true of it.
        return Action(
            candidate=candidate,
            kind=Kind.IMAGE,
            verb=Verb.SKIP,
            reason=f"already {info.width}x{info.height} (long edge <= {rules.max_edge})",
            image_info=info,
            notes=list(notes or []),
        )

    return Action(
        candidate=candidate,
        kind=Kind.IMAGE,
        verb=Verb.RESIZE_IMAGE,
        reason=f"{info.width}x{info.height} -> long edge {rules.max_edge}",
        target_size=_image_target((info.width, info.height), rules.max_edge),
        image_info=info,
        clear_motion_container=was_motion_photo,
        notes=list(notes or []),
    )


def _image_target(size: tuple[int, int], max_edge: int) -> tuple[int, int]:
    """Predicted output size. Both callers are past their own long-edge gate, so
    this is only ever asked about an image that really is being shrunk."""
    width, height = size
    scale = max_edge / max(size)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _plan_motion_photo(candidate: discovery.Candidate, prober: probe.Prober) -> Action:
    rules = candidate.rules.images
    info = prober.image_info(candidate.path)

    # Gate on the *primary image's* resolution. A container-level property would
    # be the wrong thing to measure here.
    #
    # This comes first because resolving the container means reading the whole file,
    # and a phone library is mostly motion photos that are already small enough to
    # leave alone -- so that read was megabytes apiece to reach a decision the
    # already-cached dimensions had made. Nothing downstream misses the layout: only
    # `Verb.REBUILD_MOTION_PHOTO` reads `Action.layout`, and a skip never gets there.
    # What is given up is `layout.notes`, which are observations about a container
    # this run is not going to touch.
    if info.long_edge <= rules.max_edge:
        return Action(
            candidate=candidate,
            kind=Kind.MOTION_PHOTO,
            verb=Verb.SKIP,
            reason=f"primary already {info.width}x{info.height} (long edge <= {rules.max_edge})",
            image_info=info,
        )

    data = candidate.path.read_bytes()
    exif = prober.exif(candidate.path)
    layout = container.resolve_layout(
        candidate.path,
        data,
        mpf_offset=probe.as_int(exif.get("MPImageStart")),
        mpf_length=probe.as_int(exif.get("MPImageLength")),
    )

    # No usable attached data: the flag is real but the components are not. Fall
    # back to a plain image resize and record it as a downgrade, not a failure.
    #
    # The container declaration has to go with it. The resize path restores metadata
    # with `-tagsFromFile <original> -all:all`, which would copy the original
    # Container:Directory -- non-zero GainMap and MotionPhoto lengths included -- onto
    # a file that now holds nothing but the primary image. That is the exact hazard
    # `container.assemble` documents: a reader trusting the declaration slices the
    # bytes that happen to follow and hands back garbage as a video. Only
    # `assemble()` rewrites the directory, and this path never reaches it.
    if not layout.has_attachments:
        notes = ["motion photo flag set but no usable attached components; treated as a plain image"]
        notes.extend(layout.notes)
        return _plan_plain_image(candidate, prober, notes=notes, was_motion_photo=True)

    components = ["primary"]
    if layout.gain_map is not None:
        components.append("gain map")
    if layout.video is not None:
        components.append("video")

    return Action(
        candidate=candidate,
        kind=Kind.MOTION_PHOTO,
        verb=Verb.REBUILD_MOTION_PHOTO,
        reason=f"{info.width}x{info.height} -> long edge {rules.max_edge} ({' + '.join(components)})",
        target_size=_image_target((info.width, info.height), rules.max_edge),
        image_info=info,
        layout=layout,
        notes=list(layout.notes),
    )
