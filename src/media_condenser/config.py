"""Configuration models and the cascading loader.

Two config files share one schema:

- a global config (``~/.config/mcon/config.yaml``) holding machine-level settings
  (storage strategy, job count, tool paths) plus the default rules
- ``.mcon.yaml`` files dropped anywhere in a media tree, which work like
  ``.gitignore``: they apply to their own directory and everything below it,
  nearest ancestor winning per field

Only the ``rules`` section is overridable per directory. Machine-level settings
come from the global config alone -- a media directory has no business changing
where temp files go or how many cores to use.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Self

import pydantic
import yaml

from media_condenser import logging_setup

LOCAL_CONFIG_NAME = ".mcon.yaml"


class Strategy(StrEnum):
    """Where processed output goes."""

    COPY = "copy"
    """Write beside the original (or into a mirror tree). Never touches sources."""

    REPLACE = "replace"
    """Overwrite the original, via a same-filesystem temp file and atomic rename."""


class ImageRules(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    enabled: bool = True
    max_edge: int = pydantic.Field(default=2048, gt=0)
    """Target for the *longest* side. Shrink-only; smaller images are untouched."""

    quality: int = pydantic.Field(default=85, ge=1, le=100)
    subsampling: str = "4:2:0"

    skip_png: bool = True
    """Skip anything that really is a PNG. Downscaling screenshots and diagrams
    hurts text legibility for almost no size win."""

    skip_name_patterns: list[str] = pydantic.Field(default_factory=lambda: [])
    """Glob patterns matched against the basename, as a deliberate *opt-out*."""


class VideoRules(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    enabled: bool = True
    max_short_edge: int = pydantic.Field(default=720, gt=0)
    """Target for the *shortest* side, so portrait and landscape both land on 720p."""

    crf: int = pydantic.Field(default=23, ge=0, le=51)
    codec: str = "libx265"
    preset: str = "medium"
    tag: str = "hvc1"
    """libx265 tags output ``hev1`` by default; originals use ``hvc1`` and some
    players reject the former."""

    copy_audio: bool = True


class MotionPhotoRules(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    enabled: bool = True
    process_video: bool = True
    """Re-encode the embedded video through the same 720p path as standalone videos."""

    process_gain_map: bool = True
    """Rescale the Ultra HDR gain map alongside the primary image."""


class Rules(pydantic.BaseModel):
    """The overridable part of the config."""

    model_config = pydantic.ConfigDict(extra="forbid")

    images: ImageRules = pydantic.Field(default_factory=ImageRules)
    videos: VideoRules = pydantic.Field(default_factory=VideoRules)
    motion_photos: MotionPhotoRules = pydantic.Field(default_factory=MotionPhotoRules)
    exclude: list[str] = pydantic.Field(default_factory=list)
    """Glob patterns excluded from processing entirely, at any media type."""


class ToolPaths(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    exiftool: str = "exiftool"
    magick: str = "magick"


class GlobalConfig(pydantic.BaseModel):
    """Machine-level settings plus the root of the rules cascade."""

    model_config = pydantic.ConfigDict(extra="forbid")

    strategy: Strategy = Strategy.COPY
    """Defaults to non-destructive. ``replace`` is opt-in."""

    output_dir: Path | None = None
    """Mirror tree for the ``copy`` strategy, which has nowhere to write without it.

    Required by ``copy`` and refused by ``replace``, so exactly one of the two
    combinations is usable. Left optional here because a :class:`GlobalConfig` is
    also what carries tool paths and rules for probing and planning, neither of
    which writes anything; the requirement belongs to a *run* and is enforced where
    a run begins. See :func:`media_condenser.storage.plan_destination`.
    """

    jobs: int = pydantic.Field(default=4, gt=0)
    """Concurrent ffmpeg/exiftool jobs. Paired with per-job encoder thread pools:
    x265's own pool defaults to every core, so N jobs each grabbing all cores
    oversubscribes badly. 4 jobs x 4 threads was the working balance on a
    16-core machine."""

    encoder_pools: int | None = None
    """Threads per encoder job. Derived from ``cores // jobs`` when unset."""

    logging: dict[str, Any] = pydantic.Field(default_factory=logging_setup.default_logging_config)
    """A ``logging.config.dictConfig`` document, merged onto the packaged default.

    Kept as a raw dict rather than a schema of our own because it already has one:
    anything ``dictConfig`` accepts works here, including handlers this tool has
    never heard of. That is what lets a user add a rotating file handler without
    the tool needing a ``--log-file`` flag to offer them."""

    tools: ToolPaths = pydantic.Field(default_factory=ToolPaths)
    rules: Rules = pydantic.Field(default_factory=Rules)

    @pydantic.field_validator("logging", mode="before")
    @classmethod
    def _merge_logging(cls, value: Any) -> Any:
        """Layer the user's fragment onto the packaged config, per leaf key.

        Replacing wholesale would mean anyone who wants one logger at DEBUG has to
        restate every formatter and handler, and a fragment that omits them is not
        a valid dictConfig at all -- so the natural thing to write in a config file
        would fail. Merging is also what the reference implementation (b4_backup)
        gets from ``OmegaConf.merge``.

        Lists are replaced rather than merged by :func:`_deep_merge`, which is the
        behaviour ``handlers: [...]`` needs: naming handlers means naming *the* set.
        """
        if not isinstance(value, dict):
            return value
        return _deep_merge(logging_setup.default_logging_config(), value)

    @pydantic.model_validator(mode="after")
    def _derive_pools(self) -> Self:
        if self.encoder_pools is None:
            object.__setattr__(self, "encoder_pools", max(1, cpu_count() // self.jobs))
        return self

    @pydantic.model_validator(mode="after")
    def _reject_output_dir_with_replace(self) -> Self:
        """``replace`` writes in place, so an ``output_dir`` alongside it is a lie.

        The two settings ask for opposite things and ``replace`` wins silently:
        nothing is ever written to ``output_dir`` and the originals are overwritten
        instead. Someone who asked for output in a separate tree and got their library
        rewritten in place has to be told, and the only safe time to tell them is
        before anything runs.
        """
        if self.strategy is Strategy.REPLACE and self.output_dir is not None:
            raise ValueError(
                "output_dir cannot be combined with the replace strategy: replace overwrites "
                "each original in place, so nothing would be written to the output tree. "
                "Use the copy strategy for a mirror tree, or drop output_dir to replace in place."
            )
        return self


class LocalConfig(pydantic.BaseModel):
    """A ``.mcon.yaml``. Same rule schema as the global config, nothing else."""

    model_config = pydantic.ConfigDict(extra="forbid")

    rules: Rules = pydantic.Field(default_factory=Rules)


def default_global_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "mcon" / "config.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(loaded).__name__}")
    return loaded


def load_global_config(path: Path | None = None) -> GlobalConfig:
    """Load the global config, falling back to all-defaults when absent."""
    target = path or default_global_config_path()
    if not target.exists():
        if path is not None:
            raise FileNotFoundError(f"config file not found: {path}")
        return GlobalConfig()
    return GlobalConfig.model_validate(_read_yaml(target))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, per leaf key.

    Merging at the leaf rather than the section means a ``.mcon.yaml`` setting only
    ``images.quality`` keeps the inherited ``images.max_edge`` instead of resetting
    it to the schema default.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


class RulesResolver:
    """Resolves the effective :class:`Rules` for any directory.

    Walks from the scan root down to the directory, layering every ``.mcon.yaml``
    found along the way. Results are memoized per directory -- a library with
    thousands of files in a handful of directories parses each config once.
    """

    def __init__(self, root_rules: Rules, scan_roots: list[Path]) -> None:
        self._root_rules = root_rules
        self._scan_roots = [p.resolve() for p in scan_roots]
        self._cache: dict[Path, Rules] = {}

    def _ancestors_within_scan(self, directory: Path) -> list[Path]:
        """Directories from the containing scan root down to ``directory``.

        A ``.mcon.yaml`` above the scan root is intentionally ignored: the user
        pointed us at a subtree, so that is where the cascade begins.
        """
        for root in self._scan_roots:
            if directory == root or root in directory.parents:
                chain = [directory, *directory.parents]
                return chain[chain.index(root) :: -1]
        return [directory]

    def for_directory(self, directory: Path) -> Rules:
        # Checked before resolving, not after: `discovery.walk` calls this once per
        # *file*, and a realpath is an lstat per path component -- which over a
        # network share is the expensive part of reaching an answer already cached.
        if (cached := self._cache.get(directory)) is not None:
            return cached
        raw = directory
        directory = directory.resolve()
        if directory in self._cache:
            self._cache[raw] = self._cache[directory]
            return self._cache[directory]

        payload = self._root_rules.model_dump(mode="json")
        for ancestor in self._ancestors_within_scan(directory):
            candidate = ancestor / LOCAL_CONFIG_NAME
            if candidate.exists():
                local = LocalConfig.model_validate(_read_yaml(candidate))
                payload = _deep_merge(payload, local.rules.model_dump(mode="json", exclude_unset=True))

        resolved = Rules.model_validate(payload)
        self._cache[directory] = resolved
        self._cache[raw] = resolved
        return resolved


@lru_cache(maxsize=1)
def cpu_count() -> int:
    return os.cpu_count() or 4
