"""Walking media trees and attaching the effective rules to each file."""

from __future__ import annotations

import fnmatch
import logging
import os
import stat
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path

from media_condenser import config, probe, storage

log = logging.getLogger(__name__)

#: Directories skipped wholesale during the walk.
_SKIP_DIRS = {".git", ".svn", "__pycache__", ".mcon_tmp", "node_modules", ".Trash", "@eaDir"}

#: Threads used to examine found paths. Oversubscribed relative to the core count on
#: purpose: each one spends its time blocked on a stat or a 32-byte read, not
#: computing, and the trees this walks are routinely on a NAS where the per-call
#: latency rather than the local syscall cost is what dominates.
_WALK_WORKERS = min(32, config.cpu_count() * 4)


@dataclass(frozen=True)
class Candidate:
    """A file found by the walk, with its measured kind and effective rules."""

    path: Path
    kind: str
    rules: config.Rules
    size: int

    @property
    def name(self) -> str:
        return self.path.name


def matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def walk(roots: list[Path], resolver: config.RulesResolver) -> list[Candidate]:
    """Collect every media file under ``roots``.

    Type identification is by magic bytes, not extension. The single place a
    filename influences anything is the user-configured exclusion patterns, which
    are an opt-out rather than a classification.

    Every recognised media file is returned, including the types this tool cannot
    process -- the planner is what decides they are unsupported. Dropping them here
    instead would make them invisible: no record, no row in the summary, and no copy
    into an ``--output-dir`` mirror. A user reading "0 failed" over a tree that
    silently omitted every HEIC has been told the mirror is complete when it is not.

    The two halves are split deliberately. Listing the tree is inherently sequential,
    but examining what it found is a few blocking syscalls per path and nothing else,
    so that part runs across a thread pool -- which is most of the walk's wall clock on
    a network share. The results keep sorted path order regardless of which thread
    finishes first, because a reproducible order is what makes two runs' logs
    comparable.
    """
    found: list[Path] = []
    seen: set[Path] = set()

    def remember(path: Path) -> None:
        # The name-only tests live here, on one thread, because they cost nothing and
        # each path they reject saves a thread dispatch and several syscalls.
        if path in seen or path.name == config.LOCAL_CONFIG_NAME:
            return
        # This tool's own half-written output. It carries real JPEG/MP4 magic bytes, so
        # without this it would be ingested as library media: resized and committed
        # under its temp name, or copied into the mirror tree as junk. `_SKIP_DIRS`
        # does not cover it -- the temp files are siblings of the destination, not a
        # directory.
        if path.name.startswith(storage.TMP_PREFIX):
            return
        seen.add(path)
        found.append(path)

    for root in roots:
        resolved = root.resolve()
        start = len(found)
        if resolved.is_file():
            remember(resolved)
        else:
            # `os.walk` rather than `rglob`, so `_SKIP_DIRS` prunes the traversal
            # instead of filtering it afterwards. A Synology `@eaDir` holds a
            # thumbnail directory per media file, so enumerating one only to discard
            # every path in it can cost more than the real tree does.
            for parent, dirnames, filenames in os.walk(resolved):
                dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
                base = Path(parent)
                for name in filenames:
                    remember(base / name)
        # Sorted within each root rather than across all of them, so naming several
        # roots processes them in the order they were named. `rglob` itself yields in
        # directory order, which is arbitrary.
        found[start:] = sorted(found[start:])

    if not found:
        return []

    # Resolved up front, on this thread, for two reasons. `RulesResolver` memoizes into
    # a plain dict that concurrent callers would race to fill, re-parsing the same
    # `.mcon.yaml` several times over; and a library of thousands of files spans only a
    # handful of directories, so this is a few reads rather than one per file.
    for directory in {path.parent for path in found}:
        resolver.for_directory(directory)

    with futures.ThreadPoolExecutor(max_workers=_WALK_WORKERS, thread_name_prefix="mcon-walk") as pool:
        examined = pool.map(lambda path: _consider(path, resolver), found)
        return [candidate for candidate in examined if candidate is not None]


def _consider(path: Path, resolver: config.RulesResolver) -> Candidate | None:
    """Examine one found path. Runs on a worker thread, so it keeps no shared state."""
    # One lstat answers all three questions -- symlink, regular file, size -- where
    # `is_symlink()`/`is_file()`/`stat()` were three round-trips per path. On the
    # network shares this walks, that is the difference that shows.
    try:
        info = os.lstat(path)
    except OSError as exc:
        log.debug("skipping unreadable %s", path, exc_info=exc)
        return None
    if not stat.S_ISREG(info.st_mode):
        # Directories, and symlinks of every sort: a link is skipped rather than
        # followed so a tree that links to itself cannot be processed twice.
        return None

    rules = resolver.for_directory(path.parent)
    if matches_any(path.name, rules.exclude):
        return None

    kind = probe.sniff_kind(path)
    if kind == probe.UNKNOWN:
        return None

    return Candidate(path=path, kind=kind, rules=rules, size=info.st_size)
