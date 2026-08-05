"""Walk tests: what gets in, what stays out, and in what order.

The walk lists the tree on one thread and then examines what it found across a pool,
because on a network share the per-path stats are most of its wall clock. That split
is why the bookkeeping tests here matter: the deduplication and the name filters run
in the listing thread, and everything after them has to hold without shared state.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import fixtures
import pytest

from media_condenser import config, discovery, probe, storage


def resolver(*roots: Path) -> config.RulesResolver:
    return config.RulesResolver(config.Rules(), list(roots))


def photo(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fixtures.IMAGES / fixtures.IMAGE_LANDSCAPE, target)
    return target


def names(candidates: list[discovery.Candidate]) -> list[str]:
    return [c.path.name for c in candidates]


def test_overlapping_roots_yield_each_file_once(tmp_path: Path) -> None:
    """A file reachable from two roots is one file, not two.

    Processing it twice under ``replace`` means re-encoding an already-encoded output,
    and under ``copy`` means two workers racing for the same destination path.
    """
    photo(tmp_path / "sub" / "a.jpg")
    photo(tmp_path / "sub" / "b.jpg")

    found = discovery.walk([tmp_path, tmp_path / "sub"], resolver(tmp_path))
    assert names(found) == ["a.jpg", "b.jpg"]


def test_the_same_root_twice_yields_each_file_once(tmp_path: Path) -> None:
    photo(tmp_path / "a.jpg")
    assert names(discovery.walk([tmp_path, tmp_path], resolver(tmp_path))) == ["a.jpg"]


def test_results_are_in_sorted_path_order(tmp_path: Path) -> None:
    """Deterministic regardless of which worker finishes first.

    Two runs over an unchanged tree should produce comparable logs; an order that
    depends on thread scheduling makes every diff between them noise.
    """
    for name in ("d.jpg", "b.jpg", "a.jpg", "c.jpg"):
        photo(tmp_path / name)
    photo(tmp_path / "sub" / "e.jpg")

    found = discovery.walk([tmp_path], resolver(tmp_path))
    assert [c.path for c in found] == sorted(c.path for c in found)
    assert names(found) == ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]


def test_roots_are_processed_in_the_order_they_were_named(tmp_path: Path) -> None:
    """Sorting is per root, so the argument order survives.

    Sorting across all of them instead would silently reorder ``mcon b a`` into ``a b``.
    """
    photo(tmp_path / "alpha" / "z.jpg")
    photo(tmp_path / "beta" / "a.jpg")

    forwards = discovery.walk([tmp_path / "beta", tmp_path / "alpha"], resolver(tmp_path))
    assert names(forwards) == ["a.jpg", "z.jpg"]
    backwards = discovery.walk([tmp_path / "alpha", tmp_path / "beta"], resolver(tmp_path))
    assert names(backwards) == ["z.jpg", "a.jpg"]


def test_symlinks_are_never_ingested(tmp_path: Path) -> None:
    """Following one would process the target twice, or reach outside the scan root."""
    photo(tmp_path / "real.jpg")
    (tmp_path / "link.jpg").symlink_to(tmp_path / "real.jpg")
    outside = photo(tmp_path.parent / f"{tmp_path.name}_outside.jpg")
    (tmp_path / "escape.jpg").symlink_to(outside)

    assert names(discovery.walk([tmp_path], resolver(tmp_path))) == ["real.jpg"]


def test_directories_are_not_candidates(tmp_path: Path) -> None:
    """The listing no longer filters them out, so the examination has to.

    A directory named like a media file is the case that would slip through: the name
    tests pass, and only the ``is_file`` check rejects it.
    """
    (tmp_path / "album.jpg").mkdir()
    photo(tmp_path / "real.jpg")
    assert names(discovery.walk([tmp_path], resolver(tmp_path))) == ["real.jpg"]


def test_skipped_directories_are_not_descended(tmp_path: Path) -> None:
    for directory in discovery._SKIP_DIRS:
        photo(tmp_path / directory / "hidden.jpg")
    photo(tmp_path / "visible.jpg")

    assert names(discovery.walk([tmp_path], resolver(tmp_path))) == ["visible.jpg"]


def test_local_config_is_not_itself_media(tmp_path: Path) -> None:
    (tmp_path / config.LOCAL_CONFIG_NAME).write_text("rules: {}\n")
    photo(tmp_path / "a.jpg")
    assert names(discovery.walk([tmp_path], resolver(tmp_path))) == ["a.jpg"]


def test_temp_files_are_not_ingested(tmp_path: Path) -> None:
    photo(tmp_path / f"{storage.TMP_PREFIX}999_a.jpg")
    photo(tmp_path / "a.jpg")
    assert names(discovery.walk([tmp_path], resolver(tmp_path))) == ["a.jpg"]


def test_a_file_named_directly_is_walked(tmp_path: Path) -> None:
    """Naming files rather than a directory is a supported way to invoke the tool."""
    target = photo(tmp_path / "a.jpg")
    photo(tmp_path / "b.jpg")
    assert names(discovery.walk([target], resolver(tmp_path))) == ["a.jpg"]


def test_an_empty_tree_yields_nothing(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert discovery.walk([tmp_path], resolver(tmp_path)) == []


# -- rules resolved per directory --------------------------------------


def test_local_rules_apply_to_the_directory_they_are_in(tmp_path: Path) -> None:
    """Resolved before the pool starts, so every worker reads a filled cache.

    Letting the workers fill it concurrently would have several of them parse the same
    ``.mcon.yaml``; the results agree either way, but the parse is wasted.
    """
    photo(tmp_path / "top.jpg")
    photo(tmp_path / "sub" / "nested.jpg")
    (tmp_path / "sub" / config.LOCAL_CONFIG_NAME).write_text("rules:\n  images:\n    max_edge: 512\n")

    found = {c.path.name: c for c in discovery.walk([tmp_path], resolver(tmp_path))}
    assert found["nested.jpg"].rules.images.max_edge == 512
    assert found["top.jpg"].rules.images.max_edge == config.ImageRules().max_edge


def test_excluded_names_are_dropped(tmp_path: Path) -> None:
    photo(tmp_path / "keep.jpg")
    photo(tmp_path / "drop.jpg")
    (tmp_path / config.LOCAL_CONFIG_NAME).write_text('rules:\n  exclude: ["drop.*"]\n')

    assert names(discovery.walk([tmp_path], config.RulesResolver(config.Rules(), [tmp_path]))) == ["keep.jpg"]


# -- measured, not guessed ---------------------------------------------


def test_kind_and_size_come_from_the_file(tmp_path: Path) -> None:
    target = photo(tmp_path / "mislabeled.png")
    found = discovery.walk([tmp_path], resolver(tmp_path))
    assert len(found) == 1
    assert found[0].kind == probe.JPEG
    assert found[0].size == target.stat().st_size


@pytest.mark.parametrize("count", [1, 200])
def test_every_file_in_a_tree_is_found(tmp_path: Path, count: int) -> None:
    """A pool that drops or duplicates work would show up as a miscount here."""
    for index in range(count):
        photo(tmp_path / f"sub{index % 7}" / f"img{index:04d}.jpg")

    found = discovery.walk([tmp_path], resolver(tmp_path))
    assert len(found) == count
    assert len({c.path for c in found}) == count
