"""Config cascade and storage safety tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from media_condenser import cli, config, discovery, pipeline, planner, probe, storage

# ---------------------------------------------------------------------------
# Cascading .mcon.yaml
# ---------------------------------------------------------------------------


def write_local(directory: Path, payload: dict) -> None:
    (directory / ".mcon.yaml").write_text(yaml.safe_dump(payload))


def test_rules_are_inherited_by_subdirectories(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    write_local(tmp_path / "a", {"rules": {"images": {"max_edge": 1024}}})

    resolver = config.RulesResolver(config.Rules(), [tmp_path])
    assert resolver.for_directory(nested).images.max_edge == 1024


def test_nearest_ancestor_wins(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    write_local(tmp_path / "a", {"rules": {"images": {"max_edge": 1024}}})
    write_local(deep, {"rules": {"images": {"max_edge": 512}}})

    resolver = config.RulesResolver(config.Rules(), [tmp_path])
    assert resolver.for_directory(deep).images.max_edge == 512
    assert resolver.for_directory(tmp_path / "a").images.max_edge == 1024


def test_partial_override_keeps_sibling_fields(tmp_path: Path) -> None:
    """Setting one field must not reset the rest of its section to defaults."""
    directory = tmp_path / "a"
    directory.mkdir()
    write_local(directory, {"rules": {"images": {"quality": 60}}})

    resolver = config.RulesResolver(config.Rules(), [tmp_path])
    rules = resolver.for_directory(directory)
    assert rules.images.quality == 60
    assert rules.images.max_edge == 2048  # inherited, not reset


def test_config_above_the_scan_root_is_ignored(tmp_path: Path) -> None:
    """Pointing the tool at a subtree starts the cascade there."""
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    write_local(outer, {"rules": {"images": {"max_edge": 1024}}})

    resolver = config.RulesResolver(config.Rules(), [inner])
    assert resolver.for_directory(inner).images.max_edge == 2048


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    """A typo in a config file should be loud, not silently ignored."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"strategy": "copy", "jobz": 8}))

    with pytest.raises(ValueError):
        config.load_global_config(config_file)


def test_encoder_pools_are_derived_from_jobs() -> None:
    """jobs x pools must stay within the core count to avoid oversubscription."""
    cores = os.cpu_count() or 4
    cfg = config.GlobalConfig(jobs=4)
    assert cfg.encoder_pools == max(1, cores // 4)
    assert cfg.encoder_pools * 4 <= max(4, cores)


def test_explicit_encoder_pools_are_respected() -> None:
    assert config.GlobalConfig(jobs=4, encoder_pools=2).encoder_pools == 2


def test_output_dir_with_replace_is_rejected() -> None:
    """The two settings ask for opposite things, and replace used to win silently.

    ``plan_destination`` short-circuits on ``replace`` before it ever looks at
    ``output_dir``, so someone who asked for their output in a separate tree got
    nothing there and their only copies overwritten in place instead.
    """
    with pytest.raises(ValueError, match="output_dir cannot be combined"):
        config.GlobalConfig(strategy=config.Strategy.REPLACE, output_dir=Path("/tmp/out"))


def test_output_dir_with_replace_is_rejected_from_a_config_file(tmp_path: Path) -> None:

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"strategy": "replace", "output_dir": str(tmp_path / "out")}))
    with pytest.raises(ValueError, match="output_dir cannot be combined"):
        config.load_global_config(config_file)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_temp_file_lands_beside_the_destination(tmp_path: Path) -> None:
    """The cross-device rename failure that broke 1,747 files in one run.

    ``os.replace`` is atomic only within a filesystem, so the temp file has to share
    the destination's directory -- never ``/tmp``.
    """
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"x")
    destination = storage.plan_destination(source, config.GlobalConfig(strategy=config.Strategy.REPLACE))

    # The property that matters is not "avoids /tmp" -- it is that the temp file and
    # the destination share a directory, and therefore a filesystem, wherever the
    # library happens to live.
    assert destination.tmp.parent == destination.final.parent
    assert destination.tmp.parent.stat().st_dev == destination.final.parent.stat().st_dev
    storage.assert_same_filesystem(destination)  # must not raise


def test_cross_filesystem_temp_is_rejected_before_encoding(tmp_path: Path, monkeypatch) -> None:
    """The check must fire up front, not after an expensive encode.

    Simulates the destination and temp directories reporting different devices, which
    is what a ``/tmp`` temp file plus a mounted media library looks like.
    """
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"x")
    destination = storage.plan_destination(source, config.GlobalConfig(strategy=config.Strategy.REPLACE))

    real_stat = Path.stat
    calls = {"n": 0}

    def fake_stat(self, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 2:  # the temp directory's stat
            fields = tuple(result)
            return os.stat_result((*fields[:2], result.st_dev + 1, *fields[3:]))
        return result

    monkeypatch.setattr(Path, "stat", fake_stat)
    with pytest.raises(storage.StorageError, match="cross-device"):
        storage.assert_same_filesystem(destination)


def test_copy_strategy_never_targets_the_source(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"x")
    cfg = config.GlobalConfig(strategy=config.Strategy.COPY, output_dir=tmp_path / "out")
    destination = storage.plan_destination(source, cfg)
    assert destination.final != source
    assert destination.final.is_relative_to(tmp_path / "out")


def test_copy_strategy_without_an_output_dir_is_refused(tmp_path: Path) -> None:
    """The old fallback wrote ``photo_compressed.jpg`` beside the original.

    That interleaved two copies of an entire library under a name no gallery or
    sync tool understands, and it happened by default -- a plain ``mcon ~/Photos``
    got it. Refusing is the only outcome that cannot quietly make a mess, so the
    combination is a hard error rather than a guess.
    """
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"x")
    with pytest.raises(storage.StorageError, match="needs an output_dir"):
        storage.plan_destination(source, config.GlobalConfig(strategy=config.Strategy.COPY))


def test_replace_strategy_targets_the_source(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"x")
    destination = storage.plan_destination(source, config.GlobalConfig(strategy=config.Strategy.REPLACE))
    assert destination.replaces_source


def test_output_dir_mirrors_the_tree(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    (root / "2026" / "05").mkdir(parents=True)
    source = root / "2026" / "05" / "photo.jpg"
    source.write_bytes(b"x")
    out = tmp_path / "out"

    destination = storage.plan_destination(
        source, config.GlobalConfig(strategy=config.Strategy.COPY, output_dir=out), scan_root=root
    )
    assert destination.final == out / "2026" / "05" / "photo.jpg"


def test_several_scan_roots_keep_their_subtrees_in_the_mirror(tmp_path: Path) -> None:
    """Same-named files under different roots must not land on one path.

    With no root to anchor against, every file's relative path collapses to its
    basename: ``2020/IMG_0001.jpg`` and ``2021/IMG_0001.jpg`` both commit to
    ``out/IMG_0001.jpg``, the second rename destroys the first, and both are reported
    as succeeded. Anchoring on the roots' common ancestor keeps them distinct.
    """

    lib = tmp_path / "lib"
    for year in ("2020", "2021"):
        (lib / year).mkdir(parents=True)
        (lib / year / "IMG_0001.jpg").write_bytes(b"x")

    roots = [lib / "2020", lib / "2021"]
    scan_root = cli._mirror_root(roots)
    assert scan_root == lib.resolve()

    cfg = config.GlobalConfig(strategy=config.Strategy.COPY, output_dir=tmp_path / "out")
    finals = {storage.plan_destination(root / "IMG_0001.jpg", cfg, scan_root).final for root in roots}
    assert finals == {
        (tmp_path / "out" / "2020" / "IMG_0001.jpg").resolve(),
        (tmp_path / "out" / "2021" / "IMG_0001.jpg").resolve(),
    }


def test_a_single_root_and_named_files_mirror_as_before(tmp_path: Path) -> None:
    """One directory anchors on itself; named files anchor on their own parent."""
    lib = tmp_path / "lib"
    (lib / "sub").mkdir(parents=True)
    photo = lib / "sub" / "photo.jpg"
    photo.write_bytes(b"x")

    assert cli._mirror_root([lib]) == lib.resolve()
    assert cli._mirror_root([photo]) == photo.parent.resolve()


def test_two_inputs_claiming_one_output_are_refused(tmp_path: Path) -> None:
    """Refusing both is what keeps a silent overwrite from being reported as success."""
    actions = []
    for year in ("2020", "2021"):
        directory = tmp_path / year
        directory.mkdir()
        path = directory / "IMG_0001.jpg"
        path.write_bytes(b"x")
        candidate = discovery.Candidate(path=path, kind=probe.JPEG, rules=config.Rules(), size=1)
        actions.append(planner.Action(candidate=candidate, kind=planner.Kind.IMAGE, verb=planner.Verb.RESIZE_IMAGE))

    cfg = config.GlobalConfig(strategy=config.Strategy.COPY, output_dir=tmp_path / "out")
    # scan_root=None is the flattening case: both files map to out/IMG_0001.jpg.
    assert len(pipeline._colliding_destinations(actions, cfg, None)) == 2
    assert pipeline._colliding_destinations(actions, cfg, tmp_path) == []


def test_commit_is_atomic_and_preserves_mtime(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"original")
    os.utime(source, (1_600_000_000, 1_600_000_000))

    destination = storage.plan_destination(source, config.GlobalConfig(strategy=config.Strategy.REPLACE))
    storage.prepare(destination)
    destination.tmp.write_bytes(b"new-and-smaller")
    size = storage.commit(destination)

    assert size == len(b"new-and-smaller")
    assert source.read_bytes() == b"new-and-smaller"
    assert not destination.tmp.exists()
    assert int(source.stat().st_mtime) == 1_600_000_000


def test_empty_output_is_never_committed(tmp_path: Path) -> None:
    """An encoder that produced nothing must not destroy the original."""
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"original")
    destination = storage.plan_destination(source, config.GlobalConfig(strategy=config.Strategy.REPLACE))
    storage.prepare(destination)
    destination.tmp.write_bytes(b"")

    with pytest.raises(storage.StorageError):
        storage.commit(destination)
    assert source.read_bytes() == b"original"


def test_discard_leaves_the_source_untouched(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"original")
    destination = storage.plan_destination(source, config.GlobalConfig(strategy=config.Strategy.REPLACE))
    storage.prepare(destination)
    destination.tmp.write_bytes(b"partial garbage")

    storage.discard(destination)
    assert not destination.tmp.exists()
    assert source.read_bytes() == b"original"


def test_stale_temp_files_can_be_cleaned_up(tmp_path: Path) -> None:
    leftover = tmp_path / f".mcon_tmp_{os.getpid()}_photo.jpg"
    leftover.write_bytes(b"interrupted")
    keep = tmp_path / "photo.jpg"
    keep.write_bytes(b"real")

    assert storage.cleanup_stale(tmp_path) == 1
    assert not leftover.exists()
    assert keep.exists()


def test_cleanup_leaves_another_live_runs_temp_files_alone(tmp_path: Path) -> None:
    """Two runs over one library must not delete each other's work in progress.

    Deleting a temp file another process is still writing turns its working run into a
    spray of "nothing to commit" failures, so a live pid means hands off. pid 1 stands
    in for "some other process that is definitely running".
    """
    in_flight = tmp_path / ".mcon_tmp_1_photo.jpg"
    in_flight.write_bytes(b"another run is writing this")
    nameless = tmp_path / ".mcon_tmp_photo.jpg"
    nameless.write_bytes(b"no pid to check")

    assert storage.cleanup_stale(tmp_path) == 1
    assert in_flight.exists()
    assert not nameless.exists()
