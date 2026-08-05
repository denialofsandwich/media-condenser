"""Test session setup.

The fixture media under ``tests/data/`` is generated rather than committed, so a
fresh clone has none of it. Without a guard, every test that touches a file fails
with an unrelated-looking ``FileNotFoundError`` and the actual problem -- one command
was never run -- is buried. Fail once, up front, saying exactly what to do.
"""

from __future__ import annotations

import fixtures
import pytest

_GENERATE = "uv run python tests/make_fixtures.py"


def pytest_collection_modifyitems(config, items) -> None:
    """Abort collection if the generated fixture media is missing or incomplete.

    Checks the actual expected filenames rather than just the directories, so a
    partial or interrupted generation is caught too.
    """
    if not items:
        return

    if not fixtures.ROOT.exists():
        pytest.exit(
            f"fixture media not found at {fixtures.ROOT}.\nIt is generated, not committed -- run:\n    {_GENERATE}",
            returncode=4,
        )

    expected = [
        *((fixtures.IMAGES / name) for name in (*fixtures.RESIZED_IMAGES, *fixtures.SKIPPED_IMAGES)),
        *((fixtures.VIDEOS / name) for name in fixtures.ALL_VIDEOS),
        *((fixtures.MOTION / name) for name in fixtures.ALL_MOTION_PHOTOS),
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        listed = "\n    ".join(str(path.relative_to(fixtures.ROOT.parent)) for path in missing[:5])
        remainder = f"\n    ... and {len(missing) - 5} more" if len(missing) > 5 else ""
        pytest.exit(
            f"fixture media is incomplete ({len(missing)} of {len(expected)} files missing):\n"
            f"    {listed}{remainder}\n"
            f"Regenerate with:\n    {_GENERATE}",
            returncode=4,
        )
