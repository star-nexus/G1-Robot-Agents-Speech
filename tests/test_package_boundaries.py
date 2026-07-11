from __future__ import annotations

import ast
from pathlib import Path


def test_standalone_package_does_not_import_legacy_features():
    root = Path(__file__).parents[1] / "src" / "g1_speech"
    forbidden = (
        "moyun",
        "PyQt5",
        "faster_whisper",
        "requests",
        "pynput",
    )
    imports = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
    for name in imports:
        assert not name.startswith(forbidden), f"forbidden legacy import: {name}"
