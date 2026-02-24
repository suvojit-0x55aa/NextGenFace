"""US-027: Final validation and documentation."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_us027_no_todos_in_renderer():
    """rendering/renderer.py should have no TODO or FIXME comments."""
    renderer_path = SRC_DIR / "rendering" / "renderer.py"
    content = renderer_path.read_text()
    todos = re.findall(r"(?i)\b(TODO|FIXME)\b", content)
    assert todos == [], f"Found TODO/FIXME in rendering/renderer.py: {todos}"


def test_us027_renderer_importable():
    """python -c 'from rendering.renderer import Renderer' should succeed."""
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, 'src'); from rendering.renderer import Renderer; print('OK')"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=30,
    )
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "OK" in result.stdout


def test_us027_no_pyredner_in_codebase():
    """No pyredner or redner imports should remain in src/."""
    for py_file in SRC_DIR.rglob("*.py"):
        content = py_file.read_text()
        assert "import pyredner" not in content, f"pyredner import in {py_file}"
        assert "import redner" not in content, f"redner import in {py_file}"
        assert "from pyredner" not in content, f"pyredner from-import in {py_file}"
