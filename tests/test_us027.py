"""US-027: Final validation and documentation."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

NEXTFACE_DIR = Path(__file__).resolve().parent.parent / "NextFace"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_us027_all_stories_pass():
    """All 26 previous stories should have passes: true in prd.json."""
    prd_path = PROJECT_ROOT / "prd.json"
    prd = json.loads(prd_path.read_text())
    failing = []
    for story in prd["userStories"]:
        if story["id"] == "US-027":
            continue  # skip self
        if not story["passes"]:
            failing.append(story["id"])
    assert failing == [], f"Stories still failing: {failing}"


def test_us027_no_todos_in_renderer():
    """renderer_mitsuba.py should have no TODO or FIXME comments."""
    renderer_path = NEXTFACE_DIR / "renderer_mitsuba.py"
    content = renderer_path.read_text()
    todos = re.findall(r"(?i)\b(TODO|FIXME)\b", content)
    assert todos == [], f"Found TODO/FIXME in renderer_mitsuba.py: {todos}"


def test_us027_renderer_importable():
    """python -c 'from renderer_mitsuba import Renderer' should succeed."""
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, 'NextFace'); from renderer_mitsuba import Renderer; print('OK')"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=30,
    )
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "OK" in result.stdout


def test_us027_no_pyredner_in_codebase():
    """No pyredner or redner imports should remain in NextFace/."""
    for py_file in NEXTFACE_DIR.glob("*.py"):
        if py_file.name == "renderer.py":
            continue  # original file kept as reference
        content = py_file.read_text()
        assert "import pyredner" not in content, f"pyredner import in {py_file.name}"
        assert "import redner" not in content, f"redner import in {py_file.name}"
        assert "from pyredner" not in content, f"pyredner from-import in {py_file.name}"


def test_us027_porting_plan_updated():
    """PORTING_PLAN.md should reflect completion."""
    plan_path = PROJECT_ROOT / "PORTING_PLAN.md"
    content = plan_path.read_text()
    assert "PORT COMPLETE" in content or "COMPLETE" in content
