"""US-025: Remove all pyredner/redner imports from the codebase."""
import subprocess
import importlib
import os
from pathlib import Path


SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')


def test_us025_no_pyredner_imports():
    """grep -r 'pyredner|import redner' src/ returns no results."""
    result = subprocess.run(
        ['grep', '-r', '-E', 'pyredner|import redner', SRC_DIR],
        capture_output=True, text=True
    )
    assert result.returncode != 0, (
        f"Found pyredner/redner references:\n{result.stdout}"
    )


def test_us025_no_pyredner_in_pyproject():
    """pyredner is not listed in pyproject.toml dependencies."""
    pyproject = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
    with open(pyproject) as f:
        content = f.read()
    assert 'pyredner' not in content, "pyredner found in pyproject.toml"


def test_us025_renderer_mitsuba_importable():
    """The replacement renderer imports without pyredner."""
    mod = importlib.import_module('rendering.renderer')
    assert hasattr(mod, 'Renderer')


def test_us025_image_no_pyredner():
    """imaging/image.py has no pyredner references."""
    image_path = os.path.join(SRC_DIR, 'imaging', 'image.py')
    with open(image_path) as f:
        content = f.read()
    assert 'pyredner' not in content, "pyredner found in imaging/image.py"
