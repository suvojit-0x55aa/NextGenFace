"""US-001: Verify UV project initialization with pyproject.toml."""

import subprocess
import sys
from pathlib import Path

import tomllib


ROOT = Path(__file__).resolve().parent.parent


def test_us001_pyproject_exists():
    """pyproject.toml exists at project root."""
    assert (ROOT / "pyproject.toml").is_file()


def test_us001_project_metadata():
    """pyproject.toml has required [project] metadata."""
    with open(ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    project = data["project"]
    assert "name" in project
    assert "version" in project
    assert ">=3.10" in project["requires-python"]


def test_us001_dependencies():
    """pyproject.toml lists all required dependencies."""
    with open(ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    deps = [d.split(">")[0].split("=")[0].strip() for d in data["project"]["dependencies"]]
    required = ["mitsuba", "torch", "numpy", "opencv-python", "h5py", "tqdm", "mediapipe"]
    for req in required:
        assert req in deps, f"Missing dependency: {req}"


def test_us001_dev_dependencies():
    """pyproject.toml has pytest and pytest-cov as dev dependencies."""
    with open(ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    dev_deps = data.get("dependency-groups", {}).get("dev", [])
    dep_names = [d.split(">")[0].split("=")[0].strip() for d in dev_deps]
    assert "pytest" in dep_names
    assert "pytest-cov" in dep_names


def test_us001_uv_sync_succeeds():
    """uv sync completes without errors."""
    result = subprocess.run(
        ["uv", "sync"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"uv sync failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
