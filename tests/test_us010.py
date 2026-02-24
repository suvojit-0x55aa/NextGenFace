"""US-010: Verify spherical harmonics to envmap pipeline.

Tests that SphericalHarmonics.toEnvMap() output integrates correctly with
build_envmap() and that SH band 0 produces expected uniform illumination.
"""

import math

import pytest
import torch
import numpy as np

from geometry.sphericalharmonics import SphericalHarmonics
from rendering._envmap import build_envmap
from rendering._variant import ensure_variant


@pytest.fixture(autouse=True)
def _variant():
    ensure_variant()


ENVMAP_RES = 16  # small for fast tests


def _make_sh(bands=4, device="cpu"):
    """Create a SphericalHarmonics instance."""
    return SphericalHarmonics(ENVMAP_RES, device)


def _band0_coeffs(value, bands=4, device="cpu"):
    """Create SH coefficients with only band 0 set to `value` (uniform).

    Returns [1, bands*bands, 3] tensor.
    """
    n_coeffs = bands * bands
    coeffs = torch.zeros(1, n_coeffs, 3, device=device)
    coeffs[0, 0, :] = value
    return coeffs


# ─── Test: SH toEnvMap output feeds into build_envmap ───


def test_us010_sh_to_envmap_integration():
    """SphericalHarmonics.toEnvMap() output is accepted by build_envmap()."""
    sh = _make_sh()
    coeffs = _band0_coeffs(1.0)
    envmaps = sh.toEnvMap(coeffs)

    assert envmaps.shape == (1, ENVMAP_RES, ENVMAP_RES, 3)

    # Should not raise
    result = build_envmap(envmaps[0])
    assert result["type"] == "envmap"
    assert "bitmap" in result


# ─── Test: Band 0 produces uniform illumination ───


def test_us010_sh_uniform_illumination():
    """Band 0 only should produce near-uniform envmap (constant lighting)."""
    sh = _make_sh()
    coeffs = _band0_coeffs(2.0)
    envmaps = sh.toEnvMap(coeffs)
    envmap = envmaps[0]  # [H, W, 3]

    # Band 0 (DC term) = Y_0^0 = 1/(2*sqrt(pi))
    # With coefficient c, the envmap value everywhere should be c * Y_0^0
    # Check that all pixels are approximately equal (uniform)
    mean_val = envmap.mean().item()
    assert mean_val > 0, "Envmap should have positive values"

    # Standard deviation across spatial dimensions should be very small
    # relative to mean (uniform illumination)
    std_val = envmap.std().item()
    assert std_val / mean_val < 0.01, (
        f"Band 0 envmap should be uniform, but std/mean = {std_val / mean_val:.4f}"
    )


# ─── Test: Brightness monotonically related to band 0 coefficient ───


def test_us010_sh_brightness_monotonic():
    """Increasing band 0 coefficient should increase envmap brightness."""
    sh = _make_sh()
    values = [0.5, 1.0, 2.0, 4.0]
    means = []

    for v in values:
        coeffs = _band0_coeffs(v)
        envmaps = sh.toEnvMap(coeffs)
        means.append(envmaps[0].mean().item())

    # Each subsequent mean should be strictly greater
    for i in range(len(means) - 1):
        assert means[i + 1] > means[i], (
            f"Brightness should increase: coeff {values[i]}→{values[i+1]}, "
            f"mean {means[i]:.4f}→{means[i+1]:.4f}"
        )


# ─── Test: Multi-frame batch works ───


def test_us010_sh_batch():
    """toEnvMap handles batch dimension correctly."""
    sh = _make_sh()
    n_frames = 3
    n_coeffs = 16  # 4 bands
    coeffs = torch.zeros(n_frames, n_coeffs, 3)
    # Different band 0 per frame
    for i in range(n_frames):
        coeffs[i, 0, :] = float(i + 1)

    envmaps = sh.toEnvMap(coeffs)
    assert envmaps.shape == (n_frames, ENVMAP_RES, ENVMAP_RES, 3)

    # Each frame should have different brightness
    frame_means = [envmaps[i].mean().item() for i in range(n_frames)]
    for i in range(n_frames - 1):
        assert frame_means[i + 1] > frame_means[i]


# ─── Test: Full render integration with envmap from SH ───


def test_us010_sh_envmap_renders():
    """A scene with SH-derived envmap renders without error."""
    import mitsuba as mi

    sh = _make_sh()
    coeffs = _band0_coeffs(2.0)
    envmaps = sh.toEnvMap(coeffs)
    envmap_dict = build_envmap(envmaps[0])

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "direct"},
        "sensor": {
            "type": "perspective",
            "fov": 45,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0, 0, 0], target=[0, 0, 1], up=[0, 1, 0]
            ),
            "film": {"type": "hdrfilm", "width": 32, "height": 32},
        },
        "envmap": envmap_dict,
        "shape": {
            "type": "sphere",
            "radius": 0.5,
            "to_world": mi.ScalarTransform4f.translate([0, 0, 3]),
        },
    }

    scene = mi.load_dict(scene_dict)
    img = mi.render(scene)
    arr = np.array(img)
    assert arr.shape == (32, 32, 3)
    assert np.any(arr > 0), "Rendered image should have non-zero pixels"
