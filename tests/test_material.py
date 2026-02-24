"""Tests for BSDF material, environment map emitter, and SH-to-envmap pipeline."""

import math

import numpy as np
import pytest
import torch

from rendering._variant import ensure_variant

ensure_variant()

import mitsuba as mi
from rendering._material import build_material, _tensor_to_bitmap
from rendering._envmap import build_envmap
from geometry.sphericalharmonics import SphericalHarmonics


# ---------------------------------------------------------------------------
# Principled BSDF material
# ---------------------------------------------------------------------------


class TestBuildMaterial:
    """Tests for build_material() function."""

    def test_material_has_diffuse(self):
        """Material dict contains base_color from diffuse texture."""
        diffuse = torch.rand(64, 64, 3)
        mat = build_material(diffuse)
        assert mat["type"] == "principled"
        assert "base_color" in mat
        assert mat["base_color"]["type"] == "bitmap"

    def test_material_has_roughness(self):
        """Material dict contains roughness from roughness texture."""
        diffuse = torch.rand(64, 64, 3)
        roughness = torch.rand(64, 64, 1)
        mat = build_material(diffuse, roughness=roughness)
        assert "roughness" in mat
        assert mat["roughness"]["type"] == "bitmap"

    def test_material_has_specular(self):
        """Material dict contains specular as a scalar float."""
        diffuse = torch.rand(64, 64, 3)
        specular = torch.rand(64, 64, 3)
        mat = build_material(diffuse, specular=specular)
        assert "specular" in mat
        assert isinstance(mat["specular"], float)
        assert 0.0 <= mat["specular"] <= 1.0

    def test_material_all_textures(self):
        """Material with all three textures builds correctly."""
        diffuse = torch.rand(32, 32, 3)
        specular = torch.rand(32, 32, 3)
        roughness = torch.rand(32, 32, 1)
        mat = build_material(diffuse, specular=specular, roughness=roughness)
        assert mat["type"] == "principled"
        assert "base_color" in mat
        assert "specular" in mat
        assert "roughness" in mat

    def test_material_diffuse_only(self):
        """Material with only diffuse (no specular/roughness) is valid."""
        diffuse = torch.rand(32, 32, 3)
        mat = build_material(diffuse)
        assert "specular" not in mat
        assert "roughness" not in mat

    def test_material_loads_in_scene(self):
        """Material dict can be used in a Mitsuba scene via mi.load_dict()."""
        diffuse = torch.rand(16, 16, 3)
        specular = torch.rand(16, 16, 3)
        roughness = torch.rand(16, 16, 1)
        mat = build_material(diffuse, specular=specular, roughness=roughness)

        scene_dict = {
            "type": "scene",
            "integrator": {"type": "direct"},
            "sensor": {
                "type": "perspective",
                "film": {"type": "hdrfilm", "width": 16, "height": 16},
            },
            "shape": {
                "type": "sphere",
                "bsdf": mat,
            },
        }
        scene = mi.load_dict(scene_dict)
        assert scene is not None


class TestTensorToBitmap:
    """Tests for _tensor_to_bitmap helper."""

    def test_rgb_tensor(self):
        """[H, W, 3] tensor converts to bitmap."""
        t = torch.rand(32, 32, 3)
        bmp = _tensor_to_bitmap(t)
        assert isinstance(bmp, mi.Bitmap)

    def test_single_channel_tensor(self):
        """[H, W, 1] tensor converts to bitmap."""
        t = torch.rand(32, 32, 1)
        bmp = _tensor_to_bitmap(t)
        assert isinstance(bmp, mi.Bitmap)


# ---------------------------------------------------------------------------
# Environment map emitter
# ---------------------------------------------------------------------------


class TestBuildEnvmap:
    """Tests for build_envmap() function."""

    def test_envmap_dict_valid(self):
        """Envmap dict has correct type and can be loaded in a scene."""
        envmap = torch.ones(16, 32, 3, dtype=torch.float32) * 0.5
        result = build_envmap(envmap)

        assert result["type"] == "envmap"
        assert "bitmap" in result
        assert isinstance(result["bitmap"], mi.Bitmap)

    def test_envmap_from_tensor_shape(self):
        """Envmap bitmap preserves spatial dimensions from input tensor."""
        H, W = 16, 32
        envmap = torch.rand(H, W, 3, dtype=torch.float32)
        result = build_envmap(envmap)

        bmp = result["bitmap"]
        size = list(bmp.size())
        assert size == [W, H]

    def test_envmap_in_scene(self):
        """Envmap dict integrates into a full Mitsuba scene."""
        envmap = torch.ones(8, 16, 3, dtype=torch.float32) * 0.3
        envmap_dict = build_envmap(envmap)

        scene_dict = {
            "type": "scene",
            "integrator": {"type": "direct"},
            "emitter": envmap_dict,
            "sensor": {
                "type": "perspective",
                "fov": 45.0,
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=[0, 0, 0], target=[0, 0, 1], up=[0, -1, 0]
                ),
                "film": {"type": "hdrfilm", "width": 8, "height": 8},
            },
            "shape": {
                "type": "sphere",
                "radius": 0.5,
                "to_world": mi.ScalarTransform4f.translate([0, 0, 3]),
            },
        }
        scene = mi.load_dict(scene_dict)
        img = mi.render(scene)
        assert img.shape[0] == 8 and img.shape[1] == 8

    def test_envmap_rejects_wrong_shape(self):
        """build_envmap raises ValueError for wrong tensor shape."""
        with pytest.raises(ValueError):
            build_envmap(torch.ones(16, 32))  # missing channel dim

        with pytest.raises(ValueError):
            build_envmap(torch.ones(16, 32, 1))  # wrong channel count

    def test_envmap_from_gpu_tensor(self):
        """Envmap works with CUDA tensors (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        envmap = torch.ones(8, 16, 3, dtype=torch.float32, device="cuda")
        result = build_envmap(envmap)
        assert result["type"] == "envmap"


# ---------------------------------------------------------------------------
# Spherical harmonics to envmap pipeline
# ---------------------------------------------------------------------------


ENVMAP_RES = 16  # small for fast tests


def _make_sh(bands=4, device="cpu"):
    """Create a SphericalHarmonics instance."""
    return SphericalHarmonics(ENVMAP_RES, device)


def _band0_coeffs(value, bands=4, device="cpu"):
    """Create SH coefficients with only band 0 set to `value` (uniform)."""
    n_coeffs = bands * bands
    coeffs = torch.zeros(1, n_coeffs, 3, device=device)
    coeffs[0, 0, :] = value
    return coeffs


def test_sh_to_envmap_integration():
    """SphericalHarmonics.toEnvMap() output is accepted by build_envmap()."""
    sh = _make_sh()
    coeffs = _band0_coeffs(1.0)
    envmaps = sh.toEnvMap(coeffs)

    assert envmaps.shape == (1, ENVMAP_RES, ENVMAP_RES, 3)

    result = build_envmap(envmaps[0])
    assert result["type"] == "envmap"
    assert "bitmap" in result


def test_sh_uniform_illumination():
    """Band 0 only should produce near-uniform envmap (constant lighting)."""
    sh = _make_sh()
    coeffs = _band0_coeffs(2.0)
    envmaps = sh.toEnvMap(coeffs)
    envmap = envmaps[0]

    mean_val = envmap.mean().item()
    assert mean_val > 0, "Envmap should have positive values"

    std_val = envmap.std().item()
    assert std_val / mean_val < 0.01, (
        f"Band 0 envmap should be uniform, but std/mean = {std_val / mean_val:.4f}"
    )


def test_sh_brightness_monotonic():
    """Increasing band 0 coefficient should increase envmap brightness."""
    sh = _make_sh()
    values = [0.5, 1.0, 2.0, 4.0]
    means = []

    for v in values:
        coeffs = _band0_coeffs(v)
        envmaps = sh.toEnvMap(coeffs)
        means.append(envmaps[0].mean().item())

    for i in range(len(means) - 1):
        assert means[i + 1] > means[i], (
            f"Brightness should increase: coeff {values[i]}->{values[i+1]}, "
            f"mean {means[i]:.4f}->{means[i+1]:.4f}"
        )


def test_sh_batch():
    """toEnvMap handles batch dimension correctly."""
    sh = _make_sh()
    n_frames = 3
    n_coeffs = 16  # 4 bands
    coeffs = torch.zeros(n_frames, n_coeffs, 3)
    for i in range(n_frames):
        coeffs[i, 0, :] = float(i + 1)

    envmaps = sh.toEnvMap(coeffs)
    assert envmaps.shape == (n_frames, ENVMAP_RES, ENVMAP_RES, 3)

    frame_means = [envmaps[i].mean().item() for i in range(n_frames)]
    for i in range(n_frames - 1):
        assert frame_means[i + 1] > frame_means[i]


def test_sh_envmap_renders():
    """A scene with SH-derived envmap renders without error."""
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
