"""Tests for US-007: Principled BSDF material from texture tensors."""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "NextFace"))

from variant_mitsuba import ensure_variant

ensure_variant()

import mitsuba as mi
from material_mitsuba import build_material, _tensor_to_bitmap


class TestBuildMaterial:
    """Tests for build_material() function."""

    def test_us007_material_has_diffuse(self):
        """Material dict contains base_color from diffuse texture."""
        diffuse = torch.rand(64, 64, 3)
        mat = build_material(diffuse)
        assert mat["type"] == "principled"
        assert "base_color" in mat
        assert mat["base_color"]["type"] == "bitmap"

    def test_us007_material_has_roughness(self):
        """Material dict contains roughness from roughness texture."""
        diffuse = torch.rand(64, 64, 3)
        roughness = torch.rand(64, 64, 1)
        mat = build_material(diffuse, roughness=roughness)
        assert "roughness" in mat
        assert mat["roughness"]["type"] == "bitmap"

    def test_us007_material_has_specular(self):
        """Material dict contains specular as a scalar float."""
        diffuse = torch.rand(64, 64, 3)
        specular = torch.rand(64, 64, 3)
        mat = build_material(diffuse, specular=specular)
        assert "specular" in mat
        assert isinstance(mat["specular"], float)
        assert 0.0 <= mat["specular"] <= 1.0

    def test_us007_material_all_textures(self):
        """Material with all three textures builds correctly."""
        diffuse = torch.rand(32, 32, 3)
        specular = torch.rand(32, 32, 3)
        roughness = torch.rand(32, 32, 1)
        mat = build_material(diffuse, specular=specular, roughness=roughness)
        assert mat["type"] == "principled"
        assert "base_color" in mat
        assert "specular" in mat
        assert "roughness" in mat

    def test_us007_material_diffuse_only(self):
        """Material with only diffuse (no specular/roughness) is valid."""
        diffuse = torch.rand(32, 32, 3)
        mat = build_material(diffuse)
        assert "specular" not in mat
        assert "roughness" not in mat

    def test_us007_material_loads_in_scene(self):
        """Material dict can be used in a Mitsuba scene via mi.load_dict()."""
        diffuse = torch.rand(16, 16, 3)
        specular = torch.rand(16, 16, 3)
        roughness = torch.rand(16, 16, 1)
        mat = build_material(diffuse, specular=specular, roughness=roughness)

        # Build a minimal scene with this material applied to a sphere
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
