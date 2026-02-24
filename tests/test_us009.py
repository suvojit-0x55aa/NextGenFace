"""Tests for US-009: Environment map emitter from tensor."""

import pytest
import torch

from NextFace.variant_mitsuba import ensure_variant

ensure_variant()

import mitsuba as mi
from NextFace.envmap_mitsuba import build_envmap


class TestBuildEnvmap:
    """Tests for build_envmap() function."""

    def test_us009_envmap_dict_valid(self):
        """Envmap dict has correct type and can be loaded in a scene."""
        envmap = torch.ones(16, 32, 3, dtype=torch.float32) * 0.5
        result = build_envmap(envmap)

        assert result["type"] == "envmap"
        assert "bitmap" in result
        assert isinstance(result["bitmap"], mi.Bitmap)

    def test_us009_envmap_from_tensor_shape(self):
        """Envmap bitmap preserves spatial dimensions from input tensor."""
        H, W = 16, 32
        envmap = torch.rand(H, W, 3, dtype=torch.float32)
        result = build_envmap(envmap)

        bmp = result["bitmap"]
        # mi.Bitmap size() returns a drjit array [width, height]
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
