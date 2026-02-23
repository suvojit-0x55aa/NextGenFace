"""US-012: Implement forward rendering (path tracing).

Tests that render_scenes() produces [N, H, W, 4] RGBA PyTorch tensors
from Mitsuba scenes built by build_scenes().
"""

import sys
import os

import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "NextFace"))

from mitsuba_variant import ensure_variant


@pytest.fixture(autouse=True)
def _variant():
    ensure_variant()


def _make_triangle_mesh_params(n_frames=1, screen_size=32):
    """Create minimal triangle mesh parameters for scene building."""
    vertices = torch.tensor([
        [[-0.5, -0.5, 3.0],
         [ 0.5, -0.5, 3.0],
         [ 0.0,  0.5, 3.0]]
    ], dtype=torch.float32).expand(n_frames, -1, -1).contiguous()

    indices = torch.tensor([[0, 1, 2]], dtype=torch.int32)

    normals = torch.tensor([
        [[0.0, 0.0, -1.0],
         [0.0, 0.0, -1.0],
         [0.0, 0.0, -1.0]]
    ], dtype=torch.float32).expand(n_frames, -1, -1).contiguous()

    uvs = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0]
    ], dtype=torch.float32)

    tex_h, tex_w = 8, 8
    diffuse = torch.full((n_frames, tex_h, tex_w, 3), 0.5, dtype=torch.float32)
    specular = torch.full((n_frames, tex_h, tex_w, 3), 0.04, dtype=torch.float32)
    roughness = torch.full((n_frames, tex_h, tex_w, 1), 0.5, dtype=torch.float32)

    focal = torch.full((n_frames,), 500.0, dtype=torch.float32)
    envmap = torch.full((n_frames, 16, 16, 3), 1.0, dtype=torch.float32)

    return {
        "vertices": vertices,
        "indices": indices,
        "normals": normals,
        "uvs": uvs,
        "diffuse": diffuse,
        "specular": specular,
        "roughness": roughness,
        "focal": focal,
        "envmap": envmap,
        "screen_width": screen_size,
        "screen_height": screen_size,
    }


def _build_test_scenes(n_frames=1, screen_size=32):
    """Build Mitsuba scenes for testing."""
    from scene_mitsuba import build_scenes
    params = _make_triangle_mesh_params(n_frames=n_frames, screen_size=screen_size)
    return build_scenes(**params), screen_size


def test_us012_render_output_shape():
    """render_scenes() returns [N, H, W, 4] tensor."""
    from render_mitsuba import render_scenes

    scenes, size = _build_test_scenes(n_frames=2, screen_size=32)
    result = render_scenes(scenes, spp=4)

    assert result.shape == (2, size, size, 4), f"Expected (2, {size}, {size}, 4), got {result.shape}"
    assert result.dtype == torch.float32


def test_us012_render_has_alpha():
    """Rendered output has a meaningful alpha channel (not all zeros)."""
    from render_mitsuba import render_scenes

    scenes, size = _build_test_scenes(n_frames=1, screen_size=32)
    result = render_scenes(scenes, spp=4)

    alpha = result[0, :, :, 3]
    # Triangle is in the scene, so some pixels should have alpha > 0
    assert alpha.max() > 0.0, "Alpha channel is all zeros — mesh not visible"


def test_us012_render_on_device():
    """render_scenes() output is on the requested device."""
    from render_mitsuba import render_scenes

    scenes, _ = _build_test_scenes(n_frames=1, screen_size=32)
    result = render_scenes(scenes, spp=4, device="cpu")

    assert result.device == torch.device("cpu")


def test_us012_render_rgb_nonzero():
    """Rendered RGB channels are not all zero (scene is illuminated)."""
    from render_mitsuba import render_scenes

    scenes, _ = _build_test_scenes(n_frames=1, screen_size=32)
    result = render_scenes(scenes, spp=4)

    rgb = result[0, :, :, :3]
    assert rgb.max() > 0.0, "RGB channels are all zeros — scene not illuminated"


def test_us012_render_single_scene():
    """render_scenes() works with a single scene."""
    from render_mitsuba import render_scenes

    scenes, size = _build_test_scenes(n_frames=1, screen_size=32)
    result = render_scenes(scenes, spp=4)

    assert result.shape == (1, size, size, 4)
