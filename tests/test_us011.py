"""US-011: Implement Mitsuba scene assembly (buildScenes equivalent).

Tests that build_scenes() assembles camera, mesh, material, and envmap
into renderable Mitsuba scenes, including shared texture mode.
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


def _make_triangle_mesh_params(n_frames=1, shared_texture=False):
    """Create minimal triangle mesh parameters for scene building."""
    # Single triangle at z=3
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

    tex_batch = 1 if shared_texture else n_frames
    tex_h, tex_w = 8, 8
    diffuse = torch.full((tex_batch, tex_h, tex_w, 3), 0.5, dtype=torch.float32)
    specular = torch.full((tex_batch, tex_h, tex_w, 3), 0.04, dtype=torch.float32)
    roughness = torch.full((tex_batch, tex_h, tex_w, 1), 0.5, dtype=torch.float32)

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
        "screen_width": 32,
        "screen_height": 32,
    }


def test_us011_scene_renders_without_error():
    """build_scenes() produces a scene that renders without error."""
    import mitsuba as mi
    from scene_mitsuba import build_scenes

    params = _make_triangle_mesh_params(n_frames=1)
    scenes = build_scenes(**params)

    assert len(scenes) == 1
    img = mi.render(scenes[0])
    arr = np.array(img)
    assert arr.shape[0] == 32
    assert arr.shape[1] == 32


def test_us011_shared_texture_mode():
    """build_scenes() handles shared texture mode (diffuse.shape[0]==1, vertices.shape[0]==N)."""
    import mitsuba as mi
    from scene_mitsuba import build_scenes

    params = _make_triangle_mesh_params(n_frames=2, shared_texture=True)
    scenes = build_scenes(**params)

    assert len(scenes) == 2
    # Both should render without error
    for scene in scenes:
        img = mi.render(scene)
        arr = np.array(img)
        assert arr.shape[0] == 32
        assert arr.shape[1] == 32


def test_us011_multi_frame():
    """build_scenes() handles multiple frames with independent textures."""
    from scene_mitsuba import build_scenes

    params = _make_triangle_mesh_params(n_frames=3, shared_texture=False)
    scenes = build_scenes(**params)
    assert len(scenes) == 3


def test_us011_scene_count_matches_batch():
    """Number of returned scenes matches the batch dimension."""
    from scene_mitsuba import build_scenes

    for n in [1, 2, 4]:
        params = _make_triangle_mesh_params(n_frames=n)
        scenes = build_scenes(**params)
        assert len(scenes) == n
