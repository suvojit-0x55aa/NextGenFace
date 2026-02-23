"""US-008: Verify UV mapping consistency with original.

The original NextFace pipeline flips V coordinates in pipeline.py:34:
    self.uvMap[:, 1] = 1.0 - self.uvMap[:, 1]

This test verifies:
1. UV coordinates remain in [0, 1] range after flipping
2. The UV flip is correctly applied and stored in Mitsuba meshes
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def mi():
    """Import mitsuba with variant set."""
    from NextFace.mitsuba_variant import ensure_variant
    ensure_variant()
    import mitsuba as mi
    return mi


def _apply_uv_flip(uvs):
    """Apply the same UV flip as pipeline.py:34."""
    flipped = uvs.clone()
    flipped[:, 1] = 1.0 - flipped[:, 1]
    return flipped


def test_us008_uv_range_zero_one():
    """After UV flip, all coordinates should remain in [0, 1] range."""
    # Synthetic UVs covering full range including boundaries
    uvs = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.25, 0.75],
        [0.75, 0.25],
    ], dtype=torch.float32)

    flipped = _apply_uv_flip(uvs)

    # U coordinates unchanged
    assert torch.allclose(flipped[:, 0], uvs[:, 0]), "U coordinates should not change"

    # V coordinates flipped
    expected_v = 1.0 - uvs[:, 1]
    assert torch.allclose(flipped[:, 1], expected_v), "V should be flipped: v = 1 - v"

    # Range check
    assert flipped.min() >= 0.0, f"UV min {flipped.min()} < 0"
    assert flipped.max() <= 1.0, f"UV max {flipped.max()} > 1"


def test_us008_uv_flip_applied(mi):
    """Verify UV flip is correctly stored in Mitsuba mesh.

    We build two meshes — one with original UVs, one with flipped UVs —
    and verify the stored texture coordinates reflect the flip exactly
    as pipeline.py would produce it.
    """
    from NextFace.mesh_mitsuba import build_mesh

    vertices = torch.tensor([
        [-1.0, -1.0, 5.0],
        [ 1.0, -1.0, 5.0],
        [ 1.0,  1.0, 5.0],
        [-1.0,  1.0, 5.0],
    ], dtype=torch.float32)

    indices = torch.tensor([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=torch.int32)

    # Standard UVs: (0,0)=bottom-left convention
    uvs_original = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=torch.float32)

    # Flipped UVs (as pipeline.py:34 does it)
    uvs_flipped = _apply_uv_flip(uvs_original)

    mesh_orig = build_mesh(vertices, indices, uvs=uvs_original)
    mesh_flip = build_mesh(vertices, indices, uvs=uvs_flipped)

    params_orig = mi.traverse(mesh_orig)
    params_flip = mi.traverse(mesh_flip)

    uv_orig = np.array(params_orig["vertex_texcoords"]).reshape(-1, 2)
    uv_flip = np.array(params_flip["vertex_texcoords"]).reshape(-1, 2)

    # U coordinates identical between both meshes
    np.testing.assert_allclose(uv_orig[:, 0], uv_flip[:, 0], atol=1e-6,
                               err_msg="U coordinates should be unchanged by flip")

    # V coordinates flipped: flip_v = 1 - orig_v
    np.testing.assert_allclose(uv_flip[:, 1], 1.0 - uv_orig[:, 1], atol=1e-6,
                               err_msg="V coordinates should be flipped: v = 1 - v")

    # Stored UVs match what pipeline.py would produce
    np.testing.assert_allclose(uv_flip, uvs_flipped.numpy(), atol=1e-6,
                               err_msg="Mesh UVs should match pipeline.py flip result")
