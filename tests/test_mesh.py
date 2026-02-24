"""Tests for mesh construction from vertex/face buffers and UV mapping."""

import numpy as np
import pytest
import torch

from rendering._variant import ensure_variant

ensure_variant()

import mitsuba as mi
from rendering._mesh import build_mesh


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _make_triangle():
    """Create a simple triangle mesh for testing."""
    vertices = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float32
    )
    indices = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    return vertices, indices


def _make_quad():
    """Create a quad (two triangles) mesh for testing."""
    vertices = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    indices = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
    return vertices, indices


def _apply_uv_flip(uvs):
    """Apply the same UV flip as pipeline.py:34."""
    flipped = uvs.clone()
    flipped[:, 1] = 1.0 - flipped[:, 1]
    return flipped


# ---------------------------------------------------------------------------
# Mesh vertex and face counts
# ---------------------------------------------------------------------------


class TestMeshVertexCount:
    """Mesh has correct vertex count."""

    def test_triangle_vertex_count(self):
        vertices, indices = _make_triangle()
        mesh = build_mesh(vertices, indices)
        assert mesh.vertex_count() == 3

    def test_quad_vertex_count(self):
        vertices, indices = _make_quad()
        mesh = build_mesh(vertices, indices)
        assert mesh.vertex_count() == 4

    def test_larger_mesh(self):
        V = 100
        F = 50
        vertices = torch.randn(V, 3, dtype=torch.float32)
        indices = torch.randint(0, V, (F, 3), dtype=torch.int32)
        mesh = build_mesh(vertices, indices)
        assert mesh.vertex_count() == V


class TestMeshFaceCount:
    """Mesh has correct face count."""

    def test_triangle_face_count(self):
        vertices, indices = _make_triangle()
        mesh = build_mesh(vertices, indices)
        assert mesh.face_count() == 1

    def test_quad_face_count(self):
        vertices, indices = _make_quad()
        mesh = build_mesh(vertices, indices)
        assert mesh.face_count() == 2

    def test_larger_mesh(self):
        V = 100
        F = 50
        vertices = torch.randn(V, 3, dtype=torch.float32)
        indices = torch.randint(0, V, (F, 3), dtype=torch.int32)
        mesh = build_mesh(vertices, indices)
        assert mesh.face_count() == F


# ---------------------------------------------------------------------------
# Mesh with UVs and normals
# ---------------------------------------------------------------------------


class TestMeshWithUVs:
    """Mesh correctly handles UV coordinates and normals."""

    def test_mesh_with_uvs(self):
        vertices, indices = _make_triangle()
        uvs = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
        )
        mesh = build_mesh(vertices, indices, uvs=uvs)
        assert mesh.vertex_count() == 3
        assert mesh.face_count() == 1

    def test_mesh_with_normals(self):
        vertices, indices = _make_triangle()
        normals = torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
            dtype=torch.float32,
        )
        mesh = build_mesh(vertices, indices, normals=normals)
        assert mesh.vertex_count() == 3

    def test_mesh_with_normals_and_uvs(self):
        vertices, indices = _make_quad()
        normals = torch.zeros(4, 3, dtype=torch.float32)
        normals[:, 2] = -1.0
        uvs = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        mesh = build_mesh(vertices, indices, normals=normals, uvs=uvs)
        assert mesh.vertex_count() == 4
        assert mesh.face_count() == 2

    def test_mesh_is_mi_mesh(self):
        vertices, indices = _make_triangle()
        mesh = build_mesh(vertices, indices)
        assert isinstance(mesh, mi.Mesh)

    def test_mesh_renders_in_scene(self):
        """Mesh can be used in a Mitsuba scene and rendered."""
        vertices, indices = _make_triangle()
        mesh = build_mesh(vertices, indices)

        scene = mi.load_dict(
            {
                "type": "scene",
                "integrator": {"type": "direct"},
                "sensor": {
                    "type": "perspective",
                    "fov": 45,
                    "to_world": mi.ScalarTransform4f.look_at(
                        origin=[0.5, 0.5, -1],
                        target=[0.5, 0.5, 1],
                        up=[0, -1, 0],
                    ),
                    "film": {"type": "hdrfilm", "width": 32, "height": 32},
                },
                "emitter": {"type": "constant"},
                "mesh": mesh,
            }
        )
        img = mi.render(scene, spp=1)
        assert img.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# UV flip consistency
# ---------------------------------------------------------------------------


def test_uv_range_zero_one():
    """After UV flip, all coordinates should remain in [0, 1] range."""
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

    assert torch.allclose(flipped[:, 0], uvs[:, 0]), "U coordinates should not change"

    expected_v = 1.0 - uvs[:, 1]
    assert torch.allclose(flipped[:, 1], expected_v), "V should be flipped: v = 1 - v"

    assert flipped.min() >= 0.0, f"UV min {flipped.min()} < 0"
    assert flipped.max() <= 1.0, f"UV max {flipped.max()} > 1"


def test_uv_flip_applied():
    """Verify UV flip is correctly stored in Mitsuba mesh."""
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

    uvs_original = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=torch.float32)

    uvs_flipped = _apply_uv_flip(uvs_original)

    mesh_orig = build_mesh(vertices, indices, uvs=uvs_original)
    mesh_flip = build_mesh(vertices, indices, uvs=uvs_flipped)

    params_orig = mi.traverse(mesh_orig)
    params_flip = mi.traverse(mesh_flip)

    uv_orig = np.array(params_orig["vertex_texcoords"]).reshape(-1, 2)
    uv_flip = np.array(params_flip["vertex_texcoords"]).reshape(-1, 2)

    np.testing.assert_allclose(uv_orig[:, 0], uv_flip[:, 0], atol=1e-6,
                               err_msg="U coordinates should be unchanged by flip")

    np.testing.assert_allclose(uv_flip[:, 1], 1.0 - uv_orig[:, 1], atol=1e-6,
                               err_msg="V coordinates should be flipped: v = 1 - v")

    np.testing.assert_allclose(uv_flip, uvs_flipped.numpy(), atol=1e-6,
                               err_msg="Mesh UVs should match pipeline.py flip result")
