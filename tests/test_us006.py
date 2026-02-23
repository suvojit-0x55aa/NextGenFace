"""Tests for US-006: Mesh construction from vertex and face buffers."""

import pytest
import torch

from NextFace.mitsuba_variant import ensure_variant

ensure_variant()

import mitsuba as mi
from NextFace.mesh_mitsuba import build_mesh


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


class TestMeshVertexCount:
    """test_us006_mesh_vertex_count: Mesh has correct vertex count."""

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
    """test_us006_mesh_face_count: Mesh has correct face count."""

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


class TestMeshWithUVs:
    """test_us006_mesh_with_uvs: Mesh correctly handles UV coordinates."""

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
