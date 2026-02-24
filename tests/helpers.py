"""Shared test geometry helpers.

Extracted from duplicated helpers across test_us011, test_us012, test_us013,
test_us020, test_us021, test_us026.
"""

import torch
import numpy as np


def make_triangle_mesh_params(
    n_frames=1, shared_texture=False, screen_size=32,
    diffuse_value=0.5, z=3.0,
):
    """Create minimal triangle mesh parameters for scene building."""
    vertices = torch.tensor([
        [[-0.5, -0.5, z],
         [ 0.5, -0.5, z],
         [ 0.0,  0.5, z]]
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
    diffuse = torch.full((tex_batch, tex_h, tex_w, 3), diffuse_value, dtype=torch.float32)
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
        "screen_width": screen_size,
        "screen_height": screen_size,
    }


def make_sphere_mesh(radius=5.0, center=(0, 0, 50), subdivisions=8):
    """Create a UV sphere mesh as torch tensors."""
    n_lat = subdivisions
    n_lon = subdivisions * 2

    vertices = []
    uvs = []

    cx, cy, cz = center
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon + 1):
            phi = 2 * np.pi * j / n_lon
            x = cx + radius * np.sin(theta) * np.cos(phi)
            y = cy + radius * np.sin(theta) * np.sin(phi)
            z = cz + radius * np.cos(theta)
            vertices.append([x, y, z])
            uvs.append([j / n_lon, i / n_lat])

    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            v0 = i * (n_lon + 1) + j
            v1 = v0 + 1
            v2 = (i + 1) * (n_lon + 1) + j
            v3 = v2 + 1
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int32)
    uvs_t = torch.tensor(uvs, dtype=torch.float32)

    return vertices, faces, uvs_t


def compute_normals(vertices, faces):
    """Compute per-vertex normals from vertices and faces."""
    v0 = vertices[faces[:, 0].long()]
    v1 = vertices[faces[:, 1].long()]
    v2 = vertices[faces[:, 2].long()]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_normals = face_normals / (face_normals.norm(dim=-1, keepdim=True) + 1e-8)

    vertex_normals = torch.zeros_like(vertices)
    for fi in range(faces.shape[0]):
        for vi in range(3):
            vertex_normals[faces[fi, vi].long()] += face_normals[fi]

    vertex_normals = vertex_normals / (vertex_normals.norm(dim=-1, keepdim=True) + 1e-8)
    return vertex_normals


def make_single_triangle(z=50.0):
    """Single triangle visible from origin looking at +Z, beyond clip_near."""
    vertices = torch.tensor([
        [-1.0, -1.0, z],
        [1.0, -1.0, z],
        [0.0, 1.0, z],
    ], dtype=torch.float32)
    indices = torch.tensor([[0, 2, 1]], dtype=torch.int32)
    normals = torch.tensor([
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
    ], dtype=torch.float32)
    uvs = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0],
    ], dtype=torch.float32)
    return vertices, indices, normals, uvs


def make_textures(h=4, w=4, diffuse_val=0.6, roughness_val=0.5):
    """Create minimal texture tensors [1, H, W, C]."""
    diffuse = torch.full((1, h, w, 3), diffuse_val, dtype=torch.float32)
    specular = torch.full((1, h, w, 3), 0.04, dtype=torch.float32)
    roughness = torch.full((1, h, w, 1), roughness_val, dtype=torch.float32)
    return diffuse, specular, roughness


def make_envmap(h=8, w=16, brightness=1.0):
    """Create minimal envmap tensor [1, H, W, 3]."""
    return torch.full((1, h, w, 3), brightness, dtype=torch.float32)
