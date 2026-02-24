"""Mitsuba 3 mesh construction from PyTorch vertex and face buffers.

Converts PyTorch tensors (vertices, indices, normals, uvs) into a Mitsuba 3 Mesh
object suitable for differentiable rendering.
"""

import torch
import numpy as np
import mitsuba as mi


def build_mesh(vertices, indices, normals=None, uvs=None):
    """Build a Mitsuba Mesh from PyTorch tensors.

    Args:
        vertices: [V, 3] float32 torch tensor of vertex positions.
        indices: [F, 3] int32 torch tensor of face indices.
        normals: [V, 3] float32 torch tensor of vertex normals (optional).
        uvs: [V, 2] float32 torch tensor of UV coordinates (optional).

    Returns:
        mi.Mesh: A Mitsuba mesh object.
    """
    V = vertices.shape[0]
    F = indices.shape[0]

    has_normals = normals is not None
    has_uvs = uvs is not None

    mesh = mi.Mesh(
        "face_mesh",
        vertex_count=V,
        face_count=F,
        has_vertex_normals=has_normals,
        has_vertex_texcoords=has_uvs,
    )

    params = mi.traverse(mesh)

    # Vertex positions: [V, 3] -> flat [V*3]
    verts_np = vertices.detach().cpu().to(torch.float32).numpy().ravel()
    params["vertex_positions"] = mi.Float(verts_np)

    # Face indices: [F, 3] -> flat [F*3]
    faces_np = indices.detach().cpu().to(torch.int32).numpy().astype(np.uint32).ravel()
    params["faces"] = mi.UInt32(faces_np)

    if has_normals:
        normals_np = normals.detach().cpu().to(torch.float32).numpy().ravel()
        params["vertex_normals"] = mi.Float(normals_np)

    if has_uvs:
        uvs_np = uvs.detach().cpu().to(torch.float32).numpy().ravel()
        params["vertex_texcoords"] = mi.Float(uvs_np)

    params.update()

    return mesh
