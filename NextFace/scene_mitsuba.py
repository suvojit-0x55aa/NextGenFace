"""Mitsuba 3 scene assembly from component builders.

Replaces the original PyRedner buildScenes() method by assembling camera,
mesh, material, and envmap dicts into complete Mitsuba scenes.
"""

import torch
import mitsuba as mi

from camera_mitsuba import build_camera
from mesh_mitsuba import build_mesh
from material_mitsuba import build_material
from envmap_mitsuba import build_envmap


def build_scenes(vertices, indices, normals, uvs, diffuse, specular,
                 roughness, focal, envmap, screen_width, screen_height,
                 samples=8, bounces=1):
    """Build a list of Mitsuba scenes from batched parameters.

    Args:
        vertices: [N, V, 3] float32 torch tensor
        indices: [F, 3] int32 torch tensor
        normals: [N, V, 3] float32 torch tensor
        uvs: [V, 2] float32 torch tensor
        diffuse: [N, H, W, 3] or [1, H, W, 3] torch tensor
        specular: [N, H, W, 3] or [1, H, W, 3] torch tensor
        roughness: [N, H, W, 1] or [1, H, W, 1] torch tensor
        focal: [N] torch tensor
        envmap: [N, H, W, 3] torch tensor
        screen_width: int
        screen_height: int
        samples: int, samples per pixel
        bounces: int, max path bounces

    Returns:
        list[mi.Scene]: list of N loaded Mitsuba scenes
    """
    assert vertices.dim() == 3 and vertices.shape[-1] == 3
    assert normals.dim() == 3 and normals.shape[-1] == 3
    assert indices.dim() == 2 and indices.shape[-1] == 3
    assert uvs.dim() == 2 and uvs.shape[-1] == 2
    assert diffuse.dim() == 4 and diffuse.shape[-1] == 3
    assert specular.dim() == 4 and specular.shape[-1] == 3
    assert roughness.dim() == 4 and roughness.shape[-1] == 1
    assert focal.dim() == 1
    assert envmap.dim() == 4 and envmap.shape[-1] == 3
    assert vertices.shape[0] == focal.shape[0] == envmap.shape[0]
    assert diffuse.shape[0] == specular.shape[0] == roughness.shape[0]
    assert diffuse.shape[0] == 1 or diffuse.shape[0] == vertices.shape[0]

    shared_texture = diffuse.shape[0] == 1
    n_frames = vertices.shape[0]
    scenes = []

    for i in range(n_frames):
        tex_idx = 0 if shared_texture else i

        # Build components
        camera_dict = build_camera(focal[i], screen_width, screen_height)
        mesh = build_mesh(vertices[i], indices, normals[i], uvs)
        material_dict = build_material(
            diffuse[tex_idx], specular[tex_idx], roughness[tex_idx]
        )
        envmap_dict = build_envmap(envmap[i])

        # Assemble scene dict with mesh + bsdf
        scene_dict = {
            "type": "scene",
            "integrator": {
                "type": "path",
                "max_depth": bounces + 1,
            },
            "sensor": camera_dict,
            "envmap": envmap_dict,
        }

        # Attach BSDF to mesh and add to scene
        mesh.set_bsdf(mi.load_dict(material_dict))
        scene_dict["face_mesh"] = mesh

        scene = mi.load_dict(scene_dict)
        scenes.append(scene)

    return scenes
