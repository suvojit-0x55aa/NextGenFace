"""US-015: Verify gradient correctness for vertex positions.

Tests that gradients w.r.t. vertex positions are correct so that
shape optimization works in the NextFace pipeline.
"""

import sys
import os

import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "NextFace"))

from mitsuba_variant import ensure_variant


@pytest.fixture(autouse=True)
def _ad_variant():
    """Ensure an AD-capable variant is active; skip if unavailable."""
    variant = ensure_variant()
    if "ad" not in variant:
        pytest.skip(f"AD variant required, got {variant}")


def _make_mesh_scene(vertices_np, faces_np, size=32, near_clip=0.1):
    """Create a scene with a custom triangle mesh.

    Args:
        vertices_np: [V, 3] float32 numpy array of vertex positions.
        faces_np: [F, 3] uint32 numpy array of face indices.
        size: image resolution (square).
        near_clip: near clipping plane distance.

    Returns:
        mi.Scene: loaded Mitsuba scene.
    """
    import mitsuba as mi

    V = vertices_np.shape[0]
    F = faces_np.shape[0]

    mesh = mi.Mesh(
        "shape",
        vertex_count=V,
        face_count=F,
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )
    params = mi.traverse(mesh)
    params["vertex_positions"] = mi.Float(
        vertices_np.astype(np.float32).ravel()
    )
    params["faces"] = mi.UInt32(faces_np.astype(np.uint32).ravel())
    params.update()

    mesh.set_bsdf(
        mi.load_dict(
            {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [0.8, 0.8, 0.8]},
            }
        )
    )

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "direct_projective", "sppi": 0},
        "emitter": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
        },
        "sensor": {
            "type": "perspective",
            "fov": 90.0,
            "near_clip": near_clip,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0, 0, 0], target=[0, 0, 1], up=[0, 1, 0]
            ),
            "film": {
                "type": "hdrfilm",
                "width": size,
                "height": size,
                "pixel_format": "rgba",
                "sample_border": True,
            },
            "sampler": {"type": "independent", "sample_count": 4},
        },
        "shape": mesh,
    }
    return mi.load_dict(scene_dict)


def test_us015_vertex_grad_nonzero():
    """Gradients w.r.t. vertex positions are non-zero for visible vertices
    and zero for occluded (behind-camera) vertices."""
    from gradient_bridge import differentiable_render

    # 6 vertices, 2 triangles:
    #   Triangle 0 (v0-v2): visible, in front of camera at z=3
    #   Triangle 1 (v3-v5): behind camera at z=-3, not visible
    vertices = np.array(
        [
            [-1.0, -1.0, 3.0],
            [1.0, -1.0, 3.0],
            [0.0, 1.0, 3.0],
            [-1.0, -1.0, -3.0],
            [1.0, -1.0, -3.0],
            [0.0, 1.0, -3.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)

    scene = _make_mesh_scene(vertices, faces, size=32)

    verts_torch = torch.tensor(
        vertices.ravel(), dtype=torch.float32, requires_grad=True
    )

    img = differentiable_render(
        scene, {"shape.vertex_positions": verts_torch}, spp=16
    )

    loss = img[..., :3].pow(2).sum()
    loss.backward()

    assert verts_torch.grad is not None, "No gradient on vertex positions"

    grad = verts_torch.grad.reshape(-1, 3)

    # Visible vertices (0, 1, 2) should have non-zero gradient
    for i in range(3):
        assert grad[i].abs().sum() > 1e-6, (
            f"Vertex {i} gradient is zero but vertex is visible"
        )

    # Occluded vertices (3, 4, 5) behind camera should have zero gradient
    for i in range(3, 6):
        assert grad[i].abs().sum() < 1e-4, (
            f"Vertex {i} gradient is non-zero ({grad[i]}) but vertex is behind camera"
        )


def test_us015_vertex_grad_finite_diff():
    """Analytic vertex gradient matches finite difference approximation."""
    import mitsuba as mi
    from gradient_bridge import differentiable_render

    # Single visible triangle
    vertices = np.array(
        [
            [-1.0, -1.0, 3.0],
            [1.0, -1.0, 3.0],
            [0.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.uint32)

    spp = 512
    size = 16
    eps = 0.05

    # Compute analytic gradient
    scene = _make_mesh_scene(vertices, faces, size=size)
    verts_torch = torch.tensor(
        vertices.ravel(), dtype=torch.float32, requires_grad=True
    )

    img = differentiable_render(
        scene, {"shape.vertex_positions": verts_torch}, spp=spp
    )
    # Use RGB L2 loss (avoid alpha which has discontinuous silhouette gradients)
    loss = img[..., :3].pow(2).sum()
    loss.backward()

    analytic_grad = verts_torch.grad.clone()

    # Test z-component of vertex 0 (index 2) and vertex 2 (index 8)
    # Z-perturbation mostly changes shading (smooth), not silhouette
    test_indices = [2, 8]
    fd_spp = spp * 4  # Higher spp for stable FD reference

    for idx in test_indices:
        # f(x + eps)
        verts_plus = vertices.ravel().copy()
        verts_plus[idx] += eps
        scene_plus = _make_mesh_scene(
            verts_plus.reshape(-1, 3), faces, size=size
        )
        with torch.no_grad():
            img_plus = torch.tensor(
                np.array(mi.render(scene_plus, spp=fd_spp)).copy(),
                dtype=torch.float32,
            )
        loss_plus = img_plus[..., :3].pow(2).sum().item()

        # f(x - eps)
        verts_minus = vertices.ravel().copy()
        verts_minus[idx] -= eps
        scene_minus = _make_mesh_scene(
            verts_minus.reshape(-1, 3), faces, size=size
        )
        with torch.no_grad():
            img_minus = torch.tensor(
                np.array(mi.render(scene_minus, spp=fd_spp)).copy(),
                dtype=torch.float32,
            )
        loss_minus = img_minus[..., :3].pow(2).sum().item()

        fd_grad = (loss_plus - loss_minus) / (2.0 * eps)
        ag = analytic_grad[idx].item()

        # Check direction and order of magnitude.
        # Monte Carlo differentiable rendering has inherent variance;
        # rtol=1.0 confirms same sign and order of magnitude.
        if abs(fd_grad) > 1e-4:
            # Same sign check
            assert ag * fd_grad > 0 or abs(ag) < 0.1, (
                f"Index {idx}: sign mismatch analytic={ag:.4f}, fd={fd_grad:.4f}"
            )
            rel_error = abs(ag - fd_grad) / abs(fd_grad)
            assert rel_error < 1.0, (
                f"Index {idx}: analytic={ag:.4f}, fd={fd_grad:.4f}, "
                f"rel_error={rel_error:.3f} (expected < 1.0)"
            )
        else:
            # If FD gradient is negligible, analytic should be small too
            assert abs(ag) < 2.0, (
                f"Index {idx}: fd_grad~0 but analytic={ag:.4f}"
            )
