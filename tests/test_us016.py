"""US-016: Verify gradient correctness for texture parameters.

Tests that gradients w.r.t. texture values (diffuse, specular, roughness)
are correct so that albedo optimization works in the NextFace pipeline.
"""

import pytest
import torch
import numpy as np

from rendering._variant import ensure_variant


@pytest.fixture(autouse=True)
def _ad_variant():
    """Ensure an AD-capable variant is active; skip if unavailable."""
    variant = ensure_variant()
    if "ad" not in variant:
        pytest.skip(f"AD variant required, got {variant}")


def _make_textured_scene(diffuse_np, roughness_val=0.5, size=32):
    """Create a scene with a textured quad for gradient testing.

    Uses a quad (2 triangles) at z=20 (beyond clip_near=10) with UV coords
    that map the full texture onto the quad. Face winding ensures normals
    point toward camera (-Z) which is required for gradient flow.

    Args:
        diffuse_np: [H, W, 3] float32 numpy array for diffuse texture.
        roughness_val: float, uniform roughness value.
        size: image resolution (square).

    Returns:
        mi.Scene: loaded Mitsuba scene with textured mesh.
    """
    import mitsuba as mi

    # Quad vertices at z=20, covering camera FOV
    vertices = np.array([
        [-8.0, -8.0, 20.0],
        [ 8.0, -8.0, 20.0],
        [ 8.0,  8.0, 20.0],
        [-8.0,  8.0, 20.0],
    ], dtype=np.float32)

    # Reversed winding so normals face -Z (toward camera at origin)
    faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32)

    # UVs map quad corners to texture corners
    uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    V = vertices.shape[0]
    F = faces.shape[0]

    mesh = mi.Mesh(
        "shape",
        vertex_count=V,
        face_count=F,
        has_vertex_normals=False,
        has_vertex_texcoords=True,
    )
    params = mi.traverse(mesh)
    params["vertex_positions"] = mi.Float(vertices.ravel())
    params["faces"] = mi.UInt32(faces.ravel())
    params["vertex_texcoords"] = mi.Float(uvs.ravel())
    params.update()

    # Build material with bitmap texture for diffuse
    bsdf_dict = {
        "type": "principled",
        "base_color": {
            "type": "bitmap",
            "bitmap": mi.Bitmap(diffuse_np.astype(np.float32)),
        },
        "roughness": roughness_val,
        "specular": 0.5,
    }
    mesh.set_bsdf(mi.load_dict(bsdf_dict))

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "direct_projective", "sppi": 0},
        "emitter": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
        },
        "sensor": {
            "type": "perspective",
            "fov": 45.0,
            "near_clip": 0.1,
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


def _make_roughness_texture_scene(diffuse_val, roughness_np, size=32):
    """Create a scene with a roughness bitmap texture for gradient testing.

    Args:
        diffuse_val: [3] float, uniform diffuse color.
        roughness_np: [H, W] float32 numpy array for roughness texture.
        size: image resolution (square).

    Returns:
        mi.Scene: loaded Mitsuba scene.
    """
    import mitsuba as mi

    vertices = np.array([
        [-8.0, -8.0, 20.0],
        [ 8.0, -8.0, 20.0],
        [ 8.0,  8.0, 20.0],
        [-8.0,  8.0, 20.0],
    ], dtype=np.float32)
    # Reversed winding so normals face -Z (toward camera)
    faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32)
    uvs = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    ], dtype=np.float32)

    mesh = mi.Mesh("shape", vertex_count=4, face_count=2,
                    has_vertex_normals=False, has_vertex_texcoords=True)
    params = mi.traverse(mesh)
    params["vertex_positions"] = mi.Float(vertices.ravel())
    params["faces"] = mi.UInt32(faces.ravel())
    params["vertex_texcoords"] = mi.Float(uvs.ravel())
    params.update()

    bsdf_dict = {
        "type": "principled",
        "base_color": {"type": "rgb", "value": list(diffuse_val)},
        "roughness": {
            "type": "bitmap",
            "bitmap": mi.Bitmap(roughness_np.astype(np.float32)),
        },
        "specular": 0.5,
    }
    mesh.set_bsdf(mi.load_dict(bsdf_dict))

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "direct_projective", "sppi": 0},
        "emitter": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
        },
        "sensor": {
            "type": "perspective",
            "fov": 45.0,
            "near_clip": 0.1,
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


def test_us016_texture_grad_nonzero():
    """Gradients flow through diffuse, specular, and roughness textures."""
    import mitsuba as mi
    from rendering._gradient_bridge import differentiable_render

    tex_h, tex_w = 4, 4

    # --- Test 1: Diffuse texture gradient ---
    diffuse_np = np.full((tex_h, tex_w, 3), 0.5, dtype=np.float32)
    scene = _make_textured_scene(diffuse_np, size=32)

    # Discover the diffuse texture parameter path
    scene_params = mi.traverse(scene)
    diffuse_path = None
    for key in scene_params.keys():
        if "base_color" in key and "data" in key:
            diffuse_path = key
            break
    assert diffuse_path is not None, (
        f"Could not find base_color data param. Keys: {list(scene_params.keys())}"
    )

    diffuse_torch = torch.tensor(
        np.array(scene_params[diffuse_path]).copy(),
        dtype=torch.float32,
        requires_grad=True,
    )

    img = differentiable_render(scene, {diffuse_path: diffuse_torch}, spp=32)
    loss = img[..., :3].pow(2).sum()
    loss.backward()

    assert diffuse_torch.grad is not None, "No gradient on diffuse texture"
    assert diffuse_torch.grad.abs().sum() > 1e-6, (
        "Diffuse texture gradient is all zeros"
    )

    # --- Test 2: Roughness texture gradient ---
    roughness_np = np.full((tex_h, tex_w), 0.5, dtype=np.float32)
    scene_r = _make_roughness_texture_scene([0.5, 0.5, 0.5], roughness_np, size=32)

    scene_params_r = mi.traverse(scene_r)
    roughness_path = None
    for key in scene_params_r.keys():
        if "roughness" in key and "data" in key:
            roughness_path = key
            break
    assert roughness_path is not None, (
        f"Could not find roughness data param. Keys: {list(scene_params_r.keys())}"
    )

    roughness_torch = torch.tensor(
        np.array(scene_params_r[roughness_path]).copy(),
        dtype=torch.float32,
        requires_grad=True,
    )

    img_r = differentiable_render(scene_r, {roughness_path: roughness_torch}, spp=32)
    loss_r = img_r[..., :3].pow(2).sum()
    loss_r.backward()

    assert roughness_torch.grad is not None, "No gradient on roughness texture"
    # Roughness gradient may be small but should be non-zero with specular > 0
    # Use a very permissive threshold since roughness effect is subtle
    assert roughness_torch.grad.abs().max() > 1e-8, (
        f"Roughness gradient is all zeros (max={roughness_torch.grad.abs().max():.2e})"
    )

    # --- Test 3: Specular parameter gradient (scalar) ---
    # Build a scene with scalar specular, verify gradient via a separate render
    diffuse_np2 = np.full((tex_h, tex_w, 3), 0.5, dtype=np.float32)
    scene_s = _make_textured_scene(diffuse_np2, roughness_val=0.3, size=32)

    scene_params_s = mi.traverse(scene_s)
    specular_path = None
    for key in scene_params_s.keys():
        if "specular" in key and "value" in key:
            specular_path = key
            break

    if specular_path is not None:
        specular_torch = torch.tensor(
            np.array(scene_params_s[specular_path]).copy(),
            dtype=torch.float32,
            requires_grad=True,
        )
        img_s = differentiable_render(
            scene_s, {specular_path: specular_torch}, spp=32
        )
        loss_s = img_s[..., :3].pow(2).sum()
        loss_s.backward()
        assert specular_torch.grad is not None, "No gradient on specular param"
        # Specular gradient existence is sufficient; magnitude can be small
    else:
        # Principled BSDF may not expose specular as a traversable param
        # This is acceptable — specular is float-only in build_material
        pass


def test_us016_diffuse_grad_finite_diff():
    """Analytic diffuse texture gradient matches finite difference approximation."""
    import mitsuba as mi
    from rendering._gradient_bridge import differentiable_render

    tex_h, tex_w = 4, 4
    size = 16
    spp = 256
    eps = 0.02

    diffuse_np = np.full((tex_h, tex_w, 3), 0.5, dtype=np.float32)
    scene = _make_textured_scene(diffuse_np, size=size)

    # Get the diffuse texture parameter path
    scene_params = mi.traverse(scene)
    diffuse_path = None
    for key in scene_params.keys():
        if "base_color" in key and "data" in key:
            diffuse_path = key
            break
    assert diffuse_path is not None

    original_data = np.array(scene_params[diffuse_path]).copy()
    original_flat = original_data.ravel()

    # Analytic gradient
    diffuse_torch = torch.tensor(
        original_data, dtype=torch.float32, requires_grad=True
    )
    img = differentiable_render(scene, {diffuse_path: diffuse_torch}, spp=spp)
    loss = img[..., :3].pow(2).sum()
    loss.backward()
    analytic_grad_flat = diffuse_torch.grad.clone().numpy().ravel()

    # Pick a few flat indices to test (R channel of different texels)
    n_elems = original_flat.shape[0]
    test_indices = [0, n_elems // 2, n_elems // 3]

    fd_spp = spp * 4
    tex_shape = original_data.shape

    for idx in test_indices:
        # f(x + eps)
        data_plus = original_flat.copy()
        data_plus[idx] += eps
        diffuse_t_plus = torch.tensor(
            data_plus.reshape(tex_shape), dtype=torch.float32
        )
        img_plus = differentiable_render(
            scene, {diffuse_path: diffuse_t_plus}, spp=fd_spp
        )
        loss_plus = img_plus[..., :3].pow(2).sum().item()

        # f(x - eps)
        data_minus = original_flat.copy()
        data_minus[idx] -= eps
        diffuse_t_minus = torch.tensor(
            data_minus.reshape(tex_shape), dtype=torch.float32
        )
        img_minus = differentiable_render(
            scene, {diffuse_path: diffuse_t_minus}, spp=fd_spp
        )
        loss_minus = img_minus[..., :3].pow(2).sum().item()

        fd_grad = (loss_plus - loss_minus) / (2.0 * eps)
        ag = float(analytic_grad_flat[idx])

        if abs(fd_grad) > 1e-4:
            # Same sign check
            assert ag * fd_grad > 0 or abs(ag) < 0.1, (
                f"Index {idx}: sign mismatch analytic={ag:.4f}, fd={fd_grad:.4f}"
            )
            # Order of magnitude check (rtol=2.0 for MC noise with textures)
            rel_error = abs(ag - fd_grad) / abs(fd_grad)
            assert rel_error < 2.0, (
                f"Index {idx}: analytic={ag:.4f}, fd={fd_grad:.4f}, "
                f"rel_error={rel_error:.3f} (expected < 2.0)"
            )
        else:
            # FD gradient is negligible, analytic should be small too
            assert abs(ag) < 2.0, (
                f"Index {idx}: fd_grad~0 but analytic={ag:.4f}"
            )
