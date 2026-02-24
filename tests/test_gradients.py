"""Tests for gradient bridge, vertex gradients, and texture gradients."""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_color_scene(color=(0.5, 0.5, 0.5), size=16):
    """Create a simple scene with a colored sphere visible to the camera."""
    import mitsuba as mi

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "direct"},
        "emitter": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
        },
        "sensor": {
            "type": "perspective",
            "fov": 90.0,
            "near_clip": 0.1,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0, 0, 0], target=[0, 0, 1], up=[0, 1, 0]
            ),
            "film": {
                "type": "hdrfilm",
                "width": size,
                "height": size,
                "pixel_format": "rgba",
            },
            "sampler": {"type": "independent", "sample_count": 4},
        },
        "shape": {
            "type": "sphere",
            "center": [0, 0, 5],
            "radius": 2.0,
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": list(color)},
            },
        },
    }
    return mi.load_dict(scene_dict)


def _make_mesh_scene(vertices_np, faces_np, size=32, near_clip=0.1):
    """Create a scene with a custom triangle mesh."""
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


def _make_textured_scene(diffuse_np, roughness_val=0.5, size=32):
    """Create a scene with a textured quad for gradient testing."""
    import mitsuba as mi

    vertices = np.array([
        [-8.0, -8.0, 20.0],
        [ 8.0, -8.0, 20.0],
        [ 8.0,  8.0, 20.0],
        [-8.0,  8.0, 20.0],
    ], dtype=np.float32)

    faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32)

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
    """Create a scene with a roughness bitmap texture for gradient testing."""
    import mitsuba as mi

    vertices = np.array([
        [-8.0, -8.0, 20.0],
        [ 8.0, -8.0, 20.0],
        [ 8.0,  8.0, 20.0],
        [-8.0,  8.0, 20.0],
    ], dtype=np.float32)
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


# ---------------------------------------------------------------------------
# Gradient bridge: color param gradients
# ---------------------------------------------------------------------------


def test_gradient_flows_to_params():
    """Gradients flow from rendered image back to scene parameters."""
    from rendering._gradient_bridge import differentiable_render

    scene = _make_color_scene(color=(0.5, 0.5, 0.5), size=16)

    color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, requires_grad=True)

    img = differentiable_render(
        scene, {"shape.bsdf.reflectance.value": color}, spp=8
    )

    loss = img.pow(2).sum()
    loss.backward()

    assert color.grad is not None, "No gradient on color parameter"
    assert color.grad.abs().sum() > 0, "Gradient is zero"


def test_color_optimization_converges():
    """Optimize a diffuse color to match a target rendering."""
    import mitsuba as mi
    from rendering._gradient_bridge import differentiable_render

    target_rgb = [0.8, 0.2, 0.1]
    size = 16
    spp = 16

    target_scene = _make_color_scene(color=target_rgb, size=size)
    with torch.no_grad():
        target = torch.tensor(
            np.array(mi.render(target_scene, spp=spp * 4)).copy(),
            dtype=torch.float32,
        )

    scene = _make_color_scene(color=(0.5, 0.5, 0.5), size=size)

    color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([color], lr=0.05)

    initial_loss = None
    final_loss = None

    for step in range(30):
        optimizer.zero_grad()

        img = differentiable_render(
            scene, {"shape.bsdf.reflectance.value": color}, spp=spp
        )
        loss = (img - target).pow(2).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            color.clamp_(0.0, 1.0)

        if step == 0:
            initial_loss = loss.item()
        if step == 29:
            final_loss = loss.item()

    assert final_loss < initial_loss * 0.5, (
        f"Loss did not decrease enough: {initial_loss:.6f} -> {final_loss:.6f}"
    )


# ---------------------------------------------------------------------------
# Vertex position gradients
# ---------------------------------------------------------------------------


def test_vertex_grad_nonzero():
    """Gradients w.r.t. vertex positions are non-zero for visible vertices."""
    from rendering._gradient_bridge import differentiable_render

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

    for i in range(3):
        assert grad[i].abs().sum() > 1e-6, (
            f"Vertex {i} gradient is zero but vertex is visible"
        )

    for i in range(3, 6):
        assert grad[i].abs().sum() < 1e-4, (
            f"Vertex {i} gradient is non-zero ({grad[i]}) but vertex is behind camera"
        )


def test_vertex_grad_finite_diff():
    """Analytic vertex gradient matches finite difference approximation."""
    import mitsuba as mi
    from rendering._gradient_bridge import differentiable_render

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

    scene = _make_mesh_scene(vertices, faces, size=size)
    verts_torch = torch.tensor(
        vertices.ravel(), dtype=torch.float32, requires_grad=True
    )

    img = differentiable_render(
        scene, {"shape.vertex_positions": verts_torch}, spp=spp
    )
    loss = img[..., :3].pow(2).sum()
    loss.backward()

    analytic_grad = verts_torch.grad.clone()

    test_indices = [2, 8]
    fd_spp = spp * 4

    for idx in test_indices:
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

        if abs(fd_grad) > 1e-4:
            assert ag * fd_grad > 0 or abs(ag) < 0.1, (
                f"Index {idx}: sign mismatch analytic={ag:.4f}, fd={fd_grad:.4f}"
            )
            rel_error = abs(ag - fd_grad) / abs(fd_grad)
            assert rel_error < 1.0, (
                f"Index {idx}: analytic={ag:.4f}, fd={fd_grad:.4f}, "
                f"rel_error={rel_error:.3f} (expected < 1.0)"
            )
        else:
            assert abs(ag) < 2.0, (
                f"Index {idx}: fd_grad~0 but analytic={ag:.4f}"
            )


# ---------------------------------------------------------------------------
# Texture gradients
# ---------------------------------------------------------------------------


def test_texture_grad_nonzero():
    """Gradients flow through diffuse, specular, and roughness textures."""
    import mitsuba as mi
    from rendering._gradient_bridge import differentiable_render

    tex_h, tex_w = 4, 4

    # --- Test 1: Diffuse texture gradient ---
    diffuse_np = np.full((tex_h, tex_w, 3), 0.5, dtype=np.float32)
    scene = _make_textured_scene(diffuse_np, size=32)

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
    assert roughness_torch.grad.abs().max() > 1e-8, (
        f"Roughness gradient is all zeros (max={roughness_torch.grad.abs().max():.2e})"
    )

    # --- Test 3: Specular parameter gradient (scalar) ---
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


def test_diffuse_grad_finite_diff():
    """Analytic diffuse texture gradient matches finite difference approximation."""
    import mitsuba as mi
    from rendering._gradient_bridge import differentiable_render

    tex_h, tex_w = 4, 4
    size = 16
    spp = 256
    eps = 0.02

    diffuse_np = np.full((tex_h, tex_w, 3), 0.5, dtype=np.float32)
    scene = _make_textured_scene(diffuse_np, size=size)

    scene_params = mi.traverse(scene)
    diffuse_path = None
    for key in scene_params.keys():
        if "base_color" in key and "data" in key:
            diffuse_path = key
            break
    assert diffuse_path is not None

    original_data = np.array(scene_params[diffuse_path]).copy()
    original_flat = original_data.ravel()

    diffuse_torch = torch.tensor(
        original_data, dtype=torch.float32, requires_grad=True
    )
    img = differentiable_render(scene, {diffuse_path: diffuse_torch}, spp=spp)
    loss = img[..., :3].pow(2).sum()
    loss.backward()
    analytic_grad_flat = diffuse_torch.grad.clone().numpy().ravel()

    n_elems = original_flat.shape[0]
    test_indices = [0, n_elems // 2, n_elems // 3]

    fd_spp = spp * 4
    tex_shape = original_data.shape

    for idx in test_indices:
        data_plus = original_flat.copy()
        data_plus[idx] += eps
        diffuse_t_plus = torch.tensor(
            data_plus.reshape(tex_shape), dtype=torch.float32
        )
        img_plus = differentiable_render(
            scene, {diffuse_path: diffuse_t_plus}, spp=fd_spp
        )
        loss_plus = img_plus[..., :3].pow(2).sum().item()

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
            assert ag * fd_grad > 0 or abs(ag) < 0.1, (
                f"Index {idx}: sign mismatch analytic={ag:.4f}, fd={fd_grad:.4f}"
            )
            rel_error = abs(ag - fd_grad) / abs(fd_grad)
            assert rel_error < 2.0, (
                f"Index {idx}: analytic={ag:.4f}, fd={fd_grad:.4f}, "
                f"rel_error={rel_error:.3f} (expected < 2.0)"
            )
        else:
            assert abs(ag) < 2.0, (
                f"Index {idx}: fd_grad~0 but analytic={ag:.4f}"
            )
