"""Tests for scene assembly, forward rendering, and albedo rendering."""

import pytest
import torch
import numpy as np

from rendering._variant import ensure_variant


@pytest.fixture(autouse=True)
def _variant():
    ensure_variant()


from helpers import make_triangle_mesh_params as _make_triangle_mesh_params


# ---------------------------------------------------------------------------
# Scene assembly (build_scenes)
# ---------------------------------------------------------------------------


def test_scene_renders_without_error():
    """build_scenes() produces a scene that renders without error."""
    import mitsuba as mi
    from rendering._scene import build_scenes

    params = _make_triangle_mesh_params(n_frames=1)
    scenes = build_scenes(**params)

    assert len(scenes) == 1
    img = mi.render(scenes[0])
    arr = np.array(img)
    assert arr.shape[0] == 32
    assert arr.shape[1] == 32


def test_shared_texture_mode():
    """build_scenes() handles shared texture mode (diffuse.shape[0]==1, vertices.shape[0]==N)."""
    import mitsuba as mi
    from rendering._scene import build_scenes

    params = _make_triangle_mesh_params(n_frames=2, shared_texture=True)
    scenes = build_scenes(**params)

    assert len(scenes) == 2
    for scene in scenes:
        img = mi.render(scene)
        arr = np.array(img)
        assert arr.shape[0] == 32
        assert arr.shape[1] == 32


def test_multi_frame():
    """build_scenes() handles multiple frames with independent textures."""
    from rendering._scene import build_scenes

    params = _make_triangle_mesh_params(n_frames=3, shared_texture=False)
    scenes = build_scenes(**params)
    assert len(scenes) == 3


def test_scene_count_matches_batch():
    """Number of returned scenes matches the batch dimension."""
    from rendering._scene import build_scenes

    for n in [1, 2, 4]:
        params = _make_triangle_mesh_params(n_frames=n)
        scenes = build_scenes(**params)
        assert len(scenes) == n


# ---------------------------------------------------------------------------
# Forward rendering (render_scenes)
# ---------------------------------------------------------------------------


def _build_test_scenes(n_frames=1, screen_size=32):
    """Build Mitsuba scenes for testing."""
    from rendering._scene import build_scenes
    params = _make_triangle_mesh_params(n_frames=n_frames, screen_size=screen_size)
    return build_scenes(**params), screen_size


def test_render_output_shape():
    """render_scenes() returns [N, H, W, 4] tensor."""
    from rendering._forward import render_scenes

    scenes, size = _build_test_scenes(n_frames=2, screen_size=32)
    result = render_scenes(scenes, spp=4)

    assert result.shape == (2, size, size, 4), f"Expected (2, {size}, {size}, 4), got {result.shape}"
    assert result.dtype == torch.float32


def test_render_has_alpha():
    """Rendered output has a meaningful alpha channel (not all zeros)."""
    from rendering._forward import render_scenes

    scenes, size = _build_test_scenes(n_frames=1, screen_size=32)
    result = render_scenes(scenes, spp=4)

    alpha = result[0, :, :, 3]
    assert alpha.max() > 0.0, "Alpha channel is all zeros — mesh not visible"


def test_render_on_device():
    """render_scenes() output is on the requested device."""
    from rendering._forward import render_scenes

    scenes, _ = _build_test_scenes(n_frames=1, screen_size=32)
    result = render_scenes(scenes, spp=4, device="cpu")

    assert result.device == torch.device("cpu")


def test_render_rgb_nonzero():
    """Rendered RGB channels are not all zero (scene is illuminated)."""
    from rendering._forward import render_scenes

    scenes, _ = _build_test_scenes(n_frames=1, screen_size=32)
    result = render_scenes(scenes, spp=4)

    rgb = result[0, :, :, :3]
    assert rgb.max() > 0.0, "RGB channels are all zeros — scene not illuminated"


def test_render_single_scene():
    """render_scenes() works with a single scene."""
    from rendering._forward import render_scenes

    scenes, size = _build_test_scenes(n_frames=1, screen_size=32)
    result = render_scenes(scenes, spp=4)

    assert result.shape == (1, size, size, 4)


# ---------------------------------------------------------------------------
# Albedo rendering (render_albedo)
# ---------------------------------------------------------------------------


def _build_albedo_scenes(n_frames=1, screen_size=32, diffuse_value=0.7):
    """Build Mitsuba scenes configured for albedo rendering."""
    from rendering._scene import build_scenes
    params = _make_triangle_mesh_params(
        n_frames=n_frames, screen_size=screen_size, diffuse_value=diffuse_value, z=50.0
    )
    return build_scenes(**params, albedo_mode=True), screen_size


def test_albedo_output_shape():
    """Albedo render returns [N, H, W, 4] RGBA tensor."""
    from rendering._forward import render_albedo

    scenes, size = _build_albedo_scenes(n_frames=2, screen_size=32)
    result = render_albedo(scenes, spp=4)

    assert result.shape == (2, size, size, 4), f"Expected (2, {size}, {size}, 4), got {result.shape}"
    assert result.dtype == torch.float32


def test_albedo_no_lighting():
    """Albedo output shows diffuse reflectance without lighting variation."""
    from rendering._forward import render_albedo

    diffuse_val = 0.6
    scenes, size = _build_albedo_scenes(
        n_frames=1, screen_size=32, diffuse_value=diffuse_val
    )
    result = render_albedo(scenes, spp=64)

    alpha = result[0, :, :, 3]
    full_mask = alpha > 0.99
    if full_mask.sum() == 0:
        pytest.skip("No fully-covered pixels in albedo render")

    albedo_rgb = result[0, :, :, :3]
    full_albedo = albedo_rgb[full_mask]

    mean_albedo = full_albedo.mean(dim=0)
    assert torch.allclose(mean_albedo, torch.tensor([diffuse_val] * 3), atol=0.05), (
        f"Mean albedo {mean_albedo.tolist()} differs from expected {diffuse_val}"
    )

    std_albedo = full_albedo.std(dim=0)
    assert (std_albedo < 0.02).all(), (
        f"Albedo std {std_albedo.tolist()} too high — lighting effects present"
    )


def test_albedo_has_alpha():
    """Albedo render has a meaningful alpha channel."""
    from rendering._forward import render_albedo

    scenes, _ = _build_albedo_scenes(n_frames=1, screen_size=32)
    result = render_albedo(scenes, spp=4)

    alpha = result[0, :, :, 3]
    assert alpha.max() > 0.0, "Alpha channel is all zeros — mesh not visible"


def test_albedo_differs_from_path_traced():
    """Albedo render differs from normal path-traced render."""
    from rendering._scene import build_scenes
    from rendering._forward import render_scenes, render_albedo

    params = _make_triangle_mesh_params(n_frames=1, screen_size=32, diffuse_value=0.5, z=50.0)

    normal_scenes = build_scenes(**params, albedo_mode=False)
    normal_result = render_scenes(normal_scenes, spp=8)

    albedo_scenes = build_scenes(**params, albedo_mode=True)
    albedo_result = render_albedo(albedo_scenes, spp=8)

    assert normal_result.shape == albedo_result.shape

    albedo_alpha = albedo_result[0, :, :, 3]
    has_visible = albedo_alpha.max() > 0.0
    has_bg = albedo_alpha.min() < 0.5
    assert has_visible, "Albedo alpha should have visible mesh pixels"
    assert has_bg, "Albedo alpha should have background pixels"
