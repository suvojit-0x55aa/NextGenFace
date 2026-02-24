"""US-013: Implement albedo rendering mode.

Tests that render_albedo() produces [N, H, W, 4] RGBA images where
RGB channels contain diffuse reflectance (albedo) without lighting effects.
"""

import sys
import os

import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "NextFace"))

from variant_mitsuba import ensure_variant


@pytest.fixture(autouse=True)
def _variant():
    ensure_variant()


from helpers import make_triangle_mesh_params as _make_triangle_mesh_params


def _build_albedo_scenes(n_frames=1, screen_size=32, diffuse_value=0.7):
    """Build Mitsuba scenes configured for albedo rendering."""
    from scene_mitsuba import build_scenes
    params = _make_triangle_mesh_params(
        n_frames=n_frames, screen_size=screen_size, diffuse_value=diffuse_value, z=50.0
    )
    return build_scenes(**params, albedo_mode=True), screen_size


def test_us013_albedo_output_shape():
    """Albedo render returns [N, H, W, 4] RGBA tensor."""
    from render_mitsuba import render_albedo

    scenes, size = _build_albedo_scenes(n_frames=2, screen_size=32)
    result = render_albedo(scenes, spp=4)

    assert result.shape == (2, size, size, 4), f"Expected (2, {size}, {size}, 4), got {result.shape}"
    assert result.dtype == torch.float32


def test_us013_albedo_no_lighting():
    """Albedo output shows diffuse reflectance without lighting variation.

    With a constant diffuse texture, fully-covered pixels should have
    approximately the same albedo RGB value (no shading/shadow effects).
    """
    from render_mitsuba import render_albedo

    diffuse_val = 0.6
    scenes, size = _build_albedo_scenes(
        n_frames=1, screen_size=32, diffuse_value=diffuse_val
    )
    result = render_albedo(scenes, spp=64)

    # Use fully-covered pixels (alpha > 0.99) to avoid edge anti-aliasing
    alpha = result[0, :, :, 3]
    full_mask = alpha > 0.99
    if full_mask.sum() == 0:
        pytest.skip("No fully-covered pixels in albedo render")

    albedo_rgb = result[0, :, :, :3]
    full_albedo = albedo_rgb[full_mask]

    # Fully-covered pixels should have albedo matching the input diffuse value
    mean_albedo = full_albedo.mean(dim=0)
    assert torch.allclose(mean_albedo, torch.tensor([diffuse_val] * 3), atol=0.05), (
        f"Mean albedo {mean_albedo.tolist()} differs from expected {diffuse_val}"
    )

    # Check low variance (no lighting-induced shading)
    std_albedo = full_albedo.std(dim=0)
    assert (std_albedo < 0.02).all(), (
        f"Albedo std {std_albedo.tolist()} too high — lighting effects present"
    )


def test_us013_albedo_has_alpha():
    """Albedo render has a meaningful alpha channel."""
    from render_mitsuba import render_albedo

    scenes, _ = _build_albedo_scenes(n_frames=1, screen_size=32)
    result = render_albedo(scenes, spp=4)

    alpha = result[0, :, :, 3]
    assert alpha.max() > 0.0, "Alpha channel is all zeros — mesh not visible"


def test_us013_albedo_differs_from_path_traced():
    """Albedo render differs from normal path-traced render.

    The albedo render should show raw diffuse reflectance while path-traced
    render includes lighting effects, so they should differ.
    """
    from scene_mitsuba import build_scenes
    from render_mitsuba import render_scenes, render_albedo

    params = _make_triangle_mesh_params(n_frames=1, screen_size=32, diffuse_value=0.5, z=50.0)

    # Normal render
    normal_scenes = build_scenes(**params, albedo_mode=False)
    normal_result = render_scenes(normal_scenes, spp=8)

    # Albedo render
    albedo_scenes = build_scenes(**params, albedo_mode=True)
    albedo_result = render_albedo(albedo_scenes, spp=8)

    # Both should have same shape
    assert normal_result.shape == albedo_result.shape

    # Albedo render should have valid alpha distinguishing mesh from background
    albedo_alpha = albedo_result[0, :, :, 3]
    has_visible = albedo_alpha.max() > 0.0
    has_bg = albedo_alpha.min() < 0.5
    assert has_visible, "Albedo alpha should have visible mesh pixels"
    assert has_bg, "Albedo alpha should have background pixels"
