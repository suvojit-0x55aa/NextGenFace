"""US-014: DrJit-PyTorch gradient bridge.

Tests that gradients flow from Mitsuba rendered images back to PyTorch
tensors via custom autograd.Function, enabling optimization with torch.optim.
"""

import sys
import os

import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "NextFace"))

from variant_mitsuba import ensure_variant


@pytest.fixture(autouse=True)
def _ad_variant():
    """Ensure an AD-capable variant is active; skip if unavailable."""
    variant = ensure_variant()
    if "ad" not in variant:
        pytest.skip(f"AD variant required, got {variant}")


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


def test_us014_gradient_flows_to_params():
    """Gradients flow from rendered image back to scene parameters."""
    from gradient_bridge import differentiable_render

    scene = _make_color_scene(color=(0.5, 0.5, 0.5), size=16)

    color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, requires_grad=True)

    img = differentiable_render(
        scene, {"shape.bsdf.reflectance.value": color}, spp=8
    )

    loss = img.pow(2).sum()
    loss.backward()

    assert color.grad is not None, "No gradient on color parameter"
    assert color.grad.abs().sum() > 0, "Gradient is zero"


def test_us014_color_optimization_converges():
    """Optimize a diffuse color to match a target rendering."""
    import mitsuba as mi
    from gradient_bridge import differentiable_render

    target_rgb = [0.8, 0.2, 0.1]
    size = 16
    spp = 16

    # Render target image (no grad needed)
    target_scene = _make_color_scene(color=target_rgb, size=size)
    with torch.no_grad():
        target = torch.tensor(
            np.array(mi.render(target_scene, spp=spp * 4)).copy(),
            dtype=torch.float32,
        )

    # Create scene with initial (wrong) color
    scene = _make_color_scene(color=(0.5, 0.5, 0.5), size=size)

    # Optimize color to match target
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
