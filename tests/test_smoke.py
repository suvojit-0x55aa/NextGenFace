"""Tests for US-003: Smoke test for mitsuba import and rendering."""

import numpy as np


def test_mitsuba_imports():
    """Verify mitsuba and drjit can be imported without error."""
    import mitsuba as mi
    import drjit as dr

    assert hasattr(mi, "set_variant")
    assert hasattr(mi, "load_dict")
    assert hasattr(dr, "zeros")


def test_minimal_render():
    """Render a 32x32 scene with a single sphere, verify output shape is (32, 32, 3)."""
    import mitsuba as mi
    from NextFace.variant_mitsuba import ensure_variant

    ensure_variant()

    scene = mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "direct"},
            "sensor": {
                "type": "perspective",
                "fov": 45,
                "film": {
                    "type": "hdrfilm",
                    "width": 32,
                    "height": 32,
                    "pixel_format": "rgb",
                },
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=[0, 0, -3],
                    target=[0, 0, 0],
                    up=[0, 1, 0],
                ),
            },
            "light": {
                "type": "point",
                "position": [0, 2, -2],
                "intensity": {"type": "spectrum", "value": 50.0},
            },
            "sphere": {
                "type": "sphere",
                "center": [0, 0, 0],
                "radius": 1.0,
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "spectrum", "value": 0.5},
                },
            },
        }
    )

    image = mi.render(scene)
    img_np = np.array(image)

    assert img_np.shape == (32, 32, 3), f"Expected (32, 32, 3), got {img_np.shape}"


def test_minimal_render_nonzero():
    """Rendered sphere image should have non-zero pixel values (not all black)."""
    import mitsuba as mi
    from NextFace.variant_mitsuba import ensure_variant

    ensure_variant()

    scene = mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "direct"},
            "sensor": {
                "type": "perspective",
                "fov": 45,
                "film": {
                    "type": "hdrfilm",
                    "width": 32,
                    "height": 32,
                    "pixel_format": "rgb",
                },
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=[0, 0, -3],
                    target=[0, 0, 0],
                    up=[0, 1, 0],
                ),
            },
            "light": {
                "type": "point",
                "position": [0, 2, -2],
                "intensity": {"type": "spectrum", "value": 50.0},
            },
            "sphere": {
                "type": "sphere",
                "center": [0, 0, 0],
                "radius": 1.0,
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {"type": "spectrum", "value": 0.5},
                },
            },
        }
    )

    image = mi.render(scene)
    img_np = np.array(image)

    assert img_np.max() > 0.0, "Rendered image is all black — rendering may have failed"
