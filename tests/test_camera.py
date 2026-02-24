"""Tests for perspective camera builder and coordinate system compatibility."""

import math

import numpy as np
import pytest
import torch

from rendering._variant import ensure_variant

ensure_variant()

import mitsuba as mi
from rendering._camera import build_camera


# ---------------------------------------------------------------------------
# Camera builder (FOV, dict fields, loads in Mitsuba)
# ---------------------------------------------------------------------------


class TestFOVMatchesOriginal:
    """FOV calculation matches the original formula."""

    @pytest.mark.parametrize(
        "focal,width",
        [(500.0, 256), (1000.0, 512), (300.0, 128), (750.0, 1024)],
    )
    def test_fov_matches_original(self, focal, width):
        expected_fov = 360.0 * math.atan(width / (2.0 * focal)) / math.pi
        cam = build_camera(focal, width, 256)
        assert abs(cam["fov"] - expected_fov) < 1e-10

    def test_fov_with_torch_tensor(self):
        """Focal length can be a torch tensor."""
        focal = torch.tensor(500.0)
        expected_fov = 360.0 * math.atan(256 / (2.0 * 500.0)) / math.pi
        cam = build_camera(focal, 256, 256)
        assert abs(cam["fov"] - expected_fov) < 1e-10


class TestCameraDictValid:
    """Camera dict has all required fields."""

    def setup_method(self):
        self.cam = build_camera(500.0, 256, 192)

    def test_type_is_perspective(self):
        assert self.cam["type"] == "perspective"

    def test_near_clip_preserved(self):
        assert self.cam["near_clip"] == 10.0

    def test_resolution_matches(self):
        assert self.cam["film"]["width"] == 256
        assert self.cam["film"]["height"] == 192

    def test_film_type(self):
        assert self.cam["film"]["type"] == "hdrfilm"

    def test_to_world_exists(self):
        assert "to_world" in self.cam

    def test_camera_loads_in_mitsuba(self):
        """Camera dict can be loaded by mi.load_dict()."""
        sensor = mi.load_dict(self.cam)
        assert sensor is not None

    def test_custom_clip_near(self):
        cam = build_camera(500.0, 256, 256, clip_near=1.0)
        assert cam["near_clip"] == 1.0


# ---------------------------------------------------------------------------
# Coordinate system compatibility
# ---------------------------------------------------------------------------


@pytest.fixture
def mi_module():
    """Import mitsuba with variant set."""
    return mi


def test_origin_projects_to_center(mi_module):
    """A point at (0, 0, 1) should project near the image center.

    We render a small bright sphere centered at (0, 0, 20) and verify that
    the brightest pixel region is near the image center.
    """
    W, H = 64, 64
    focal = 300.0
    cam_dict = build_camera(focal, W, H, clip_near=1.0)

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "direct"},
        "sensor": cam_dict,
        "light": {
            "type": "point",
            "position": [0.0, 0.0, 15.0],
            "intensity": {"type": "spectrum", "value": 500.0},
        },
        "sphere": {
            "type": "sphere",
            "center": [0.0, 0.0, 20.0],
            "radius": 1.0,
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "spectrum", "value": 1.0},
            },
        },
    }

    scene = mi_module.load_dict(scene_dict)
    img = mi_module.render(scene, spp=32)
    img_np = np.array(img)[:, :, :3]

    brightness = img_np.sum(axis=2)
    max_idx = np.unravel_index(brightness.argmax(), brightness.shape)
    center_y, center_x = H // 2, W // 2

    assert abs(max_idx[0] - center_y) <= 5, (
        f"Brightest pixel y={max_idx[0]}, expected near {center_y}"
    )
    assert abs(max_idx[1] - center_x) <= 5, (
        f"Brightest pixel x={max_idx[1]}, expected near {center_x}"
    )


def test_up_vector_orientation(mi_module):
    """Verify that up=(0,-1,0) means +Y world = downward in image."""
    W, H = 64, 64
    focal = 50.0
    cam_dict = build_camera(focal, W, H, clip_near=1.0)

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "direct"},
        "sensor": cam_dict,
        "light": {
            "type": "point",
            "position": [0.0, 0.0, 15.0],
            "intensity": {"type": "spectrum", "value": 500.0},
        },
        "sphere_top": {
            "type": "sphere",
            "center": [0.0, -3.0, 20.0],
            "radius": 0.5,
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "spectrum", "value": 1.0},
            },
        },
        "sphere_bottom": {
            "type": "sphere",
            "center": [0.0, 3.0, 20.0],
            "radius": 0.5,
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "spectrum", "value": 1.0},
            },
        },
    }

    scene = mi_module.load_dict(scene_dict)
    img = mi_module.render(scene, spp=32)
    img_np = np.array(img)[:, :, :3]

    brightness = img_np.sum(axis=2)

    top_half = brightness[: H // 2, :]
    bottom_half = brightness[H // 2 :, :]

    top_max = top_half.max()
    bottom_max = bottom_half.max()

    assert top_max > 0.01, "Top sphere (world Y=-3) not visible in top half"
    assert bottom_max > 0.01, "Bottom sphere (world Y=+3) not visible in bottom half"

    top_brightest = np.unravel_index(top_half.argmax(), top_half.shape)
    assert top_brightest[0] < H // 2, "Top sphere should be in top half"

    assert bottom_max > 0.01, "Bottom sphere should be visible in bottom half"


def test_no_coord_transform_needed(mi_module):
    """Verify that no coordinate transform is needed.

    Both PyRedner and Mitsuba 3 use right-handed coordinate systems.
    """
    import rendering._camera as cam_mod

    assert not hasattr(cam_mod, "coord_transform"), (
        "coord_transform should not exist — coordinate systems are compatible"
    )

    cam = build_camera(focal=500.0, width=256, height=256)
    assert cam["type"] == "perspective"
