"""Tests for US-004: Perspective camera builder for Mitsuba."""

import math
import pytest

from NextFace.mitsuba_variant import ensure_variant

ensure_variant()

import mitsuba as mi
from NextFace.camera_mitsuba import build_camera


class TestFOVMatchesOriginal:
    """test_us004_fov_matches_original: FOV calculation matches the original formula."""

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
        import torch

        focal = torch.tensor(500.0)
        expected_fov = 360.0 * math.atan(256 / (2.0 * 500.0)) / math.pi
        cam = build_camera(focal, 256, 256)
        assert abs(cam["fov"] - expected_fov) < 1e-10


class TestCameraDictValid:
    """test_us004_camera_dict_valid: Camera dict has all required fields."""

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
