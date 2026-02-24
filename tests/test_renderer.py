"""Tests for Renderer class: attributes, setupCamera, build+render, albedo."""

import math

import torch
import numpy as np
import pytest

from rendering._variant import ensure_variant

ensure_variant()
import mitsuba as mi

from rendering.renderer import Renderer

from helpers import make_single_triangle as _make_single_triangle
from helpers import make_textures as _make_textures
from helpers import make_envmap as _make_envmap


# ---------------------------------------------------------------------------
# Constructor and attributes
# ---------------------------------------------------------------------------

class TestRendererAttributes:
    def test_constructor_sets_attributes(self):
        r = Renderer(32, 1, "cpu")
        assert r.samples == 32
        assert r.bounces == 1
        assert r.device == torch.device("cpu")
        assert r.clip_near == 10.0
        assert r.counter == 0

    def test_default_screen_dimensions(self):
        r = Renderer(16, 1, "cpu")
        assert r.screenWidth == 256
        assert r.screenHeight == 256

    def test_up_vector(self):
        r = Renderer(16, 1, "cpu")
        expected = torch.tensor([0.0, -1.0, 0.0])
        assert torch.allclose(r.upVector, expected)

    def test_screen_dimensions_mutable(self):
        r = Renderer(16, 1, "cpu")
        r.screenWidth = 128
        r.screenHeight = 64
        assert r.screenWidth == 128
        assert r.screenHeight == 64


# ---------------------------------------------------------------------------
# setupCamera
# ---------------------------------------------------------------------------

class TestSetupCamera:
    def test_fov_matches_formula(self):
        r = Renderer(16, 1, "cpu")
        focal = 500.0
        width = 256
        height = 256
        fov = r.setupCamera(focal, width, height)
        expected = 360.0 * math.atan(width / (2.0 * focal)) / math.pi
        assert abs(fov.item() - expected) < 1e-5

    def test_returns_tensor(self):
        r = Renderer(16, 1, "cpu")
        fov = r.setupCamera(300.0, 128, 128)
        assert isinstance(fov, torch.Tensor)


# ---------------------------------------------------------------------------
# buildScenes + render (single triangle)
# ---------------------------------------------------------------------------

class TestBuildAndRender:
    def test_single_triangle_render(self):
        """Render a single triangle — exercises buildScenes + render (no grad)."""
        r = Renderer(4, 1, "cpu")
        r.screenWidth = 32
        r.screenHeight = 32

        verts, indices, normals, uvs = _make_single_triangle()
        diffuse, specular, roughness = _make_textures()
        envmap = _make_envmap()
        focal = torch.tensor([300.0])

        verts = verts.unsqueeze(0)
        normals = normals.unsqueeze(0)

        scenes = r.buildScenes(verts, indices, normals, uvs,
                               diffuse, specular, roughness, focal, envmap)
        assert len(scenes) == 1

        with torch.no_grad():
            images = r.render(scenes)
        assert images.shape == (1, 32, 32, 4)
        assert r.counter == 1

    def test_counter_increments(self):
        """Render twice -> counter == 2."""
        r = Renderer(4, 1, "cpu")
        r.screenWidth = 16
        r.screenHeight = 16

        verts, indices, normals, uvs = _make_single_triangle()
        diffuse, specular, roughness = _make_textures()
        envmap = _make_envmap()
        focal = torch.tensor([300.0])

        verts = verts.unsqueeze(0)
        normals = normals.unsqueeze(0)

        scenes = r.buildScenes(verts, indices, normals, uvs,
                               diffuse, specular, roughness, focal, envmap)
        with torch.no_grad():
            r.render(scenes)
            r.render(scenes)
        assert r.counter == 2

    def test_zero_roughness(self):
        """Edge case: roughness near zero should not crash."""
        r = Renderer(4, 1, "cpu")
        r.screenWidth = 16
        r.screenHeight = 16

        verts, indices, normals, uvs = _make_single_triangle()
        diffuse, specular, _ = _make_textures()
        roughness = torch.full((1, 4, 4, 1), 1e-6, dtype=torch.float32)
        envmap = _make_envmap()
        focal = torch.tensor([300.0])

        verts = verts.unsqueeze(0)
        normals = normals.unsqueeze(0)

        scenes = r.buildScenes(verts, indices, normals, uvs,
                               diffuse, specular, roughness, focal, envmap)
        with torch.no_grad():
            images = r.render(scenes)
        assert images.shape == (1, 16, 16, 4)
        assert not torch.isnan(images).any()


# ---------------------------------------------------------------------------
# renderAlbedo
# ---------------------------------------------------------------------------

class TestRenderAlbedo:
    def test_albedo_with_cached_params(self):
        """renderAlbedo uses cached params to rebuild in albedo mode."""
        r = Renderer(4, 1, "cpu")
        r.screenWidth = 16
        r.screenHeight = 16

        verts, indices, normals, uvs = _make_single_triangle()
        diffuse, specular, roughness = _make_textures()
        envmap = _make_envmap()
        focal = torch.tensor([300.0])

        verts = verts.unsqueeze(0)
        normals = normals.unsqueeze(0)

        scenes = r.buildScenes(verts, indices, normals, uvs,
                               diffuse, specular, roughness, focal, envmap)
        albedo = r.renderAlbedo(scenes)
        assert albedo.shape == (1, 16, 16, 4)

    def test_albedo_without_cached_params(self):
        """renderAlbedo fallback: no cached params, use scenes directly."""
        r = Renderer(4, 1, "cpu")
        r.screenWidth = 16
        r.screenHeight = 16

        from rendering._scene import build_scenes

        verts, indices, normals, uvs = _make_single_triangle()
        diffuse, specular, roughness = _make_textures()
        envmap = _make_envmap()
        focal = torch.tensor([300.0])

        verts = verts.unsqueeze(0)
        normals = normals.unsqueeze(0)

        albedo_scenes = build_scenes(
            verts, indices, normals, uvs, diffuse, specular,
            roughness, focal, envmap,
            screen_width=r.screenWidth, screen_height=r.screenHeight,
            samples=r.samples, bounces=r.bounces, albedo_mode=True,
        )

        albedo = r.renderAlbedo(albedo_scenes)
        assert albedo.shape == (1, 16, 16, 4)
