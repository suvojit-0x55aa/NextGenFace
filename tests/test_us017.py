"""US-017: Verify new Renderer class has matching API."""

import inspect

import pytest
import torch

from rendering.renderer import Renderer


class TestRendererAPICompatible:
    """Verify the Renderer class matches the original PyRedner-based API."""

    def test_constructor_signature(self):
        """Constructor accepts (samples, bounces, device)."""
        sig = inspect.signature(Renderer.__init__)
        params = list(sig.parameters.keys())
        assert params == ['self', 'samples', 'bounces', 'device']

    def test_constructor_creates_instance(self):
        r = Renderer(8, 1, 'cpu')
        assert r is not None

    def test_buildScenes_signature(self):
        """buildScenes has same parameter names as original."""
        sig = inspect.signature(Renderer.buildScenes)
        params = list(sig.parameters.keys())
        assert params == [
            'self', 'vertices', 'indices', 'normal', 'uv',
            'diffuse', 'specular', 'roughness', 'focal', 'envMap'
        ]

    def test_render_method_exists(self):
        r = Renderer(8, 1, 'cpu')
        assert callable(r.render)

    def test_renderAlbedo_method_exists(self):
        r = Renderer(8, 1, 'cpu')
        assert callable(r.renderAlbedo)

    def test_setupCamera_method_exists(self):
        r = Renderer(8, 1, 'cpu')
        assert callable(r.setupCamera)


class TestRendererAttributes:
    """Verify the Renderer has the same public attributes as original."""

    def test_samples(self):
        r = Renderer(16, 2, 'cpu')
        assert r.samples == 16

    def test_bounces(self):
        r = Renderer(16, 2, 'cpu')
        assert r.bounces == 2

    def test_device(self):
        r = Renderer(8, 1, 'cpu')
        assert r.device == torch.device('cpu')

    def test_screenWidth_default(self):
        r = Renderer(8, 1, 'cpu')
        assert r.screenWidth == 256

    def test_screenHeight_default(self):
        r = Renderer(8, 1, 'cpu')
        assert r.screenHeight == 256

    def test_screenWidth_settable(self):
        r = Renderer(8, 1, 'cpu')
        r.screenWidth = 512
        assert r.screenWidth == 512

    def test_screenHeight_settable(self):
        r = Renderer(8, 1, 'cpu')
        r.screenHeight = 1024
        assert r.screenHeight == 1024

    def test_clip_near(self):
        r = Renderer(8, 1, 'cpu')
        assert r.clip_near == 10.0

    def test_counter_initial(self):
        r = Renderer(8, 1, 'cpu')
        assert r.counter == 0

    def test_upVector(self):
        r = Renderer(8, 1, 'cpu')
        expected = torch.tensor([0.0, -1.0, 0.0])
        assert torch.allclose(r.upVector, expected)


class TestSetupCamera:
    """Test setupCamera returns correct FOV."""

    def test_fov_value(self):
        import math
        r = Renderer(8, 1, 'cpu')
        focal = 500.0
        width = 256
        height = 256
        result = r.setupCamera(focal, width, height)
        expected = 360.0 * math.atan(width / (2.0 * focal)) / math.pi
        assert abs(float(result) - expected) < 1e-5


class TestBuildScenesIntegration:
    """Integration tests for buildScenes + render."""

    @pytest.fixture
    def renderer(self):
        r = Renderer(4, 1, 'cpu')
        r.screenWidth = 32
        r.screenHeight = 32
        return r

    @pytest.fixture
    def simple_scene_params(self):
        """Minimal scene parameters for a single triangle beyond clip_near."""
        vertices = torch.tensor([[[0.0, -1.0, 50.0],
                                   [1.0, 1.0, 50.0],
                                   [-1.0, 1.0, 50.0]]], dtype=torch.float32)
        indices = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        normals = torch.tensor([[[0.0, 0.0, -1.0],
                                  [0.0, 0.0, -1.0],
                                  [0.0, 0.0, -1.0]]], dtype=torch.float32)
        uvs = torch.tensor([[0.5, 0.0], [1.0, 1.0], [0.0, 1.0]],
                           dtype=torch.float32)
        diffuse = torch.full((1, 4, 4, 3), 0.5, dtype=torch.float32)
        specular = torch.full((1, 4, 4, 3), 0.04, dtype=torch.float32)
        roughness = torch.full((1, 4, 4, 1), 0.5, dtype=torch.float32)
        focal = torch.tensor([50.0], dtype=torch.float32)
        envmap = torch.full((1, 8, 16, 3), 1.0, dtype=torch.float32)
        return (vertices, indices, normals, uvs, diffuse, specular,
                roughness, focal, envmap)

    def test_buildScenes_returns_list(self, renderer, simple_scene_params):
        scenes = renderer.buildScenes(*simple_scene_params)
        assert isinstance(scenes, list)
        assert len(scenes) == 1

    def test_render_output_shape(self, renderer, simple_scene_params):
        scenes = renderer.buildScenes(*simple_scene_params)
        images = renderer.render(scenes)
        assert images.shape == (1, 32, 32, 4)

    def test_render_increments_counter(self, renderer, simple_scene_params):
        scenes = renderer.buildScenes(*simple_scene_params)
        assert renderer.counter == 0
        renderer.render(scenes)
        assert renderer.counter == 1
        renderer.render(scenes)
        assert renderer.counter == 2

    def test_renderAlbedo_output_shape(self, renderer, simple_scene_params):
        scenes = renderer.buildScenes(*simple_scene_params)
        images = renderer.renderAlbedo(scenes)
        assert images.shape == (1, 32, 32, 4)

    def test_render_on_device(self, renderer, simple_scene_params):
        scenes = renderer.buildScenes(*simple_scene_params)
        images = renderer.render(scenes)
        assert images.device == torch.device('cpu')
