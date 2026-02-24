"""End-to-end tests: single image and shared identity multi-image reconstruction."""

import os
import shutil
import sys

import pytest
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
H5_MODEL_PATH = os.path.join(
    DATA_DIR, "baselMorphableModel", "model2017-1_face12_nomouth.h5"
)
INPUT_IMAGE = os.path.join(DATA_DIR, "input", "s1.png")
has_model = os.path.isfile(H5_MODEL_PATH)
has_input = os.path.isfile(INPUT_IMAGE)


# ---------------------------------------------------------------------------
# E2E single image reconstruction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not has_model, reason="Basel Face Model .h5 not available")
@pytest.mark.skipif(not has_input, reason="Input image s1.png not available")
class TestE2ESingleImage:
    """End-to-end single image reconstruction test."""

    def test_e2e_single_image(self, tmp_path):
        """Run the full pipeline on s1.png and verify outputs."""
        from optim.config import Config
        from optim.optimizer import Optimizer

        config = Config()
        config.device = "cpu"
        config.path = os.path.join(DATA_DIR, "baselMorphableModel")
        config.textureResolution = 256
        config.maxResolution = 64
        config.iterStep1 = 5
        config.iterStep2 = 5
        config.iterStep3 = 5
        config.rtSamples = 4
        config.rtTrainingSamples = 4
        config.debugFrequency = 0
        config.saveIntermediateStage = False
        config.verbose = False
        config.lamdmarksDetectorType = "fan"
        config.optimizeFocalLength = False
        config.saveExr = False

        output_dir = str(tmp_path / "output" / "s1.png")

        optimizer = Optimizer(output_dir, config)
        optimizer.run(INPUT_IMAGE, sharedIdentity=False)

        expected_files = [
            "render_0.png",
            "diffuseMap_0.png",
            "specularMap_0.png",
            "roughnessMap_0.png",
            "mesh0.obj",
            "envMap_0.png",
        ]
        for fname in expected_files:
            fpath = os.path.join(output_dir, fname)
            assert os.path.isfile(fpath), f"Missing output file: {fname}"


# ---------------------------------------------------------------------------
# Shared identity: scene building
# ---------------------------------------------------------------------------


class TestSharedIdentitySceneBuilding:
    """Verify build_scenes handles shared texture mode correctly."""

    def test_shared_texture_builds_n_scenes(self):
        """When diffuse.shape[0]==1, build_scenes still creates N scenes."""
        from rendering._variant import ensure_variant
        from rendering._scene import build_scenes

        ensure_variant()

        n_frames = 3
        n_verts = 4
        tex_res = 4

        vertices = torch.randn(n_frames, n_verts, 3)
        vertices[..., 2] = vertices[..., 2].abs() + 50.0
        indices = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32)
        normals = torch.randn(n_frames, n_verts, 3)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        uvs = torch.rand(n_verts, 2)

        diffuse = torch.rand(1, tex_res, tex_res, 3)
        specular = torch.rand(1, tex_res, tex_res, 3)
        roughness = torch.rand(1, tex_res, tex_res, 1).clamp(0.01, 1.0)

        focal = torch.tensor([500.0] * n_frames)
        envmap = torch.rand(n_frames, 8, 16, 3)

        scenes = build_scenes(
            vertices, indices, normals, uvs, diffuse, specular,
            roughness, focal, envmap,
            screen_width=32, screen_height=32, samples=1, bounces=1,
        )

        assert len(scenes) == n_frames
        for scene in scenes:
            assert scene is not None

    def test_shared_vs_independent_texture_scenes(self):
        """Shared texture mode (1,H,W,3) vs independent (N,H,W,3) both work."""
        from rendering._variant import ensure_variant
        from rendering._scene import build_scenes

        ensure_variant()

        n_frames = 2
        n_verts = 3
        tex_res = 4

        vertices = torch.randn(n_frames, n_verts, 3)
        vertices[..., 2] = vertices[..., 2].abs() + 50.0
        indices = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        normals = torch.randn(n_frames, n_verts, 3)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        uvs = torch.rand(n_verts, 2)
        focal = torch.tensor([500.0, 500.0])
        envmap = torch.rand(n_frames, 8, 16, 3)

        shared_diff = torch.rand(1, tex_res, tex_res, 3)
        shared_spec = torch.rand(1, tex_res, tex_res, 3)
        shared_rough = torch.rand(1, tex_res, tex_res, 1).clamp(0.01, 1.0)

        scenes_shared = build_scenes(
            vertices, indices, normals, uvs, shared_diff, shared_spec,
            shared_rough, focal, envmap,
            screen_width=32, screen_height=32, samples=1, bounces=1,
        )

        indep_diff = torch.rand(n_frames, tex_res, tex_res, 3)
        indep_spec = torch.rand(n_frames, tex_res, tex_res, 3)
        indep_rough = torch.rand(n_frames, tex_res, tex_res, 1).clamp(0.01, 1.0)

        scenes_indep = build_scenes(
            vertices, indices, normals, uvs, indep_diff, indep_spec,
            indep_rough, focal, envmap,
            screen_width=32, screen_height=32, samples=1, bounces=1,
        )

        assert len(scenes_shared) == len(scenes_indep) == n_frames


# ---------------------------------------------------------------------------
# Shared identity: parameter init
# ---------------------------------------------------------------------------


class TestSharedIdentityParameterInit:
    """Verify initSceneParameters sets correct dimensions for shared identity."""

    @pytest.mark.skipif(not has_model, reason="Basel Face Model .h5 not available")
    def test_shared_identity_shape_coeffs_dim_1(self):
        """With sharedIdentity=True, shape and albedo coefficients have batch dim 1."""
        from optim.config import Config
        from optim.pipeline import Pipeline

        config = Config()
        config.device = "cpu"
        config.path = os.path.join(DATA_DIR, "baselMorphableModel")
        config.textureResolution = 256
        config.rtTrainingSamples = 1
        config.lamdmarksDetectorType = "fan"
        config.bands = 2
        config.envMapRes = 8
        config.trimPca = 80

        pipe = Pipeline(config)
        n_frames = 3
        pipe.initSceneParameters(n_frames, sharedIdentity=True)

        assert pipe.vShapeCoeff.shape[0] == 1
        assert pipe.vAlbedoCoeff.shape[0] == 1
        assert pipe.vRoughness.shape[0] == 1

        assert pipe.vExpCoeff.shape[0] == n_frames
        assert pipe.vRotation.shape[0] == n_frames
        assert pipe.vTranslation.shape[0] == n_frames
        assert pipe.vFocals.shape[0] == n_frames
        assert pipe.vShCoeffs.shape[0] == n_frames

        assert pipe.sharedIdentity is True

    @pytest.mark.skipif(not has_model, reason="Basel Face Model .h5 not available")
    def test_non_shared_identity_all_dims_n(self):
        """Without sharedIdentity, all coefficients have batch dim N."""
        from optim.config import Config
        from optim.pipeline import Pipeline

        config = Config()
        config.device = "cpu"
        config.path = os.path.join(DATA_DIR, "baselMorphableModel")
        config.textureResolution = 256
        config.rtTrainingSamples = 1
        config.lamdmarksDetectorType = "fan"
        config.bands = 2
        config.envMapRes = 8
        config.trimPca = 80

        pipe = Pipeline(config)
        n_frames = 3
        pipe.initSceneParameters(n_frames, sharedIdentity=False)

        assert pipe.vShapeCoeff.shape[0] == n_frames
        assert pipe.vAlbedoCoeff.shape[0] == n_frames
        assert pipe.vRoughness.shape[0] == n_frames
        assert pipe.vExpCoeff.shape[0] == n_frames

        assert pipe.sharedIdentity is False


# ---------------------------------------------------------------------------
# Shared identity: rendering
# ---------------------------------------------------------------------------


class TestSharedIdentityRendering:
    """Verify rendering works with shared texture mode."""

    def test_renderer_shared_texture_render(self):
        """Renderer.buildScenes + render works with shared textures (batch 1)."""
        from rendering._variant import ensure_variant
        from rendering.renderer import Renderer

        ensure_variant()

        renderer = Renderer(samples=2, bounces=1, device="cpu")
        renderer.screenWidth = 32
        renderer.screenHeight = 32

        n_frames = 2
        n_verts = 3
        tex_res = 4

        vertices = torch.zeros(n_frames, n_verts, 3)
        vertices[0, 0] = torch.tensor([-5.0, -5.0, 50.0])
        vertices[0, 1] = torch.tensor([5.0, -5.0, 50.0])
        vertices[0, 2] = torch.tensor([0.0, 5.0, 50.0])
        vertices[1, 0] = torch.tensor([-5.0, -5.0, 60.0])
        vertices[1, 1] = torch.tensor([5.0, -5.0, 60.0])
        vertices[1, 2] = torch.tensor([0.0, 5.0, 60.0])

        normals = torch.zeros(n_frames, n_verts, 3)
        normals[..., 2] = -1.0

        indices = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        uvs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])

        diffuse = torch.full((1, tex_res, tex_res, 3), 0.5)
        specular = torch.full((1, tex_res, tex_res, 3), 0.1)
        roughness = torch.full((1, tex_res, tex_res, 1), 0.4)

        focal = torch.tensor([500.0, 500.0])
        envmap = torch.full((n_frames, 8, 16, 3), 0.3)

        scenes = renderer.buildScenes(
            vertices, indices, normals, uvs, diffuse, specular,
            roughness, focal, envmap,
        )
        assert len(scenes) == n_frames

        images = renderer.render(scenes)
        assert images.shape == (n_frames, 32, 32, 4)
        for i in range(n_frames):
            assert images[i].sum() > 0, f"Frame {i} is all zeros"


# ---------------------------------------------------------------------------
# E2E shared identity multi-image
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not has_model, reason="Basel Face Model .h5 not available")
@pytest.mark.skipif(not has_input, reason="Input image s1.png not available")
class TestE2ESharedIdentity:
    """End-to-end multi-image reconstruction with shared identity."""

    def test_e2e_shared_identity(self, tmp_path):
        """Run the full pipeline on a directory of images with sharedIdentity."""
        from optim.config import Config
        from optim.optimizer import Optimizer

        multiview_dir = tmp_path / "multiview"
        multiview_dir.mkdir()
        shutil.copy(INPUT_IMAGE, str(multiview_dir / "view1.png"))
        shutil.copy(INPUT_IMAGE, str(multiview_dir / "view2.png"))

        config = Config()
        config.device = "cpu"
        config.path = os.path.join(DATA_DIR, "baselMorphableModel")
        config.textureResolution = 256
        config.maxResolution = 64
        config.iterStep1 = 3
        config.iterStep2 = 3
        config.iterStep3 = 3
        config.rtSamples = 4
        config.rtTrainingSamples = 4
        config.debugFrequency = 0
        config.saveIntermediateStage = False
        config.verbose = False
        config.lamdmarksDetectorType = "fan"
        config.optimizeFocalLength = False
        config.saveExr = False
        config.camFocalLength = 500.0
        config.bands = 2
        config.envMapRes = 8
        config.trimPca = 80
        config.smoothSh = False
        config.weightLandmarksLossStep2 = 0.1
        config.weightLandmarksLossStep3 = 0.1
        config.weightAlbedoReg = 0.001
        config.weightShapeReg = 0.001
        config.weightExpressionReg = 0.001
        config.weightDiffuseSymmetryReg = 0.0
        config.weightDiffuseConsistencyReg = 0.0
        config.weightDiffuseSmoothnessReg = 0.0
        config.weightSpecularSymmetryReg = 0.0
        config.weightSpecularConsistencyReg = 0.0
        config.weightSpecularSmoothnessReg = 0.0
        config.weightRoughnessSymmetryReg = 0.0
        config.weightRoughnessConsistencyReg = 0.0
        config.weightRoughnessSmoothnessReg = 0.0

        output_dir = str(tmp_path / "output" / "multiview")

        optimizer = Optimizer(output_dir, config)
        optimizer.run(str(multiview_dir), sharedIdentity=True)

        assert optimizer.pipeline.vShapeCoeff.shape[0] == 1
        assert optimizer.pipeline.vAlbedoCoeff.shape[0] == 1
        assert optimizer.pipeline.sharedIdentity is True

        assert optimizer.pipeline.vExpCoeff.shape[0] == 2
        assert optimizer.pipeline.vRotation.shape[0] == 2
        assert optimizer.pipeline.vTranslation.shape[0] == 2

        for i in range(2):
            render_path = os.path.join(output_dir, f"render_{i}.png")
            assert os.path.isfile(render_path), f"Missing render_{i}.png"

        assert os.path.isfile(os.path.join(output_dir, "diffuseMap_0.png"))
        assert os.path.isfile(os.path.join(output_dir, "specularMap_0.png"))
        assert os.path.isfile(os.path.join(output_dir, "roughnessMap_0.png"))
