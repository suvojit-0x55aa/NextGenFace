"""Tests for optimization steps 1-3: landmark, photometric, and texture refinement."""

import math

import pytest
import torch
import numpy as np

from geometry.camera import Camera
from rendering._variant import ensure_variant


# ---------------------------------------------------------------------------
# Step 1: Landmark optimization helpers
# ---------------------------------------------------------------------------


def _project_landmarks(camera_vertices, landmark_indices, focals, centers):
    """Project 3D vertices to 2D using perspective projection."""
    head_points = camera_vertices[:, landmark_indices]
    proj_points = focals.view(-1, 1, 1) * head_points[..., :2] / head_points[..., 2:]
    proj_points += centers.unsqueeze(1)
    return proj_points


def _landmark_loss(camera_vertices, landmark_indices, focals, centers, target_landmarks):
    """Compute landmark reprojection loss."""
    proj_points = _project_landmarks(camera_vertices, landmark_indices, focals, centers)
    loss = torch.norm(proj_points - target_landmarks, 2, dim=-1).pow(2).mean()
    return loss


def _reg_stat_model(coeff, var):
    """Regularization loss for statistical model coefficients."""
    return ((coeff * coeff) / var).mean()


def _create_synthetic_face(n_vertices=100, n_landmarks=62, device='cpu'):
    """Create synthetic 3D face-like vertices for testing."""
    torch.manual_seed(42)

    vertices = torch.randn(n_vertices, 3, device=device) * 30.0
    vertices[:, 2] = 0

    landmark_indices = torch.arange(0, min(n_landmarks, n_vertices), device=device)

    n_shape = 10
    n_exp = 10
    shape_pca = torch.randn(n_shape, n_vertices, 3, device=device) * 0.5
    shape_var = torch.ones(n_shape, device=device) * 100.0
    exp_pca = torch.randn(n_exp, n_vertices, 3, device=device) * 0.3
    exp_var = torch.ones(n_exp, device=device) * 50.0

    return vertices, landmark_indices, shape_pca, shape_var, exp_pca, exp_var


def _compute_shape(shape_mean, shape_pca, exp_pca, shape_coeff, exp_coeff):
    """Compute vertices from shape/expression coefficients."""
    vertices = shape_mean + torch.einsum('ni,ijk->njk', shape_coeff, shape_pca) + \
               torch.einsum('ni,ijk->njk', exp_coeff, exp_pca)
    return vertices


def _run_step1_optimization(
    camera, shape_mean, shape_pca, exp_pca, exp_var,
    landmark_indices, target_landmarks, focals, centers,
    n_iters=200, device='cpu'
):
    """Run Step 1 optimization loop."""
    n_frames = target_landmarks.shape[0]
    n_exp = exp_pca.shape[0]

    vRotation = torch.zeros([n_frames, 3], dtype=torch.float32, device=device)
    vTranslation = torch.zeros([n_frames, 3], dtype=torch.float32, device=device)
    vTranslation[:, 2] = 500.0
    vRotation[:, 0] = 3.14

    vRotation = vRotation + torch.randn_like(vRotation) * 0.05
    vTranslation = vTranslation + torch.randn_like(vTranslation) * 10.0

    vRotation.requires_grad = True
    vTranslation.requires_grad = True

    vExpCoeff = torch.zeros([n_frames, n_exp], dtype=torch.float32, device=device)
    vExpCoeff.requires_grad = True

    vShapeCoeff = torch.zeros([n_frames, shape_pca.shape[0]], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([
        {'params': vRotation, 'lr': 0.02},
        {'params': vTranslation, 'lr': 0.02},
        {'params': vExpCoeff, 'lr': 0.02},
    ])

    losses = []
    for _ in range(n_iters):
        optimizer.zero_grad()
        vertices = _compute_shape(shape_mean, shape_pca, exp_pca, vShapeCoeff, vExpCoeff)
        camera_vertices = camera.transformVertices(vertices, vTranslation, vRotation)

        loss = _landmark_loss(camera_vertices, landmark_indices, focals, centers, target_landmarks)
        loss = loss + 0.1 * _reg_stat_model(vExpCoeff, exp_var)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses, vRotation, vTranslation, vExpCoeff


@pytest.fixture
def step1_setup():
    """Set up synthetic face data and generate target landmarks."""
    device = 'cpu'
    camera = Camera(device)

    (shape_mean, landmark_indices, shape_pca, shape_var,
     exp_pca, exp_var) = _create_synthetic_face(
        n_vertices=100, n_landmarks=62, device=device
    )

    n_frames = 1
    gt_rotation = torch.tensor([[3.14, 0.0, 0.0]], device=device)
    gt_translation = torch.tensor([[0.0, 0.0, 500.0]], device=device)
    gt_shape_coeff = torch.zeros([n_frames, shape_pca.shape[0]], device=device)
    gt_exp_coeff = torch.zeros([n_frames, exp_pca.shape[0]], device=device)

    focal = torch.tensor([500.0], device=device)
    center = torch.tensor([[128.0, 128.0]], device=device)

    with torch.no_grad():
        gt_vertices = _compute_shape(shape_mean, shape_pca, exp_pca, gt_shape_coeff, gt_exp_coeff)
        gt_camera_verts = camera.transformVertices(gt_vertices, gt_translation, gt_rotation)
        target_landmarks = _project_landmarks(gt_camera_verts, landmark_indices, focal, center)

    return {
        'camera': camera,
        'shape_mean': shape_mean,
        'shape_pca': shape_pca,
        'exp_pca': exp_pca,
        'exp_var': exp_var,
        'landmark_indices': landmark_indices,
        'target_landmarks': target_landmarks,
        'focals': focal,
        'centers': center,
        'device': device,
    }


# ---------------------------------------------------------------------------
# Step 1 tests
# ---------------------------------------------------------------------------


def test_step1_loss_decreases(step1_setup):
    """Step 1 loss should decrease over iterations."""
    losses, _, _, _ = _run_step1_optimization(
        camera=step1_setup['camera'],
        shape_mean=step1_setup['shape_mean'],
        shape_pca=step1_setup['shape_pca'],
        exp_pca=step1_setup['exp_pca'],
        exp_var=step1_setup['exp_var'],
        landmark_indices=step1_setup['landmark_indices'],
        target_landmarks=step1_setup['target_landmarks'],
        focals=step1_setup['focals'],
        centers=step1_setup['centers'],
        n_iters=500,
        device=step1_setup['device'],
    )

    n = len(losses)
    early_avg = sum(losses[:n // 10]) / (n // 10)
    late_avg = sum(losses[-n // 10:]) / (n // 10)

    assert late_avg < early_avg, (
        f"Loss did not decrease: early avg={early_avg:.4f}, late avg={late_avg:.4f}"
    )

    assert losses[-1] < losses[0] * 0.8, (
        f"Loss did not decrease enough: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )


def test_step1_no_nan(step1_setup):
    """Step 1 parameters should not contain NaN or explode."""
    losses, vRotation, vTranslation, vExpCoeff = _run_step1_optimization(
        camera=step1_setup['camera'],
        shape_mean=step1_setup['shape_mean'],
        shape_pca=step1_setup['shape_pca'],
        exp_pca=step1_setup['exp_pca'],
        exp_var=step1_setup['exp_var'],
        landmark_indices=step1_setup['landmark_indices'],
        target_landmarks=step1_setup['target_landmarks'],
        focals=step1_setup['focals'],
        centers=step1_setup['centers'],
        n_iters=300,
        device=step1_setup['device'],
    )

    assert all(not math.isnan(l) for l in losses), "Loss contains NaN values"
    assert all(not math.isinf(l) for l in losses), "Loss contains Inf values"

    assert not torch.isnan(vRotation).any(), "Rotation contains NaN"
    assert not torch.isnan(vTranslation).any(), "Translation contains NaN"
    assert not torch.isnan(vExpCoeff).any(), "Expression coefficients contain NaN"

    assert vRotation.abs().max() < 100.0, f"Rotation exploded: max={vRotation.abs().max():.2f}"
    assert vTranslation.abs().max() < 10000.0, f"Translation exploded: max={vTranslation.abs().max():.2f}"
    assert vExpCoeff.abs().max() < 1000.0, f"Expression coeffs exploded: max={vExpCoeff.abs().max():.2f}"


def test_step1_landmark_error_converges(step1_setup):
    """After optimization, landmark reprojection error should be small."""
    losses, vRotation, vTranslation, vExpCoeff = _run_step1_optimization(
        camera=step1_setup['camera'],
        shape_mean=step1_setup['shape_mean'],
        shape_pca=step1_setup['shape_pca'],
        exp_pca=step1_setup['exp_pca'],
        exp_var=step1_setup['exp_var'],
        landmark_indices=step1_setup['landmark_indices'],
        target_landmarks=step1_setup['target_landmarks'],
        focals=step1_setup['focals'],
        centers=step1_setup['centers'],
        n_iters=1000,
        device=step1_setup['device'],
    )

    with torch.no_grad():
        gt_shape_coeff = torch.zeros(
            [1, step1_setup['shape_pca'].shape[0]], device=step1_setup['device']
        )
        vertices = _compute_shape(
            step1_setup['shape_mean'], step1_setup['shape_pca'],
            step1_setup['exp_pca'], gt_shape_coeff, vExpCoeff
        )
        camera_verts = step1_setup['camera'].transformVertices(
            vertices, vTranslation, vRotation
        )
        proj = _project_landmarks(
            camera_verts, step1_setup['landmark_indices'],
            step1_setup['focals'], step1_setup['centers']
        )
        pixel_error = torch.norm(
            proj - step1_setup['target_landmarks'], 2, dim=-1
        ).mean().item()

    assert pixel_error < 5.0, (
        f"Landmark reprojection error too large: {pixel_error:.2f} pixels (threshold: 5.0)"
    )


def test_step1_multiframe_no_nan():
    """Step 1 should handle multiple frames without NaN."""
    device = 'cpu'
    camera = Camera(device)
    n_frames = 3

    (shape_mean, landmark_indices, shape_pca, shape_var,
     exp_pca, exp_var) = _create_synthetic_face(
        n_vertices=100, n_landmarks=62, device=device
    )

    focals = torch.tensor([500.0] * n_frames, device=device)
    centers = torch.tensor([[128.0, 128.0]] * n_frames, device=device)

    gt_rotations = torch.tensor([
        [3.14, 0.0, 0.0],
        [3.14, 0.1, 0.0],
        [3.14, -0.1, 0.0],
    ], device=device)
    gt_translations = torch.tensor([
        [0.0, 0.0, 500.0],
        [5.0, 0.0, 500.0],
        [-5.0, 0.0, 500.0],
    ], device=device)

    gt_shape_coeff = torch.zeros([n_frames, shape_pca.shape[0]], device=device)
    gt_exp_coeff = torch.zeros([n_frames, exp_pca.shape[0]], device=device)

    with torch.no_grad():
        gt_verts = _compute_shape(shape_mean, shape_pca, exp_pca, gt_shape_coeff, gt_exp_coeff)
        gt_cam_verts = camera.transformVertices(gt_verts, gt_translations, gt_rotations)
        target_landmarks = _project_landmarks(gt_cam_verts, landmark_indices, focals, centers)

    losses, vRot, vTrans, vExp = _run_step1_optimization(
        camera=camera,
        shape_mean=shape_mean,
        shape_pca=shape_pca,
        exp_pca=exp_pca,
        exp_var=exp_var,
        landmark_indices=landmark_indices,
        target_landmarks=target_landmarks,
        focals=focals,
        centers=centers,
        n_iters=300,
        device=device,
    )

    assert all(not math.isnan(l) for l in losses), "Multi-frame loss contains NaN"
    assert not torch.isnan(vRot).any(), "Multi-frame rotation contains NaN"
    assert not torch.isnan(vTrans).any(), "Multi-frame translation contains NaN"

    assert losses[-1] < losses[0], "Multi-frame loss did not decrease"


# ---------------------------------------------------------------------------
# Step 2: Photometric optimization helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup_variant():
    ensure_variant()


from helpers import make_sphere_mesh as _make_sphere_mesh, compute_normals as _compute_normals


def _make_test_scene_data(tex_res=32, screen_size=64):
    """Create a complete set of scene data for testing."""
    verts, faces, uvs = _make_sphere_mesh(radius=5.0, center=(0, 0, 50))
    normals = _compute_normals(verts, faces)

    return {
        "vertices": verts.unsqueeze(0),
        "faces": faces,
        "normals": normals.unsqueeze(0),
        "uvs": uvs,
        "diffuse": torch.full((1, tex_res, tex_res, 3), 0.6, dtype=torch.float32),
        "specular": torch.full((1, tex_res, tex_res, 3), 0.04, dtype=torch.float32),
        "roughness": torch.full((1, tex_res, tex_res, 1), 0.5, dtype=torch.float32),
        "focal": torch.tensor([500.0]),
        "envmap": torch.full((1, 16, 32, 3), 0.5, dtype=torch.float32),
        "screen_size": screen_size,
    }


# ---------------------------------------------------------------------------
# Step 2 tests
# ---------------------------------------------------------------------------


def test_step2_loss_decreases():
    """Test that a Step 2-like optimization loop shows decreasing loss."""
    from rendering.renderer import Renderer

    data = _make_test_scene_data(tex_res=32, screen_size=64)

    renderer = Renderer(samples=4, bounces=1, device='cpu')
    renderer.screenWidth = data["screen_size"]
    renderer.screenHeight = data["screen_size"]

    target_diffuse = torch.full((1, 32, 32, 3), 0.8, dtype=torch.float32)
    target_diffuse[:, :, :, 1] = 0.2
    target_diffuse[:, :, :, 2] = 0.1

    with torch.no_grad():
        target_scenes = renderer.buildScenes(
            data["vertices"], data["faces"], data["normals"], data["uvs"],
            target_diffuse, data["specular"], data["roughness"],
            data["focal"], data["envmap"],
        )
        target_images = renderer.render(target_scenes)

    assert target_images.shape == (1, 64, 64, 4), f"Target shape: {target_images.shape}"
    assert target_images.abs().sum() > 0, "Target image is all zeros"

    opt_diffuse = torch.full((1, 32, 32, 3), 0.5, dtype=torch.float32)
    opt_diffuse.requires_grad_(True)

    optimizer = torch.optim.Adam([opt_diffuse], lr=0.05)
    losses = []

    n_iters = 15
    for iteration in range(n_iters):
        torch.set_grad_enabled(True)
        optimizer.zero_grad()

        diffuse_clamped = torch.clamp(opt_diffuse, 0.01, 1.0)

        scenes = renderer.buildScenes(
            data["vertices"].detach(), data["faces"],
            data["normals"].detach(), data["uvs"],
            diffuse_clamped, data["specular"].detach(),
            data["roughness"].detach(), data["focal"],
            data["envmap"].detach(),
        )
        images = renderer.render(scenes)

        mask = images[..., 3:]
        diff = mask * (images[..., :3] - target_images.detach()[..., :3]).abs()
        loss = 1000.0 * diff.mean()

        losses.append(loss.item())
        loss.backward()

        assert opt_diffuse.grad is not None, f"No gradient at iteration {iteration}"
        assert not torch.isnan(opt_diffuse.grad).any(), f"NaN gradient at iter {iteration}"
        assert opt_diffuse.grad.abs().sum() > 0, f"Zero gradient at iter {iteration}"

        optimizer.step()

    torch.set_grad_enabled(False)

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )

    mid = n_iters // 2
    early_avg = sum(losses[:mid]) / mid
    late_avg = sum(losses[mid:]) / (n_iters - mid)
    assert late_avg < early_avg, (
        f"Loss trend not decreasing: early_avg={early_avg:.4f}, late_avg={late_avg:.4f}"
    )


def test_step2_no_nan():
    """Test that differentiable rendering produces no NaN values."""
    from rendering.renderer import Renderer

    data = _make_test_scene_data(tex_res=32, screen_size=64)
    renderer = Renderer(samples=4, bounces=1, device='cpu')
    renderer.screenWidth = data["screen_size"]
    renderer.screenHeight = data["screen_size"]

    opt_diffuse = data["diffuse"].clone().requires_grad_(True)
    opt_envmap = data["envmap"].clone().requires_grad_(True)

    torch.set_grad_enabled(True)

    scenes = renderer.buildScenes(
        data["vertices"].detach(), data["faces"],
        data["normals"].detach(), data["uvs"],
        opt_diffuse, data["specular"].detach(),
        data["roughness"].detach(), data["focal"],
        opt_envmap,
    )
    images = renderer.render(scenes)

    assert not torch.isnan(images).any(), "NaN in rendered images"
    assert not torch.isinf(images).any(), "Inf in rendered images"

    loss = images[..., :3].mean()
    loss.backward()

    assert opt_diffuse.grad is not None, "No gradient for diffuse"
    assert not torch.isnan(opt_diffuse.grad).any(), "NaN in diffuse gradient"

    assert opt_envmap.grad is not None, "No gradient for envmap"
    assert not torch.isnan(opt_envmap.grad).any(), "NaN in envmap gradient"

    torch.set_grad_enabled(False)


def test_step2_renders_face():
    """Test that rendered images are visually reasonable."""
    from rendering.renderer import Renderer

    data = _make_test_scene_data(tex_res=32, screen_size=64)
    renderer = Renderer(samples=4, bounces=1, device='cpu')
    renderer.screenWidth = data["screen_size"]
    renderer.screenHeight = data["screen_size"]

    with torch.no_grad():
        scenes = renderer.buildScenes(
            data["vertices"], data["faces"], data["normals"], data["uvs"],
            data["diffuse"], data["specular"], data["roughness"],
            data["focal"], data["envmap"],
        )
        images = renderer.render(scenes)

    assert images.shape == (1, 64, 64, 4), f"Wrong shape: {images.shape}"

    rgb = images[0, :, :, :3]
    alpha = images[0, :, :, 3]

    assert rgb.abs().sum() > 0, "Rendered image is all black"
    assert (rgb < 0.99).any(), "Rendered image is all white"
    assert (alpha > 0.5).any(), "No mesh visible in alpha"

    assert not torch.isnan(images).any(), "NaN in rendered image"
    assert not torch.isinf(images).any(), "Inf in rendered image"


# ---------------------------------------------------------------------------
# Step 3: Texture refinement tests
# ---------------------------------------------------------------------------


def test_step3_loss_decreases():
    """Test that a Step 3-like texture optimization loop shows decreasing loss."""
    from rendering.renderer import Renderer

    data = _make_test_scene_data(tex_res=32, screen_size=64)

    renderer = Renderer(samples=4, bounces=1, device='cpu')
    renderer.screenWidth = data["screen_size"]
    renderer.screenHeight = data["screen_size"]

    target_diffuse = torch.full((1, 32, 32, 3), 0.8, dtype=torch.float32)
    target_diffuse[:, :, :, 1] = 0.2
    target_diffuse[:, :, :, 2] = 0.1
    target_roughness = torch.full((1, 32, 32, 1), 0.2, dtype=torch.float32)

    with torch.no_grad():
        target_scenes = renderer.buildScenes(
            data["vertices"], data["faces"], data["normals"], data["uvs"],
            target_diffuse, data["specular"], target_roughness,
            data["focal"], data["envmap"],
        )
        target_images = renderer.render(target_scenes)

    assert target_images.shape == (1, 64, 64, 4), f"Target shape: {target_images.shape}"
    assert target_images.abs().sum() > 0, "Target image is all zeros"

    opt_diffuse = data["diffuse"].clone().requires_grad_(True)
    opt_specular = data["specular"].clone().requires_grad_(True)
    opt_roughness = data["roughness"].clone().requires_grad_(True)

    optimizer = torch.optim.Adam([
        {'params': opt_diffuse, 'lr': 0.005},
        {'params': opt_specular, 'lr': 0.02},
        {'params': opt_roughness, 'lr': 0.02},
    ])

    losses = []
    n_iters = 15
    for iteration in range(n_iters):
        torch.set_grad_enabled(True)
        optimizer.zero_grad()

        diff_clamped = torch.clamp(opt_diffuse, 0.01, 1.0)
        rough_clamped = torch.clamp(opt_roughness, 1e-20, 10.0)

        scenes = renderer.buildScenes(
            data["vertices"].detach(), data["faces"],
            data["normals"].detach(), data["uvs"],
            diff_clamped, opt_specular.detach(),
            rough_clamped, data["focal"],
            data["envmap"].detach(),
        )
        images = renderer.render(scenes)

        mask = images[..., 3:]
        diff = mask * (images[..., :3] - target_images.detach()[..., :3]).abs()
        loss = 1000.0 * diff.mean()

        losses.append(loss.item())
        loss.backward()

        assert opt_diffuse.grad is not None, f"No gradient for diffuse at iter {iteration}"
        assert not torch.isnan(opt_diffuse.grad).any(), f"NaN in diffuse grad at iter {iteration}"

        optimizer.step()

    torch.set_grad_enabled(False)

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )

    mid = n_iters // 2
    early_avg = sum(losses[:mid]) / mid
    late_avg = sum(losses[mid:]) / (n_iters - mid)
    assert late_avg < early_avg, (
        f"Loss trend not decreasing: early_avg={early_avg:.4f}, late_avg={late_avg:.4f}"
    )


def test_step3_textures_refined():
    """Test that optimized textures diverge from initial uniform textures."""
    from rendering.renderer import Renderer

    data = _make_test_scene_data(tex_res=32, screen_size=64)

    renderer = Renderer(samples=4, bounces=1, device='cpu')
    renderer.screenWidth = data["screen_size"]
    renderer.screenHeight = data["screen_size"]

    target_diffuse = torch.zeros((1, 32, 32, 3), dtype=torch.float32)
    for i in range(32):
        target_diffuse[:, i, :, 0] = i / 31.0
    target_diffuse[:, :, :, 1] = 0.3
    target_diffuse[:, :, :, 2] = 0.2

    with torch.no_grad():
        target_scenes = renderer.buildScenes(
            data["vertices"], data["faces"], data["normals"], data["uvs"],
            target_diffuse, data["specular"], data["roughness"],
            data["focal"], data["envmap"],
        )
        target_images = renderer.render(target_scenes)

    initial_diffuse = torch.full((1, 32, 32, 3), 0.5, dtype=torch.float32)
    opt_diffuse = initial_diffuse.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([opt_diffuse], lr=0.05)

    n_iters = 20
    for iteration in range(n_iters):
        torch.set_grad_enabled(True)
        optimizer.zero_grad()

        diff_clamped = torch.clamp(opt_diffuse, 0.01, 1.0)

        scenes = renderer.buildScenes(
            data["vertices"].detach(), data["faces"],
            data["normals"].detach(), data["uvs"],
            diff_clamped, data["specular"].detach(),
            data["roughness"].detach(), data["focal"],
            data["envmap"].detach(),
        )
        images = renderer.render(scenes)

        mask = images[..., 3:]
        diff = mask * (images[..., :3] - target_images.detach()[..., :3]).abs()
        loss = 1000.0 * diff.mean()

        loss.backward()
        optimizer.step()

    torch.set_grad_enabled(False)

    refined = opt_diffuse.detach().clone()
    initial = initial_diffuse.detach().clone()

    texture_diff = (refined - initial).abs().mean().item()
    assert texture_diff > 0.001, (
        f"Texture barely changed from initial: mean diff = {texture_diff:.6f}"
    )

    refined_std = refined.std().item()
    initial_std = initial.std().item()
    assert refined_std > initial_std, (
        f"Refined texture not more varied: refined_std={refined_std:.6f}, "
        f"initial_std={initial_std:.6f}"
    )


def test_step3_all_texture_grads():
    """Test that gradients flow to all three texture types simultaneously."""
    from rendering.renderer import Renderer

    data = _make_test_scene_data(tex_res=32, screen_size=64)

    renderer = Renderer(samples=4, bounces=1, device='cpu')
    renderer.screenWidth = data["screen_size"]
    renderer.screenHeight = data["screen_size"]

    opt_diffuse = data["diffuse"].clone().requires_grad_(True)
    opt_roughness = data["roughness"].clone().requires_grad_(True)

    torch.set_grad_enabled(True)

    scenes = renderer.buildScenes(
        data["vertices"].detach(), data["faces"],
        data["normals"].detach(), data["uvs"],
        opt_diffuse, data["specular"].detach(),
        opt_roughness, data["focal"],
        data["envmap"].detach(),
    )
    images = renderer.render(scenes)

    loss = images[..., :3].mean()
    loss.backward()

    torch.set_grad_enabled(False)

    assert opt_diffuse.grad is not None, "No gradient for diffuse"
    assert not torch.isnan(opt_diffuse.grad).any(), "NaN in diffuse gradient"
    assert opt_diffuse.grad.abs().sum() > 0, "Zero gradient for diffuse"

    assert opt_roughness.grad is not None, "No gradient for roughness"
    assert not torch.isnan(opt_roughness.grad).any(), "NaN in roughness gradient"
    assert opt_roughness.grad.abs().sum() > 0, "Zero gradient for roughness"
