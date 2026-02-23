"""US-019: Verify Step 1 optimization (landmarks only).

Step 1 optimizes head pose (rotation + translation) and expression
coefficients using landmark reprojection loss. It is pure PyTorch —
no rendering involved.

These tests verify the optimization loop converges correctly using
synthetic data that mimics the real Pipeline's Step 1 flow.
"""

import sys
import os
import math
import torch
import numpy as np
import pytest

# Add NextFace to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NextFace'))

from camera import Camera


def _project_landmarks(camera_vertices, landmark_indices, focals, centers):
    """Project 3D vertices to 2D using perspective projection (same as Pipeline.landmarkLoss)."""
    head_points = camera_vertices[:, landmark_indices]
    proj_points = focals.view(-1, 1, 1) * head_points[..., :2] / head_points[..., 2:]
    proj_points += centers.unsqueeze(1)
    return proj_points


def _landmark_loss(camera_vertices, landmark_indices, focals, centers, target_landmarks):
    """Compute landmark reprojection loss (same formula as Pipeline.landmarkLoss)."""
    proj_points = _project_landmarks(camera_vertices, landmark_indices, focals, centers)
    loss = torch.norm(proj_points - target_landmarks, 2, dim=-1).pow(2).mean()
    return loss


def _reg_stat_model(coeff, var):
    """Regularization loss for statistical model coefficients."""
    return ((coeff * coeff) / var).mean()


def _create_synthetic_face(n_vertices=100, n_landmarks=62, device='cpu'):
    """Create synthetic 3D face-like vertices for testing.

    Returns:
        vertices: [V, 3] mean shape
        landmark_indices: [L] indices of landmark vertices
        shape_pca: [K, V, 3] shape basis
        shape_var: [K] PCA variances
        exp_pca: [K_exp, V, 3] expression basis
        exp_var: [K_exp] expression variances
    """
    torch.manual_seed(42)

    # Create face-like vertices centered around (0, 0, 0)
    # Face is roughly a flat disc facing +Z
    vertices = torch.randn(n_vertices, 3, device=device) * 30.0
    vertices[:, 2] = 0  # Flat face at z=0

    # Landmark indices (subset of vertices)
    landmark_indices = torch.arange(0, min(n_landmarks, n_vertices), device=device)

    # PCA bases (small perturbations)
    n_shape = 10
    n_exp = 10
    shape_pca = torch.randn(n_shape, n_vertices, 3, device=device) * 0.5
    shape_var = torch.ones(n_shape, device=device) * 100.0
    exp_pca = torch.randn(n_exp, n_vertices, 3, device=device) * 0.3
    exp_var = torch.ones(n_exp, device=device) * 50.0

    return vertices, landmark_indices, shape_pca, shape_var, exp_pca, exp_var


def _compute_shape(shape_mean, shape_pca, exp_pca, shape_coeff, exp_coeff):
    """Compute vertices from shape/expression coefficients (same as MorphableModel.computeShape)."""
    vertices = shape_mean + torch.einsum('ni,ijk->njk', shape_coeff, shape_pca) + \
               torch.einsum('ni,ijk->njk', exp_coeff, exp_pca)
    return vertices


def _run_step1_optimization(
    camera, shape_mean, shape_pca, exp_pca, exp_var,
    landmark_indices, target_landmarks, focals, centers,
    n_iters=200, device='cpu'
):
    """Run Step 1 optimization loop (same structure as Optimizer.runStep1).

    Returns:
        losses: list of loss values per iteration
        vRotation: optimized rotation
        vTranslation: optimized translation
        vExpCoeff: optimized expression coefficients
    """
    n_frames = target_landmarks.shape[0]
    n_exp = exp_pca.shape[0]

    # Initialize parameters (same as Pipeline.initSceneParameters)
    vRotation = torch.zeros([n_frames, 3], dtype=torch.float32, device=device)
    vTranslation = torch.zeros([n_frames, 3], dtype=torch.float32, device=device)
    vTranslation[:, 2] = 500.0
    vRotation[:, 0] = 3.14  # ~pi rotation around X (face is flipped)
    vExpCoeff = torch.zeros([n_frames, n_exp], dtype=torch.float32, device=device)

    # Perturb initial guess to make optimization nontrivial
    vRotation = vRotation + torch.randn_like(vRotation) * 0.05
    vTranslation = vTranslation + torch.randn_like(vTranslation) * 10.0

    vRotation.requires_grad = True
    vTranslation.requires_grad = True
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


class TestStep1LossDecreases:
    """Test that Step 1 optimization converges."""

    @pytest.fixture
    def setup(self):
        """Set up synthetic face data and generate target landmarks."""
        device = 'cpu'
        camera = Camera(device)

        (shape_mean, landmark_indices, shape_pca, shape_var,
         exp_pca, exp_var) = _create_synthetic_face(
            n_vertices=100, n_landmarks=62, device=device
        )

        # Generate ground-truth target landmarks by projecting with known pose
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

    def test_us019_step1_loss_decreases(self, setup):
        """Step 1 loss should decrease over iterations."""
        losses, _, _, _ = _run_step1_optimization(
            camera=setup['camera'],
            shape_mean=setup['shape_mean'],
            shape_pca=setup['shape_pca'],
            exp_pca=setup['exp_pca'],
            exp_var=setup['exp_var'],
            landmark_indices=setup['landmark_indices'],
            target_landmarks=setup['target_landmarks'],
            focals=setup['focals'],
            centers=setup['centers'],
            n_iters=500,
            device=setup['device'],
        )

        # Loss should decrease overall (compare first 10% average to last 10% average)
        n = len(losses)
        early_avg = sum(losses[:n // 10]) / (n // 10)
        late_avg = sum(losses[-n // 10:]) / (n // 10)

        assert late_avg < early_avg, (
            f"Loss did not decrease: early avg={early_avg:.4f}, late avg={late_avg:.4f}"
        )

        # Final loss should be smaller than initial (with tolerance for noise)
        assert losses[-1] < losses[0] * 0.8, (
            f"Loss did not decrease enough: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )

    def test_us019_step1_no_nan(self, setup):
        """Step 1 parameters should not contain NaN or explode."""
        losses, vRotation, vTranslation, vExpCoeff = _run_step1_optimization(
            camera=setup['camera'],
            shape_mean=setup['shape_mean'],
            shape_pca=setup['shape_pca'],
            exp_pca=setup['exp_pca'],
            exp_var=setup['exp_var'],
            landmark_indices=setup['landmark_indices'],
            target_landmarks=setup['target_landmarks'],
            focals=setup['focals'],
            centers=setup['centers'],
            n_iters=300,
            device=setup['device'],
        )

        # No NaN in losses
        assert all(not math.isnan(l) for l in losses), "Loss contains NaN values"
        assert all(not math.isinf(l) for l in losses), "Loss contains Inf values"

        # No NaN in parameters
        assert not torch.isnan(vRotation).any(), "Rotation contains NaN"
        assert not torch.isnan(vTranslation).any(), "Translation contains NaN"
        assert not torch.isnan(vExpCoeff).any(), "Expression coefficients contain NaN"

        # Parameters should not explode (reasonable ranges)
        assert vRotation.abs().max() < 100.0, f"Rotation exploded: max={vRotation.abs().max():.2f}"
        assert vTranslation.abs().max() < 10000.0, f"Translation exploded: max={vTranslation.abs().max():.2f}"
        assert vExpCoeff.abs().max() < 1000.0, f"Expression coeffs exploded: max={vExpCoeff.abs().max():.2f}"

    def test_us019_step1_landmark_error_converges(self, setup):
        """After optimization, landmark reprojection error should be small."""
        losses, vRotation, vTranslation, vExpCoeff = _run_step1_optimization(
            camera=setup['camera'],
            shape_mean=setup['shape_mean'],
            shape_pca=setup['shape_pca'],
            exp_pca=setup['exp_pca'],
            exp_var=setup['exp_var'],
            landmark_indices=setup['landmark_indices'],
            target_landmarks=setup['target_landmarks'],
            focals=setup['focals'],
            centers=setup['centers'],
            n_iters=1000,
            device=setup['device'],
        )

        # Compute final reprojection error in pixels
        with torch.no_grad():
            gt_shape_coeff = torch.zeros(
                [1, setup['shape_pca'].shape[0]], device=setup['device']
            )
            vertices = _compute_shape(
                setup['shape_mean'], setup['shape_pca'],
                setup['exp_pca'], gt_shape_coeff, vExpCoeff
            )
            camera_verts = setup['camera'].transformVertices(
                vertices, vTranslation, vRotation
            )
            proj = _project_landmarks(
                camera_verts, setup['landmark_indices'],
                setup['focals'], setup['centers']
            )
            pixel_error = torch.norm(
                proj - setup['target_landmarks'], 2, dim=-1
            ).mean().item()

        # With synthetic data, regularization prevents exact convergence.
        # The real pipeline uses 2000 iterations. We use a relaxed threshold.
        assert pixel_error < 5.0, (
            f"Landmark reprojection error too large: {pixel_error:.2f} pixels (threshold: 5.0)"
        )


class TestStep1MultiFrame:
    """Test Step 1 with multiple frames (batch dimension)."""

    def test_us019_step1_multiframe_no_nan(self):
        """Step 1 should handle multiple frames without NaN."""
        device = 'cpu'
        camera = Camera(device)
        n_frames = 3

        (shape_mean, landmark_indices, shape_pca, shape_var,
         exp_pca, exp_var) = _create_synthetic_face(
            n_vertices=100, n_landmarks=62, device=device
        )

        # Generate different target landmarks for each frame
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

        # No NaN
        assert all(not math.isnan(l) for l in losses), "Multi-frame loss contains NaN"
        assert not torch.isnan(vRot).any(), "Multi-frame rotation contains NaN"
        assert not torch.isnan(vTrans).any(), "Multi-frame translation contains NaN"

        # Loss should decrease
        assert losses[-1] < losses[0], "Multi-frame loss did not decrease"
