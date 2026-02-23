"""US-018: saveImage works without pyredner dependency."""
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NextFace'))
from image import saveImage


def test_us018_save_image_no_pyredner():
    """saveImage must not import pyredner."""
    import importlib
    # Ensure pyredner is NOT required
    # We verify by checking that saveImage source has no pyredner reference
    import inspect
    source = inspect.getsource(saveImage)
    assert 'pyredner' not in source


def test_us018_save_with_gamma_3channel():
    """Save a 3-channel image with gamma correction."""
    img = torch.full((16, 16, 3), 0.5, dtype=torch.float32)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        path = f.name
    try:
        saveImage(img, path, gamma=2.2)
        assert os.path.exists(path)
        loaded = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        assert loaded is not None
        assert loaded.shape == (16, 16, 3)
        # Gamma-corrected 0.5 should be ~0.5^(1/2.2) ≈ 0.73 → ~186
        expected = int((0.5 ** (1.0 / 2.2)) * 255)
        mean_val = loaded.mean()
        assert abs(mean_val - expected) < 5, f"Expected ~{expected}, got {mean_val}"
    finally:
        os.unlink(path)


def test_us018_save_with_gamma_4channel():
    """Save a 4-channel RGBA image."""
    img = torch.full((16, 16, 4), 0.8, dtype=torch.float32)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        path = f.name
    try:
        saveImage(img, path, gamma=2.2)
        assert os.path.exists(path)
        loaded = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        assert loaded is not None
        assert loaded.shape == (16, 16, 4)
    finally:
        os.unlink(path)


def test_us018_save_no_gamma():
    """Save with gamma=1.0 (no correction)."""
    val = 0.6
    img = torch.full((8, 8, 3), val, dtype=torch.float32)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        path = f.name
    try:
        saveImage(img, path, gamma=1.0)
        loaded = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        expected = int(val * 255)
        mean_val = loaded.mean()
        assert abs(mean_val - expected) < 2, f"Expected ~{expected}, got {mean_val}"
    finally:
        os.unlink(path)
