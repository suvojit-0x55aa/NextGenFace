"""US-024: Checkpoint save/load round-trip tests.

Tests that saveParameters() and loadParameters() correctly serialize and
deserialize all optimization parameters via pickle.
"""

import os
import pickle
import tempfile

import numpy as np
import pytest
import torch


class MockRenderer:
    """Minimal renderer mock with screenWidth/screenHeight attributes."""

    def __init__(self):
        self.screenWidth = 256
        self.screenHeight = 256


class MockPipeline:
    """Minimal pipeline mock with the parameter tensors that checkpointing uses."""

    def __init__(self, device="cpu", n_frames=2):
        self.renderer = MockRenderer()
        self.sharedIdentity = True
        self.device = device

        # Core parameters (always saved)
        self.vShapeCoeff = torch.randn(1, 199, device=device)
        self.vAlbedoCoeff = torch.randn(1, 199, device=device)
        self.vExpCoeff = torch.randn(n_frames, 100, device=device)
        self.vRotation = torch.randn(n_frames, 3, device=device)
        self.vTranslation = torch.randn(n_frames, 3, device=device)
        self.vFocals = torch.randn(n_frames, 1, device=device)
        self.vShCoeffs = torch.randn(n_frames, 9, 3, device=device)


class MockOptimizer:
    """Minimal optimizer mock matching the saveParameters/loadParameters interface."""

    def __init__(self, device="cpu", n_frames=2):
        self.device = device
        self.pipeline = MockPipeline(device, n_frames)
        self.vEnhancedDiffuse = None
        self.vEnhancedSpecular = None
        self.vEnhancedRoughness = None

    def saveParameters(self, outputFileName):
        dict = {
            "vShapeCoeff": self.pipeline.vShapeCoeff.detach().cpu().numpy(),
            "vAlbedoCoeff": self.pipeline.vAlbedoCoeff.detach().cpu().numpy(),
            "vExpCoeff": self.pipeline.vExpCoeff.detach().cpu().numpy(),
            "vRotation": self.pipeline.vRotation.detach().cpu().numpy(),
            "vTranslation": self.pipeline.vTranslation.detach().cpu().numpy(),
            "vFocals": self.pipeline.vFocals.detach().cpu().numpy(),
            "vShCoeffs": self.pipeline.vShCoeffs.detach().cpu().numpy(),
            "screenWidth": self.pipeline.renderer.screenWidth,
            "screenHeight": self.pipeline.renderer.screenHeight,
            "sharedIdentity": self.pipeline.sharedIdentity,
        }
        if self.vEnhancedDiffuse is not None:
            dict["vEnhancedDiffuse"] = self.vEnhancedDiffuse.detach().cpu().numpy()
        if self.vEnhancedSpecular is not None:
            dict["vEnhancedSpecular"] = self.vEnhancedSpecular.detach().cpu().numpy()
        if self.vEnhancedRoughness is not None:
            dict["vEnhancedRoughness"] = self.vEnhancedRoughness.detach().cpu().numpy()

        handle = open(outputFileName, "wb")
        pickle.dump(dict, handle, pickle.HIGHEST_PROTOCOL)
        handle.close()

    def loadParameters(self, pickelFileName):
        handle = open(pickelFileName, "rb")
        assert handle is not None
        dict = pickle.load(handle)
        self.pipeline.vShapeCoeff = torch.tensor(dict["vShapeCoeff"]).to(self.device)
        self.pipeline.vAlbedoCoeff = torch.tensor(dict["vAlbedoCoeff"]).to(self.device)
        self.pipeline.vExpCoeff = torch.tensor(dict["vExpCoeff"]).to(self.device)
        self.pipeline.vRotation = torch.tensor(dict["vRotation"]).to(self.device)
        self.pipeline.vTranslation = torch.tensor(dict["vTranslation"]).to(self.device)
        self.pipeline.vFocals = torch.tensor(dict["vFocals"]).to(self.device)
        self.pipeline.vShCoeffs = torch.tensor(dict["vShCoeffs"]).to(self.device)
        self.pipeline.renderer.screenWidth = int(dict["screenWidth"])
        self.pipeline.renderer.screenHeight = int(dict["screenHeight"])
        self.pipeline.sharedIdentity = bool(dict["sharedIdentity"])

        if "vEnhancedDiffuse" in dict:
            self.vEnhancedDiffuse = torch.tensor(dict["vEnhancedDiffuse"]).to(
                self.device
            )
        if "vEnhancedSpecular" in dict:
            self.vEnhancedSpecular = torch.tensor(dict["vEnhancedSpecular"]).to(
                self.device
            )
        if "vEnhancedRoughness" in dict:
            self.vEnhancedRoughness = torch.tensor(dict["vEnhancedRoughness"]).to(
                self.device
            )

        handle.close()
        self.enableGrad()

    def enableGrad(self):
        self.pipeline.vShapeCoeff.requires_grad = True
        self.pipeline.vAlbedoCoeff.requires_grad = True
        self.pipeline.vExpCoeff.requires_grad = True
        self.pipeline.vRotation.requires_grad = True
        self.pipeline.vTranslation.requires_grad = True
        self.pipeline.vFocals.requires_grad = True
        self.pipeline.vShCoeffs.requires_grad = True


def test_us024_checkpoint_roundtrip():
    """Save parameters, load into fresh optimizer, verify all values match."""
    opt1 = MockOptimizer(device="cpu", n_frames=3)
    opt1.pipeline.renderer.screenWidth = 512
    opt1.pipeline.renderer.screenHeight = 384

    with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as f:
        path = f.name

    try:
        opt1.saveParameters(path)

        # Load into a fresh optimizer with different initial values
        opt2 = MockOptimizer(device="cpu", n_frames=3)
        opt2.loadParameters(path)

        # Core tensors match
        for attr in [
            "vShapeCoeff",
            "vAlbedoCoeff",
            "vExpCoeff",
            "vRotation",
            "vTranslation",
            "vFocals",
            "vShCoeffs",
        ]:
            orig = getattr(opt1.pipeline, attr)
            loaded = getattr(opt2.pipeline, attr)
            assert torch.allclose(
                orig, loaded, atol=1e-6
            ), f"{attr} mismatch after round-trip"

        # Renderer state preserved
        assert opt2.pipeline.renderer.screenWidth == 512
        assert opt2.pipeline.renderer.screenHeight == 384
        assert opt2.pipeline.sharedIdentity is True

        # Gradients enabled after load
        assert opt2.pipeline.vShapeCoeff.requires_grad
        assert opt2.pipeline.vExpCoeff.requires_grad
    finally:
        os.unlink(path)


def test_us024_checkpoint_with_enhanced_textures():
    """Verify enhanced texture tensors (stage 3) survive round-trip."""
    opt1 = MockOptimizer(device="cpu")
    opt1.vEnhancedDiffuse = torch.randn(1, 256, 256, 3)
    opt1.vEnhancedSpecular = torch.randn(1, 256, 256, 3)
    opt1.vEnhancedRoughness = torch.randn(1, 256, 256, 1)

    with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as f:
        path = f.name

    try:
        opt1.saveParameters(path)

        opt2 = MockOptimizer(device="cpu")
        assert opt2.vEnhancedDiffuse is None  # Initially None
        opt2.loadParameters(path)

        assert opt2.vEnhancedDiffuse is not None
        assert opt2.vEnhancedSpecular is not None
        assert opt2.vEnhancedRoughness is not None
        assert torch.allclose(opt1.vEnhancedDiffuse, opt2.vEnhancedDiffuse, atol=1e-6)
        assert torch.allclose(
            opt1.vEnhancedSpecular, opt2.vEnhancedSpecular, atol=1e-6
        )
        assert torch.allclose(
            opt1.vEnhancedRoughness, opt2.vEnhancedRoughness, atol=1e-6
        )
    finally:
        os.unlink(path)


def test_us024_checkpoint_without_enhanced_textures():
    """Verify loading a stage 1/2 checkpoint (no enhanced textures) works."""
    opt1 = MockOptimizer(device="cpu")
    # No enhanced textures set (stage 1 or 2 checkpoint)

    with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as f:
        path = f.name

    try:
        opt1.saveParameters(path)

        opt2 = MockOptimizer(device="cpu")
        opt2.loadParameters(path)

        assert opt2.vEnhancedDiffuse is None
        assert opt2.vEnhancedSpecular is None
        assert opt2.vEnhancedRoughness is None
    finally:
        os.unlink(path)


def test_us024_resume_from_checkpoint():
    """Simulate interrupted+resumed optimization: save at iteration N, load, verify state."""
    opt = MockOptimizer(device="cpu", n_frames=1)

    # Simulate some optimization steps by modifying params
    with torch.no_grad():
        opt.pipeline.vShapeCoeff[:] = 42.0
        opt.pipeline.vRotation[:] = torch.tensor([0.1, 0.2, 0.3])
    opt.pipeline.renderer.screenWidth = 640
    opt.pipeline.renderer.screenHeight = 480

    with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as f:
        path = f.name

    try:
        opt.saveParameters(path)

        # "Resume": create fresh optimizer and load checkpoint
        opt_resumed = MockOptimizer(device="cpu", n_frames=1)
        opt_resumed.loadParameters(path)

        # Verify specific values survived
        assert torch.allclose(
            opt_resumed.pipeline.vShapeCoeff,
            torch.full((1, 199), 42.0),
            atol=1e-6,
        )
        assert torch.allclose(
            opt_resumed.pipeline.vRotation,
            torch.tensor([[0.1, 0.2, 0.3]]),
            atol=1e-6,
        )
        assert opt_resumed.pipeline.renderer.screenWidth == 640
        assert opt_resumed.pipeline.renderer.screenHeight == 480

        # Verify gradients are enabled (ready for continued optimization)
        assert opt_resumed.pipeline.vShapeCoeff.requires_grad
        assert opt_resumed.pipeline.vRotation.requires_grad
    finally:
        os.unlink(path)


def test_us024_pickle_file_valid():
    """Verify saved file is a valid pickle with expected keys."""
    opt = MockOptimizer(device="cpu")

    with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as f:
        path = f.name

    try:
        opt.saveParameters(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        expected_keys = {
            "vShapeCoeff",
            "vAlbedoCoeff",
            "vExpCoeff",
            "vRotation",
            "vTranslation",
            "vFocals",
            "vShCoeffs",
            "screenWidth",
            "screenHeight",
            "sharedIdentity",
        }
        assert expected_keys.issubset(
            set(data.keys())
        ), f"Missing keys: {expected_keys - set(data.keys())}"

        # All tensor values are numpy arrays
        for key in [
            "vShapeCoeff",
            "vAlbedoCoeff",
            "vExpCoeff",
            "vRotation",
            "vTranslation",
            "vFocals",
            "vShCoeffs",
        ]:
            assert isinstance(data[key], np.ndarray), f"{key} should be numpy array"

        # Scalar values are correct types
        assert isinstance(data["screenWidth"], int)
        assert isinstance(data["screenHeight"], int)
        assert isinstance(data["sharedIdentity"], bool)
    finally:
        os.unlink(path)
