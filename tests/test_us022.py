"""US-022: End-to-end single image reconstruction.

Tests that the full pipeline runs to completion on a single image
with no pyredner imports anywhere in the call stack.
"""

import importlib
import os
import sys
import types

import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
H5_MODEL_PATH = os.path.join(DATA_DIR, "baselMorphableModel", "model2017-1_face12_nomouth.h5")
INPUT_IMAGE = os.path.join(DATA_DIR, "input", "s1.png")
has_model = os.path.isfile(H5_MODEL_PATH)
has_input = os.path.isfile(INPUT_IMAGE)


def _get_all_imports(module, visited=None):
    """Recursively collect all imported module names from a module."""
    if visited is None:
        visited = set()
    if module.__name__ in visited:
        return visited
    visited.add(module.__name__)
    for attr_name in dir(module):
        attr = getattr(module, attr_name, None)
        if isinstance(attr, types.ModuleType) and attr.__name__ not in visited:
            _get_all_imports(attr, visited)
    return visited


class TestNoPyrednerImports:
    """Verify no pyredner/redner imports in the active call stack."""

    def test_pipeline_import_no_pyredner(self):
        """Importing pipeline should not pull in pyredner."""
        # Clear any cached imports
        mods_before = set(sys.modules.keys())

        # Import pipeline (which now imports rendering.renderer)
        import optim.pipeline as pipeline  # noqa: F811

        mods_after = set(sys.modules.keys())
        new_mods = mods_after - mods_before

        pyredner_mods = [m for m in new_mods if "pyredner" in m or m == "redner"]
        assert pyredner_mods == [], (
            f"pyredner/redner modules imported by pipeline: {pyredner_mods}"
        )

    def test_renderer_mitsuba_no_pyredner(self):
        """renderer_mitsuba.py must not reference pyredner."""
        import rendering.renderer as renderer_mitsuba

        source_file = renderer_mitsuba.__file__
        with open(source_file) as f:
            source = f.read()
        assert "pyredner" not in source
        assert "import redner" not in source

    def test_image_no_pyredner(self):
        """image.py must not reference pyredner."""
        import imaging.image as image

        source_file = image.__file__
        with open(source_file) as f:
            source = f.read()
        assert "pyredner" not in source
        assert "import redner" not in source

    def test_pipeline_source_no_pyredner(self):
        """pipeline.py source must not reference pyredner."""
        import optim.pipeline as pipeline

        source_file = pipeline.__file__
        with open(source_file) as f:
            source = f.read()
        assert "pyredner" not in source
        assert "import redner" not in source


@pytest.mark.skipif(not has_model, reason="Basel Face Model .h5 not available")
@pytest.mark.skipif(not has_input, reason="Input image s1.png not available")
class TestE2ESingleImage:
    """End-to-end single image reconstruction test.

    Requires the Basel Face Model .h5 files to be present.
    Uses minimal iterations for speed.
    """

    def test_e2e_single_image(self, tmp_path):
        """Run the full pipeline on s1.png and verify outputs."""
        import torch
        from optim.config import Config
        from optim.optimizer import Optimizer

        # Create a minimal config for fast test
        config = Config()
        config.device = "cpu"
        config.path = os.path.join(DATA_DIR, "baselMorphableModel")
        config.textureResolution = 256
        config.maxResolution = 64  # tiny resolution for speed
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

        # Verify expected output files exist
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

        # Verify no pyredner was loaded during the run
        pyredner_mods = [m for m in sys.modules if "pyredner" in m or m == "redner"]
        assert pyredner_mods == [], (
            f"pyredner/redner loaded during e2e run: {pyredner_mods}"
        )
