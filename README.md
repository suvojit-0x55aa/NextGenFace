# NextGenFace

A modernized fork of [NextFace](https://github.com/abdallahdib/NextFace) by Abdallah Dib — high-fidelity 3D face reconstruction from monocular images using Mitsuba 3 differentiable ray tracing. Scene attributes — 3D geometry, reflectance (diffuse, specular and roughness), pose, camera parameters, and scene illumination — are estimated via a first-order optimization method that fits a statistical morphable model to input image(s).

<p align="center">
<img src="data/resources/emily.png" style="float: left; width: 23%; margin-right: 1%; margin-bottom: 0.5em;">
<img src="data/resources/emily.gif" style="float: left; margin-right: 1%; margin-bottom: 0.5em;">
<img src="data/resources/beard.png" style="float: left; width: 23%; margin-right: 1%; margin-bottom: 0.5em;">
<img src="data/resources/beard.gif" style="float: left; margin-right: 1%; margin-bottom: 0.5em;">
<img src="data/resources/visual.jpg">
</p>

<p align="center">
<strong>Demo on YouTube:</strong>
</p>
<p align="center">
<a href="http://www.youtube.com/watch?v=bPFp0oZ9plg" title="Practical Face Reconstruction via Differentiable Ray Tracing">
<img src="http://img.youtube.com/vi/bPFp0oZ9plg/0.jpg" alt="Practical Face Reconstruction via Differentiable Ray Tracing" />
</a>
</p>

<p align="center">
<img src="data/resources/results1.gif" style="width: 50%;">
</p>


## What This Port Changes

This is a fork of the original [NextFace](https://github.com/abdallahdib/NextFace) that replaces the unmaintained PyRedner renderer with [Mitsuba 3](https://www.mitsuba-renderer.org/):

| Original | This Port |
|----------|-----------|
| PyRedner (unmaintained) | Mitsuba 3 >= 3.5 |
| conda environment | pip/uv installable package |
| Flat file layout | Modular `src/` packages |
| Python 3.7+ | Python >= 3.10 |
| PyTorch 1.x | PyTorch >= 2.0 |
| No test suite | 148 tests |

The original 3-stage optimization pipeline is preserved exactly.


## Features

- Reconstructs face at high fidelity from single or multiple RGB images
- Estimates face geometry
- Estimates detailed face reflectance (diffuse, specular and roughness)
- Estimates scene illumination with spherical harmonics
- Estimates head pose and orientation
- Runs on both CPU and CUDA-enabled GPU
- **Mitsuba 3 differentiable ray tracing** (replaces PyRedner)
- **Automatic variant selection**: CUDA > LLVM > scalar fallback
- **DrJit-PyTorch gradient bridge** for seamless optimization
- **Installable package** with `nextgenface-reconstruct` CLI entry point
- **MediaPipe** landmark detection (more accurate than FAN; FAN still available)


## Installation

### 1. Clone

```bash
git clone <repo-url>
cd nfaceport
```

### 2. Install the package

```bash
# Recommended: uv
uv pip install -e .

# Or: standard pip
pip install -e .
```

### 3. Download required model files

The Basel Face Model and AlbedoMM are not included and must be downloaded separately.

**Basel Face Model 2017**
- Fill the form at https://faces.dmi.unibas.ch/bfm/bfm2017.html — a direct download link is sent instantly to your inbox
- Download `model2017-1_face12_nomouth.h5`
- Place it in `data/baselMorphableModel/`

**AlbedoMM**
- Download `albedoModel2020_face12_albedoPart.h5` from https://github.com/waps101/AlbedoMM/releases/download/v1.0/albedoModel2020_face12_albedoPart.h5
- Place it in `data/baselMorphableModel/`

### 4. (macOS) LLVM backend for differentiable rendering

On macOS, CUDA is unavailable so Mitsuba 3 falls back to LLVM:

```bash
brew install llvm
export DRJIT_LIBLLVM_PATH=/opt/homebrew/opt/llvm/lib/libLLVM.dylib
```

Add the export to your shell profile (`~/.zshrc` or `~/.bashrc`) to persist it.

### Dependencies

| Package | Version |
|---------|---------|
| Python | >= 3.10 |
| mitsuba | >= 3.5 |
| torch | >= 2.0 |
| torchvision | >= 0.25.0 |
| numpy | latest |
| opencv-python | latest |
| h5py | latest |
| tqdm | latest |
| mediapipe | latest |
| face-alignment | >= 1.4.1 |


## Agent-Assisted Installation

Using an AI coding agent? You can install NextGenFace with this prompt:

```
Clone and install the NextGenFace project. Use `uv pip install -e .` for installation.
On macOS, install LLVM via Homebrew and set
DRJIT_LIBLLVM_PATH=/opt/homebrew/opt/llvm/lib/libLLVM.dylib for the differentiable
rendering backend.
```

**Note:** The Basel Face Model 2017 (.h5) cannot be downloaded automatically. It requires submitting a license agreement form at https://faces.dmi.unibas.ch/bfm/bfm2017.html and the download link is sent by email. The AlbedoMM can be downloaded directly from https://github.com/waps101/AlbedoMM/releases. Place both files in `data/baselMorphableModel/`.


## How to Use

### CLI

```bash
# Single image
nextgenface-reconstruct --input path/to/image.png --output ./output/

# Batch reconstruction (images with same resolution)
nextgenface-reconstruct --input path/to/folder/ --output ./output/

# Same person, multiple images (shared identity)
nextgenface-reconstruct --sharedIdentity --input path/to/folder/ --output ./output/

# Custom configuration
nextgenface-reconstruct --config configs/shadows.ini --input image.png --output ./output/

# Resume from checkpoint (skip stages 1 and 2)
nextgenface-reconstruct \
  --checkpoint output/checkpoints/stage2_output.pickle \
  --skipStage1 --skipStage2 \
  --input image.png --output ./output/
```

The `--sharedIdentity` flag tells the optimizer that all images belong to the same person. Shape identity and face reflectance attributes are then shared across all images, which generally improves reflectance and geometry estimation.

### Python API

```python
from nextgenface import Renderer, Pipeline, Optimizer, Config

config = Config()
config.fillFromDicFile("configs/default.ini")

optimizer = Optimizer("./output", config)
optimizer.run("./data/input/s1.png")
```


## Configuration

Config files live in `configs/`:

| File | Description |
|------|-------------|
| `configs/default.ini` | Standard settings |
| `configs/shadows.ini` | Higher regularizer weights for harsh lighting |

Key settings:

| Setting | Description |
|---------|-------------|
| `device` | `cpu` or `cuda` |
| `landmarksDetectorType` | `mediapipe` (default) or `fan` |
| `textureResolution` | UV map resolution (512, 1024, 2048) |
| `bands` | Spherical harmonics bands (default: 9) |
| `maxResolution` | Input image resize limit |
| Iterations | Per-stage iteration counts |
| Regularizer weights | Symmetry, consistency, smoothness per map |
| Ray tracing samples | Samples per pixel for Mitsuba 3 |

For harsh lighting with residual shadows, try `configs/shadows.ini` or increase these regularizer weights in your config:

- Diffuse: `weightDiffuseSymmetryReg`, `weightDiffuseConsistencyReg`
- Specular: `weightSpecularSymmetryReg`, `weightSpecularConsistencyReg`
- Roughness: `weightRoughnessSymmetryReg`, `weightRoughnessConsistencyReg`


## How It Works

NextGenFace reproduces the optimization strategy from the [original paper](https://arxiv.org/abs/2101.05356) using a 3-stage pipeline, now powered by Mitsuba 3 for differentiable rendering:

**Stage 1 — Coarse geometric alignment**
Face expression and head pose are estimated by minimizing the geometric loss between 2D landmarks and their corresponding face vertices. No rendering is involved. This produces a good initialization for Stage 2.

**Stage 2 — Photometric optimization**
Shape identity/expression, statistical diffuse and specular albedos, head pose, and scene illumination are estimated by minimizing the photometric consistency loss between the Mitsuba 3 ray-traced image and the real image. Spherical harmonics (9 bands by default) capture scene lighting.

**Stage 3 — Texture refinement**
Per-pixel albedo optimization refines the statistical albedos from Stage 2 to capture finer detail. Consistency, symmetry, and smoothness regularizers prevent overfitting and add robustness against lighting conditions.

The DrJit-PyTorch gradient bridge propagates gradients from Mitsuba 3 back through the PyTorch optimization loop.


## Output

Optimization takes approximately 4-5 minutes depending on GPU performance. Per input image, the following files are produced:

| File | Contents |
|------|----------|
| `render_{i}.png` | Input image, overlay, reconstruction, diffuse/specular/roughness maps |
| `diffuseMap_{i}.png` | Estimated diffuse map in UV space |
| `specularMap_{i}.png` | Estimated specular map in UV space |
| `roughnessMap_{i}.png` | Estimated roughness map in UV space |
| `mesh{i}.obj` | 3D mesh of the reconstructed face |
| `checkpoints/stage{1,2,3}_output.pickle` | Serialized scene attributes for resuming |


## Project Structure

```
src/
  nextgenface/    # Public API (Renderer, Pipeline, Optimizer, Config), CLI entry point
  rendering/      # Mitsuba 3 renderer, DrJit-PyTorch gradient bridge, scene assembly
  facemodel/      # Basel Face Model loader, mesh normals, normal sampler
  geometry/       # Camera model, projection, spherical harmonics, OBJ export
  landmarks/      # FAN and MediaPipe landmark detectors, visualization
  imaging/        # Image I/O, Gaussian smoothing
  optim/          # Pipeline, optimizer, config parser, texture loss
configs/          # INI configuration files (default.ini, shadows.ini)
data/             # Model data (baselMorphableModel/), sample inputs, resources
tests/            # 148 tests
scripts/          # reconstruct.py, replay.py
docs/             # demo.ipynb
```


## Good Practices for Best Results

- Use images taken in good lighting conditions with no harsh shadows and even illumination.
- For single-image reconstruction, use a frontal face to ensure complete diffuse/specular/roughness recovery (only visible surface is reconstructed).
- Avoid extreme expressions — the underlying morphable model may not capture them accurately.
- For intrinsic (view-independent) reflectance maps, use multiple images of the same subject with `--sharedIdentity`.
- On low-memory GPUs, reduce `maxResolution` in the config to trade quality for speed.


## Limitations

- **Landmark sensitivity**: Stage 1 relies on landmark detectors. Inaccurate landmarks lead to sub-optimal reconstruction. MediaPipe is more robust than FAN for most cases.
- **Speed**: Optimization speed scales with input image resolution. Mitsuba 3 ray tracing is more physically accurate but computationally heavier than rasterization-based alternatives.
- **Geometry detail**: The Basel Face Model cannot represent fine geometric details (wrinkles, pores). These may appear baked into the albedo maps.
- **Spherical harmonics**: Models only distant (infinite) lighting. Under strong directional shadows, residual artifacts may appear in albedo estimates.
- **Single image**: Using a single image to estimate face attributes is an ill-posed problem; resulting reflectance maps are view-dependent.


## Porting Notes

Summary of changes from the original NextFace codebase:

- **PyRedner → Mitsuba 3**: All rendering code was rewritten. The original PyRedner-based forward/backward pass is replaced by Mitsuba 3 scene assembly and rendering, with a custom DrJit-PyTorch gradient bridge (`src/rendering/_gradient_bridge.py`) to route gradients back into PyTorch autograd.
- **Automatic variant selection**: At runtime, the renderer selects `cuda_ad_rgb` > `llvm_ad_rgb` > `scalar_rgb` based on hardware availability.
- **Package structure**: Flat file layout reorganized into `src/` with distinct subpackages (`rendering`, `facemodel`, `geometry`, `landmarks`, `imaging`, `optim`) and a thin `nextgenface` wrapper for public API re-exports.
- **Installation**: conda + INSTALL script replaced by `pyproject.toml` with hatchling; installable via `pip` or `uv`.
- **Python/PyTorch**: Minimum versions bumped to Python 3.10 and PyTorch 2.0.
- **MediaPipe**: Updated to the current MediaPipe API (the original used an older interface).
- **Test suite**: 148 tests added covering rendering, geometry, face model, imaging, landmarks, and optimization components.


## License

NextGenFace is available under the GPL license for research and educational purposes only. See [LICENSE](LICENSE) for details.


## Acknowledgements

- UV map from [parametric-face-image-generator](https://github.com/unibas-gravis/parametric-face-image-generator/blob/master/data/regions/face12.json)
- Landmark association from [Face2face](https://github.com/kimoktm/Face2face/blob/master/data/custom_mapping.txt)
- Differentiable rendering via [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3)
- Albedo model from [AlbedoMM](https://github.com/waps101/AlbedoMM/)
- MediaPipe landmark detector contributed by [Jack Saunders](https://researchportal.bath.ac.uk/en/persons/jack-saunders)
- Original NextFace by [Abdallah Dib](https://github.com/abdallahdib/NextFace)


## Contact

**Port author**: Suvojit Manna (twitter: @_smanna)

**Original author**: Abdallah Dib (deeb.abdallah @at gmail / twitter: abdallah_dib)


## Citation

If you use this work, please cite the original NextFace papers:

```bibtex
@inproceedings{dib2021practical,
  title={Practical face reconstruction via differentiable ray tracing},
  author={Dib, Abdallah and Bharaj, Gaurav and Ahn, Junghyun and Th{\'e}bault, C{\'e}dric and Gosselin, Philippe and Romeo, Marco and Chevallier, Louis},
  booktitle={Computer Graphics Forum},
  volume={40},
  number={2},
  pages={153--164},
  year={2021},
  organization={Wiley Online Library}
}

@inproceedings{dib2021towards,
  title={Towards High Fidelity Monocular Face Reconstruction with Rich Reflectance using Self-supervised Learning and Ray Tracing},
  author={Dib, Abdallah and Thebault, Cedric and Ahn, Junghyun and Gosselin, Philippe-Henri and Theobalt, Christian and Chevallier, Louis},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12819--12829},
  year={2021}
}

@article{dib2022s2f2,
  title={S2F2: Self-Supervised High Fidelity Face Reconstruction from Monocular Image},
  author={Dib, Abdallah and Ahn, Junghyun and Thebault, Cedric and Gosselin, Philippe-Henri and Chevallier, Louis},
  journal={arXiv preprint arXiv:2203.07732},
  year={2022}
}
```
