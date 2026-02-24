# NextGenFace Usage Guide

NextGenFace reconstructs a photorealistic 3D face from one or more photographs. It estimates head pose, facial shape, skin reflectance (diffuse, specular, roughness), and scene lighting using a 3-stage differentiable optimization pipeline built on Mitsuba 3.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Pipeline Stages](#3-pipeline-stages)
4. [Configuration Reference](#4-configuration-reference)
5. [The Rendering Pipeline](#5-the-rendering-pipeline)
6. [Working with Checkpoints](#6-working-with-checkpoints)
7. [Output Files](#7-output-files)
8. [Tips for Best Results](#8-tips-for-best-results)
9. [Programmatic Usage Examples](#9-programmatic-usage-examples)

---

## 1. Overview

NextGenFace takes a photograph of a face and produces:

- A textured 3D mesh with estimated diffuse, specular, and roughness maps
- An estimated environment map representing scene lighting
- Per-stage optimization checkpoints for resuming or inspecting intermediate results

The pipeline has three stages that run sequentially:

| Stage | Name | Method | What gets optimized |
|-------|------|--------|---------------------|
| 1 | Coarse Pose | Landmark loss | Head rotation, translation, expression, focal length |
| 2 | Photometric | Ray-traced photo loss | Shape, albedo (statistical), expression, pose, SH lighting |
| 3 | Texture Refinement | Ray-traced photo loss | Per-pixel diffuse, specular, roughness textures |

The Basel Face Model (BFM) provides a statistical prior for shape and skin reflectance. Mitsuba 3 performs differentiable ray tracing in Stage 2 and 3, propagating gradients from the rendered image back to all scene parameters via a DrJit-to-PyTorch autograd bridge.

---

## 2. Prerequisites

### Software

- Python >= 3.10
- PyTorch >= 2.0
- Mitsuba 3 (`pip install mitsuba`)
- DrJit (`pip install drjit`, installed with Mitsuba)
- MediaPipe (recommended) or face-alignment (`pip install mediapipe`)
- OpenCV, NumPy, tqdm

Install the package in editable mode:

```bash
uv pip install -e .
```

### LLVM on macOS

Mitsuba 3 requires LLVM for its LLVM-based rendering variants on macOS:

```bash
brew install llvm
```

Verify Mitsuba can load its variant:

```python
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
print(mi.variant())  # should print: llvm_ad_rgb
```

On CUDA-capable Linux machines, use the `cuda_ad_rgb` variant (selected automatically when `device = 'cuda'` is set in config).

### Basel Face Model Data (Manual Download Required)

NextGenFace requires BFM 2017 `.h5` files from the University of Basel. This step **cannot be automated** -- you must submit a license agreement form, and the download link is sent to your email.

1. Go to https://faces.dmi.unibas.ch/bfm/bfm2017.html
2. Fill out and submit the license agreement form
3. Wait for the download link to arrive by email
4. Download and extract the model files

The expected directory layout:

```
data/baselMorphableModel/
    model2009-publicmm1-bfm.h5          # shape and expression model
    albedoModel2020_FLAME_albedoPart.hh  # albedo PCA model
    landmark_62.txt                      # FAN landmark associations
    landmark_62_mp.txt                   # MediaPipe landmark associations
```

Place the directory at `./data/baselMorphableModel/` relative to your working directory, or set the environment variable:

```bash
export NEXTGENFACE_DATA_DIR=/path/to/baselMorphableModel
```

The `Config` class resolves the data path in this priority order:
1. `config.path` (set explicitly or via `.ini` file)
2. `NEXTGENFACE_DATA_DIR` environment variable
3. `./data/baselMorphableModel`
4. `./baselMorphableModel`

---

## 3. Pipeline Stages

### Stage 1: Coarse Pose Estimation (Landmarks)

**What it does.** Fits the head pose (rotation and translation in camera space), expression coefficients, and optionally the camera focal length by minimizing a 2D landmark loss. No rendering happens — this is pure geometry.

The landmark detector (FAN or MediaPipe) detects 62 facial keypoints in the input image. The optimizer projects the corresponding 3D vertices of the morphable model onto the image plane and minimizes the L2 distance between projected and detected landmarks.

**Parameters optimized:** `vRotation`, `vTranslation`, `vExpCoeff`, optionally `vFocals`

**Optimizer:** Adam, `lr = 0.02`, for `iterStep1` iterations (default: 2000)

**Loss:**
```
L = landmark_loss(projected_vertices, detected_landmarks)
  + 0.1 * expression_regularization
```

**Why this matters.** A good initial head pose dramatically speeds up Stage 2 and prevents it from getting stuck in local minima. If Stage 2 looks misaligned, increase `iterStep1` or switch to MediaPipe.

**Example:**

```python
from nextgenface import Config, Optimizer

config = Config()
config.fillFromDicFile("configs/default.ini")
config.iterStep1 = 2000

optimizer = Optimizer("./output", config)
optimizer.setImage("./data/input/s1.png")
optimizer.runStep1()
# Saves: output/checkpoints/stage1_output.pickle
# Saves: output/checkpoints/stage1_loss.png
# Saves: output/landmarks0.png  (detected landmarks overlay)
```

**Landmark detectors:**

- **MediaPipe** (recommended): More accurate, robust to pose variation. Requires `pip install mediapipe`.
- **FAN** (face-alignment): Alternative when MediaPipe is unavailable. Uses a different landmark association file (`landmark_62.txt` vs `landmark_62_mp.txt`).

Set via config: `lamdmarksDetectorType = 'mediapipe'` or `'fan'`.

---

### Stage 2: Photometric Optimization (Statistical Model)

**What it does.** Jointly optimizes shape identity, expression, statistical diffuse and specular albedo coefficients, head pose, and spherical harmonics (SH) scene lighting by minimizing the photometric difference between the Mitsuba-rendered image and the real photograph.

This is the core differentiable rendering stage. Mitsuba 3 ray-traces the face mesh with a physically-based BSDF (diffuse + specular + roughness), lit by an SH environment map. Gradients flow through the renderer to all scene parameters via the DrJit-PyTorch bridge.

**Parameters optimized:**

| Iterations | Parameters added |
|------------|-----------------|
| 0 | `vShCoeffs` (SH lighting), `vAlbedoCoeff` |
| 100 | `vShapeCoeff`, `vExpCoeff`, `vRotation`, `vTranslation` |

This two-phase warmup stabilizes lighting before refining geometry.

**Optimizer:** Adam with per-parameter learning rates. SH: `lr=0.005`, albedo: `lr=0.007`, shape/expression: `lr=0.01`, pose: `lr=0.0001`

**Loss:**
```
L = 1000 * photometric_loss(rendered, target)
  + weightLandmarksLossStep2 * landmark_loss
  + 0.0001 * sh_regularization
  + weightAlbedoReg * albedo_regularization
  + weightShapeReg * shape_regularization
  + weightExpressionReg * expression_regularization
```

Regularization uses the PCA variance of the morphable model as the reference (Mahalanobis-style), keeping coefficients within a statistically plausible range.

**Training samples:** `rtTrainingSamples = 8` (fast noisy renders during optimization)

**Example:**

```python
# Continuing from Stage 1
optimizer.runStep2()
# Saves: output/checkpoints/stage2_output.pickle
# Saves: output/checkpoints/stage2_loss.png
```

---

### Stage 3: Texture Refinement

**What it does.** Refines the statistical albedo textures from Stage 2 on a per-pixel basis. Instead of operating in the low-dimensional PCA coefficient space, Stage 3 directly optimizes the full-resolution UV texture maps for diffuse color, specular color, and roughness.

This captures fine-grained skin details (freckles, pores, asymmetries) that the statistical model cannot represent, while regularization prevents overfitting to lighting artifacts.

**Parameters optimized:** `vDiffTextures` (H×W×3), `vSpecTextures` (H×W×3), `vRoughTextures` (H×W×1)

**Optimizer:** Adam. Diffuse: `lr=0.005`, specular: `lr=0.02`, roughness: `lr=0.02`

**Regularizers applied to each texture:**

- **Symmetry**: Penalizes left-right asymmetry in UV space. Reduce if subject has natural asymmetry.
- **Consistency**: Penalizes deviation from the Stage 2 reference textures. Increase in harsh lighting to prevent shadows being baked into albedo.
- **Smoothness**: Penalizes high-frequency texture variation. Prevents noise artifacts.

**Loss:**
```
L = 1000 * photometric_loss(rendered, target)
  + 0.2 * (diffuse_reg + specular_reg + roughness_reg)
  + landmark_loss + shape_reg + expression_reg + sh_reg
```

**Example:**

```python
optimizer.runStep3()
# Saves: output/checkpoints/stage3_output.pickle
# Saves: output/checkpoints/stage3_loss.png
# Enhanced textures stored in optimizer.vEnhancedDiffuse/Specular/Roughness
```

After Stage 3, call `saveOutput()` to render and save final results:

```python
optimizer.saveOutput(samples=optimizer.config.rtSamples)
```

---

## 4. Configuration Reference

Config can be loaded from an `.ini` file or set programmatically on the `Config` object.

```python
from nextgenface import Config

config = Config()
config.fillFromDicFile("configs/default.ini")
# Or override individual values:
config.iterStep1 = 3000
config.device = 'cuda'
```

### Device

| Parameter | Type | Default (ini) | Description |
|-----------|------|---------------|-------------|
| `device` | str | `'cpu'` | Compute device: `'cpu'` or `'cuda'`. On CUDA, uses `cuda_ad_rgb` Mitsuba variant. Falls back to CPU if CUDA unavailable. |

### Landmarks

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lamdmarksDetectorType` | str | `'mediapipe'` | Landmark detector: `'mediapipe'` or `'fan'`. MediaPipe is significantly more accurate and recommended. Switch to `'fan'` only if MediaPipe is not installable. |

### Morphable Model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | `'./data/baselMorphableModel'` | Path to the BFM data directory. Override with `NEXTGENFACE_DATA_DIR` env var or set explicitly. |
| `textureResolution` | int | `512` | UV texture map resolution in pixels (square). `256` is faster; `512` gives finer detail. Must match the resolution supported by the morphable model UV layout. |
| `trimPca` | bool | `False` | If `True`, uses a reduced PCA basis (fewer eigenvectors). Speeds up computation slightly at the cost of expressivity. Leave `False` unless memory-constrained. |

### Spherical Harmonics

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bands` | int | `9` | Number of SH bands. `9` bands = 81 coefficients, capturing up to 8th-order lighting (directional shadows, highlights). Reducing to `4` speeds up but loses directional light accuracy. |
| `envMapRes` | int | `64` | Resolution of the environment map generated from SH coefficients (square: `envMapRes × envMapRes`). Does not affect optimization; only affects the saved `.png/.exr` output. |
| `smoothSh` | bool | `True` | If `True`, applies Gaussian smoothing to the SH environment map before saving. Recommended; raw SH maps can have ringing artifacts. |
| `saveExr` | bool | `False` | If `True`, saves the environment map as a 32-bit `.exr` file. If `False`, saves as `.png` (8-bit, tone-mapped). Use `.exr` for HDR downstream workflows. |

### Image

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `maxResolution` | int | `512` | Maximum image dimension (width or height). Larger images are scaled down proportionally. Reduce to `256` or `384` on hardware with limited VRAM. Larger values give higher-quality results but are slower and require more memory. |

### Camera

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camFocalLength` | float | `3000.0` | Initial focal length in pixels. Relates to physical focal length by: `f_pixels = f_mm * image_width / sensor_width_mm`. The default `3000.0` suits standard portrait lenses. |
| `optimizeFocalLength` | bool | `True` | If `True`, Stage 1 estimates the focal length from landmark fitting. If `False`, the initial `camFocalLength` is used throughout. Disable if you know the exact focal length (e.g., from EXIF data). |

### Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iterStep1` | int | `2000` | Number of Adam iterations for Stage 1 (landmark-based pose fitting). Increase to `3000` for challenging poses or when landmarks are noisy. |
| `iterStep2` | int | `400` | Number of Adam iterations for Stage 2 (statistical photometric optimization). Increase to `600–800` for difficult subjects or images. |
| `iterStep3` | int | `500` | Number of Adam iterations for Stage 3 (per-pixel texture refinement). Increase to `800–1000` for finer texture detail. |
| `weightLandmarksLossStep2` | float | `0.001` | Weight of landmark loss during Stage 2. Keeps the face from drifting too far from the landmark-fitted pose while optimizing lighting and albedo. Increase if Stage 2 loses alignment. |
| `weightLandmarksLossStep3` | float | `0.001` | Weight of landmark loss during Stage 3. Same role as above but for texture refinement. |
| `weightShapeReg` | float | `0.001` | PCA regularization weight for shape coefficients. Higher values keep shape closer to the average face. Increase if shape becomes unrealistic. |
| `weightExpressionReg` | float | `0.001` | PCA regularization weight for expression coefficients. Increase for neutral faces to suppress spurious expressions. |
| `weightAlbedoReg` | float | `0.001` | PCA regularization weight for albedo coefficients. Increase if albedo looks noisy or implausible after Stage 2. |

### Texture Regularizers (Stage 3)

These control the trade-off between texture detail and robustness to lighting/shadow artifacts.

#### Diffuse Texture

| Parameter | Type | Default (ini) | Description |
|-----------|------|---------------|-------------|
| `weightDiffuseSymmetryReg` | float | `300.0` | Penalizes left-right albedo asymmetry. Increase (`500+`) when harsh directional lighting causes shadows to be baked into the diffuse map. |
| `weightDiffuseConsistencyReg` | float | `100.0` | Penalizes deviation from Stage 2 diffuse texture. Keeps refinement close to the statistical prior. Increase with harsh lighting. |
| `weightDiffuseSmoothnessReg` | float | `0.001` | Penalizes high-frequency texture variation. Increase if diffuse map shows noise. |

#### Specular Texture

| Parameter | Type | Default (ini) | Description |
|-----------|------|---------------|-------------|
| `weightSpecularSymmetryReg` | float | `200.0` | Symmetry regularizer for specular texture. High values enforce bilateral symmetry on skin specularity. |
| `weightSpecularConsistencyReg` | float | `2.0` | Consistency with Stage 2 specular texture. Lower than diffuse because specular can vary more across the face. |
| `weightSpecularSmoothnessReg` | float | `0.001` | Smoothness for specular texture. Increase to suppress high-frequency specular noise. |

#### Roughness Texture

| Parameter | Type | Default (ini) | Description |
|-----------|------|---------------|-------------|
| `weightRoughnessSymmetryReg` | float | `200.0` | Symmetry regularizer for roughness texture. |
| `weightRoughnessConsistencyReg` | float | `0.0` | Consistency with Stage 2 roughness. Default `0.0` allows roughness to adapt freely per-pixel. |
| `weightRoughnessSmoothnessReg` | float | `0.002` | Smoothness for roughness texture. Slightly higher default than diffuse/specular because roughness should vary gradually across skin regions. |

### Debug

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `debugFrequency` | int | `30` | Save debug frames every N iterations during optimization. Frames are written to `output/debug/`. Set to `0` to disable (faster). |
| `saveIntermediateStage` | bool | `False` | If `True`, saves full rendered output after Stage 1 and Stage 2 (not just the final Stage 3 result). Useful for inspecting per-stage quality. Stage 3 output is always saved. |
| `verbose` | bool | `False` | Print loss values to stdout at each iteration. Useful for debugging loss divergence. |

### Ray Tracing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rtSamples` | int | `10000` | Samples per pixel for the **final output render** (after optimization completes). Higher values produce cleaner, less noisy final images. `10000` is the recommended minimum; `20000` gives publication-quality renders. |
| `rtTrainingSamples` | int | `8` | Samples per pixel **during training** (Stages 2 and 3). Low values (4–16) are intentionally noisy but fast. Increasing slows optimization significantly without proportional benefit since the optimizer averages over many iterations. |

---

## 5. The Rendering Pipeline

### Initialization

`Pipeline.__init__()` sets up all rendering components:

```python
self.camera = Camera(device)            # perspective projection
self.sh = SphericalHarmonics(envMapRes) # SH-to-envmap conversion
self.morphableModel = MorphableModel(   # BFM loader
    path, textureResolution, trimPca,
    landmarksPathName, device
)
self.renderer = Renderer(               # Mitsuba scene builder + renderer
    rtTrainingSamples, bounces=1, device
)
```

### Rendering a Frame

`Pipeline.render()` follows this sequence for each frame:

1. **Compute normals** via `morphableModel.meshNormals.computeNormals(cameraVerts)` — vertex normals from the deformed mesh geometry.

2. **Generate environment map** via `self.sh.toEnvMap(vShCoeffs)` — converts the 81 SH coefficients (per RGB channel) into an HDR `[envMapRes, envMapRes, 3]` environment map texture.

3. **Build Mitsuba scenes** via `renderer.buildScenes(...)` — assembles one Mitsuba scene per frame from:
   - `_camera.py`: perspective camera from focal length and screen dimensions
   - `_mesh.py`: face mesh with UV-mapped BSDF
   - `_material.py`: principled BSDF (diffuse + specular + roughness)
   - `_envmap.py`: environment map emitter
   - `_scene.py`: assembles all components into a Mitsuba scene dict

4. **Render** — two paths depending on gradient mode:
   - `torch.is_grad_enabled() == False`: calls `render_scenes()` directly (standard Mitsuba forward render, no AD overhead)
   - `torch.is_grad_enabled() == True`: calls `_differentiable_render()` which uses the gradient bridge

5. **Albedo extraction** via `renderer.renderAlbedo()` — rebuilds scenes in AOV (Arbitrary Output Variable) integrator mode, which outputs the raw albedo texture projected onto the image plane without lighting.

### The Gradient Bridge

`_gradient_bridge.py` implements a `torch.autograd.Function` (`_DiffRender`) that connects Mitsuba's DrJit AD system to PyTorch's autograd:

**Forward pass:**
1. Receives a Mitsuba scene and a dict mapping parameter paths (e.g., `"face_mesh.bsdf.base_color.data"`) to PyTorch tensors
2. Sets scene parameters from the detached tensors via `mi.traverse()`
3. Calls `mi.render(scene, spp=spp)` — standard Mitsuba forward render
4. Returns the rendered image as a PyTorch tensor

**Backward pass:**
1. Recreates DrJit AD-tracked values from the saved input tensors
2. Sets them back into the scene via `mi.traverse()` with `dr.enable_grad()`
3. Re-renders with `mi.render(scene, params=params, spp=spp)` — Mitsuba records the AD graph
4. Seeds the output gradient: `dr.set_grad(img, grad_output)`
5. Backpropagates through DrJit: `dr.enqueue(ADMode.Backward, img)` + `dr.traverse(ADMode.Backward)`
6. Extracts gradients from DrJit values and returns them as PyTorch tensors

The scene parameters bridged with gradients are:
- `face_mesh.vertex_positions` — mesh vertices
- `face_mesh.vertex_normals` — vertex normals
- `face_mesh.bsdf.base_color.data` — diffuse texture
- `face_mesh.bsdf.roughness.data` — roughness texture
- `envmap.data` — environment map

---

## 6. Working with Checkpoints

Checkpoints are Python pickle files saved after each optimization stage. They contain all scene parameters as NumPy arrays.

### Checkpoint Contents

```python
{
    'vShapeCoeff':   np.ndarray,  # [n or 1, shapeBasisSize]
    'vAlbedoCoeff':  np.ndarray,  # [n or 1, albedoBasisSize]
    'vExpCoeff':     np.ndarray,  # [n, expBasisSize]
    'vRotation':     np.ndarray,  # [n, 3]
    'vTranslation':  np.ndarray,  # [n, 3]
    'vFocals':       np.ndarray,  # [n]
    'vShCoeffs':     np.ndarray,  # [n, bands*bands, 3]
    'screenWidth':   int,
    'screenHeight':  int,
    'sharedIdentity': bool,
    # Optional (present only if Stage 3 completed):
    'vEnhancedDiffuse':   np.ndarray,  # [n or 1, H, W, 3]
    'vEnhancedSpecular':  np.ndarray,  # [n or 1, H, W, 3]
    'vEnhancedRoughness': np.ndarray,  # [n or 1, H, W, 1]
}
```

### Resuming Optimization

Use the CLI to resume from a checkpoint and skip earlier stages:

```bash
nextgenface-reconstruct \
    --input ./data/input/s1.png \
    --output ./output \
    --config configs/default.ini \
    --checkpoint ./output/s1/checkpoints/stage2_output.pickle \
    --skipStage1 \
    --skipStage2
```

Programmatically:

```python
optimizer.setImage("./data/input/s1.png")
optimizer.loadParameters("./output/s1/checkpoints/stage2_output.pickle")
optimizer.runStep3()
optimizer.saveOutput(samples=config.rtSamples)
```

### Saving and Loading Manually

```python
# Save current state
optimizer.saveParameters("./my_checkpoint.pickle")

# Load and resume
optimizer.setImage("./data/input/s1.png")
optimizer.loadParameters("./my_checkpoint.pickle")
# Parameters are automatically put in grad-enabled mode after loading
```

---

## 7. Output Files

All output is written to `outputDir/` (or `outputDir/<image_basename>/` when using the CLI).

| File | Description |
|------|-------------|
| `render_{i}.png` | Composite strip (7 panels side-by-side): input image, overlay (reconstruction on input), full reconstruction, illumination only, diffuse albedo, specular albedo, roughness albedo |
| `diffuseMap_{i}.png` | Full-resolution diffuse albedo UV texture map |
| `specularMap_{i}.png` | Full-resolution specular albedo UV texture map |
| `roughnessMap_{i}.png` | Full-resolution roughness UV texture map (grayscale saved as RGB) |
| `mesh{i}.obj` | 3D mesh in camera space with UV coordinates |
| `material{i}.mtl` | Material file referencing `diffuseMap_{i}.png` |
| `envMap_{i}.png` or `.exr` | Estimated scene environment map from SH coefficients |
| `landmarks{i}.png` | Detected 2D landmarks overlaid on input image |
| `checkpoints/stage1_output.pickle` | Scene parameters after Stage 1 |
| `checkpoints/stage2_output.pickle` | Scene parameters after Stage 2 |
| `checkpoints/stage3_output.pickle` | Scene parameters + enhanced textures after Stage 3 |
| `checkpoints/stage1_loss.png` | Loss curve plot for Stage 1 |
| `checkpoints/stage2_loss.png` | Loss curve plot for Stage 2 |
| `checkpoints/stage3_loss.png` | Loss curve plot for Stage 3 |
| `debug/debug1_iter{N}_frame{i}.png` | Debug frames during Stage 2 (if `debugFrequency > 0`) |
| `debug/debug2_iter{N}_frame{i}.png` | Debug frames during Stage 3 |

The index `{i}` corresponds to the frame number (0-based). For single-image input, all output uses `i = 0`.

When `sharedIdentity = True` (multiple images, same person), texture maps use index `0` for all frames since the identity is shared.

---

## 8. Tips for Best Results

**Image quality.** Use well-lit, approximately frontal portrait photographs. Even, diffuse lighting (overcast outdoors, softbox studio) gives the most accurate albedo recovery. Harsh directional lighting can be handled but requires tuning regularizer weights.

**Harsh lighting (shadows).** If shadows are being baked into the diffuse texture, increase `weightDiffuseSymmetryReg` (e.g., `500–1000`) and `weightDiffuseConsistencyReg` (e.g., `200–400`). Use `configs/shadows.ini` as a starting point — it has pre-tuned values for challenging lighting conditions.

**Multiple images of the same person.** Provide a directory of images and pass `sharedIdentity=True`. The optimizer will share the shape and albedo identity across all frames while fitting per-frame expression, pose, and lighting independently. This yields significantly better albedo estimates because the lighting-from-multiple-views disambiguation helps separate reflectance from illumination.

```bash
nextgenface-reconstruct \
    --input ./data/input/person_images/ \
    --output ./output \
    --config configs/default.ini \
    --sharedIdentity
```

**Limited hardware (low VRAM).** Reduce `maxResolution` to `256` or `384`. This is the single most effective knob for reducing memory usage. Also reduce `textureResolution` to `256`. The optimizer automatically halves render resolution and retries on out-of-memory errors during final output rendering.

**Higher quality final renders.** Increase `rtSamples` to `20000` or higher. This only affects the final output render (after optimization), not the optimization speed. On NVIDIA RTX GPUs with hardware ray tracing, `20000` samples renders in seconds.

**Slow optimization.** Reduce `iterStep2` to `200` and `iterStep3` to `300` for a quick preview. For publication quality, use `iterStep2 = 600` and `iterStep3 = 800`.

**Focal length.** If you know the exact focal length (from EXIF: `f_pixels = f_mm * image_width / sensor_width_mm`), set `optimizeFocalLength = False` and provide the correct `camFocalLength`. This eliminates one degree of freedom from Stage 1 and often improves pose accuracy.

**Landmark detection failures.** If MediaPipe fails to detect a face (returns no landmarks), the image will be skipped. Check `landmarks0.png` in the output directory to verify detection quality before running full optimization.

---

## 9. Programmatic Usage Examples

### Basic Reconstruction

```python
from nextgenface import Config, Optimizer

config = Config()
config.fillFromDicFile("configs/default.ini")

optimizer = Optimizer("./output/result", config)
optimizer.run(
    imagePathOrDir="./data/input/s1.png",
    sharedIdentity=False,
)
# Output saved to ./output/result/
```

### Batch Processing (Multiple Different Subjects)

```python
import os
from nextgenface import Config, Optimizer

config = Config()
config.fillFromDicFile("configs/default.ini")
config.debugFrequency = 0  # disable debug frames for speed

image_dir = "./data/input/subjects/"
output_base = "./output/batch/"

for filename in os.listdir(image_dir):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    name = os.path.splitext(filename)[0]
    image_path = os.path.join(image_dir, filename)
    output_dir = os.path.join(output_base, name)

    optimizer = Optimizer(output_dir, config)
    optimizer.run(imagePathOrDir=image_path)
    print(f"Done: {name}")
```

### Loading a Checkpoint and Inspecting Parameters

```python
import pickle
import numpy as np

checkpoint_path = "./output/result/checkpoints/stage3_output.pickle"
with open(checkpoint_path, "rb") as f:
    params = pickle.load(f)

print("Shape coefficients:", params["vShapeCoeff"].shape)
print("Albedo coefficients:", params["vAlbedoCoeff"].shape)
print("Expression coefficients:", params["vExpCoeff"].shape)
print("Rotation (Euler angles):", params["vRotation"])
print("Translation:", params["vTranslation"])
print("Focal length (px):", params["vFocals"])
print("SH coefficients shape:", params["vShCoeffs"].shape)
print("Screen size:", params["screenWidth"], "x", params["screenHeight"])

if "vEnhancedDiffuse" in params:
    print("Enhanced diffuse texture:", params["vEnhancedDiffuse"].shape)
    print("Enhanced specular texture:", params["vEnhancedSpecular"].shape)
    print("Enhanced roughness texture:", params["vEnhancedRoughness"].shape)
```

### Extracting Just the 3D Mesh

```python
import pickle
import torch
from nextgenface import Config, Pipeline
from geometry.obj_export import saveObj

config = Config()
config.fillFromDicFile("configs/default.ini")

# Load pipeline (morphable model, renderer)
pipeline = Pipeline(config)

# Load checkpoint
with open("./output/result/checkpoints/stage3_output.pickle", "rb") as f:
    params = pickle.load(f)

device = config.device
pipeline.initSceneParameters(n=1)
pipeline.vShapeCoeff = torch.tensor(params["vShapeCoeff"]).to(device)
pipeline.vExpCoeff = torch.tensor(params["vExpCoeff"]).to(device)
pipeline.vRotation = torch.tensor(params["vRotation"]).to(device)
pipeline.vTranslation = torch.tensor(params["vTranslation"]).to(device)
pipeline.vFocals = torch.tensor(params["vFocals"]).to(device)
pipeline.renderer.screenWidth = int(params["screenWidth"])
pipeline.renderer.screenHeight = int(params["screenHeight"])

# Compute camera-space vertices and normals
vertices = pipeline.computeShape()
camera_verts = pipeline.transformVertices(vertices)
normals = pipeline.morphableModel.computeNormals(camera_verts)

# Save mesh
saveObj(
    "./output/mesh.obj",
    "material.mtl",
    camera_verts[0],
    pipeline.faces32,
    normals[0],
    pipeline.morphableModel.uvMap,
    "diffuseMap_0.png",
)
print("Mesh saved to ./output/mesh.obj")
```

### Custom Optimization Loop (Stage 2 Only, Custom Learning Rates)

```python
import torch
from nextgenface import Config, Optimizer

config = Config()
config.fillFromDicFile("configs/default.ini")
config.iterStep2 = 600
config.weightShapeReg = 0.0005   # looser shape prior

optimizer = Optimizer("./output/custom", config)
optimizer.setImage("./data/input/s1.png")

# Run Stage 1 to get a good initial pose
optimizer.runStep1()

# Run Stage 2 with custom settings already applied via config
optimizer.runStep2()

# Skip Stage 3 and save with the statistical textures from Stage 2
# (vEnhancedDiffuse will be None, so saveOutput uses Stage 2 textures)
optimizer.saveOutput(samples=optimizer.config.rtSamples)
```

### Multi-Image Reconstruction (Same Person, Shared Identity)

```python
from nextgenface import Config, Optimizer

config = Config()
config.fillFromDicFile("configs/default.ini")
config.iterStep2 = 600   # more iterations benefits multi-image

optimizer = Optimizer("./output/person_A", config)
optimizer.run(
    imagePathOrDir="./data/input/person_A/",  # directory of images
    sharedIdentity=True,
)
# Shape and albedo are shared; per-image: expression, pose, lighting
```
