# Mitsuba 3 Differentiable Renderer — Agent Reference

A precision reference for implementing differentiable rendering pipelines with Mitsuba 3.
Covers installation, scene construction, materials, cameras, lighting, mesh handling,
differentiable rendering, PyTorch integration, optimization loops, and migration from PyRedner.

All code is tested against Mitsuba 3 (pip package, current stable release).

---

## Table of Contents

1. [Installation](#1-installation)
2. [Variants](#2-variants)
3. [Scene Construction](#3-scene-construction)
4. [Materials (BSDFs)](#4-materials-bsdfs)
5. [Cameras (Sensors)](#5-cameras-sensors)
6. [Lighting (Emitters)](#6-lighting-emitters)
7. [Mesh Handling](#7-mesh-handling)
8. [Differentiable Rendering](#8-differentiable-rendering)
9. [PyTorch Integration](#9-pytorch-integration)
10. [Inverse Rendering Optimization Loop](#10-inverse-rendering-optimization-loop)
11. [Common Gotchas](#11-common-gotchas)
12. [PyRedner to Mitsuba 3 Migration](#12-pyredner-to-mitsuba-3-migration)
13. [NextFace-Specific Notes](#13-nextface-specific-notes)

---

## 1. Installation

```bash
pip install mitsuba
# or with uv:
uv add mitsuba
```

DrJit ships with Mitsuba and does not need a separate install.

### Verify Installation

```python
import mitsuba as mi
import drjit as dr

print(mi.variants())
# ['scalar_rgb', 'scalar_spectral', 'llvm_ad_rgb', 'cuda_ad_rgb']
```

---

## 2. Variants

### Naming Convention: `{backend}_{mode}_{spectrum}`

| Component | Options | Description |
|-----------|---------|-------------|
| Backend | `scalar`, `llvm`, `cuda` | CPU single-ray, CPU JIT (LLVM), GPU JIT (CUDA) |
| Mode | (none), `ad` | No AD, or automatic differentiation enabled |
| Spectrum | `rgb`, `spectral`, `mono` | Color representation |

### Default pip variants

| Variant | Backend | Differentiable | Device | Use Case |
|---------|---------|----------------|--------|----------|
| `scalar_rgb` | CPU | No | CPU | Fast preview, debugging |
| `scalar_spectral` | CPU | No | CPU | Spectral preview |
| `llvm_ad_rgb` | LLVM | Yes | CPU | CPU inverse rendering |
| `cuda_ad_rgb` | CUDA | Yes | GPU | GPU inverse rendering (fast) |

### Setting a Variant

**CRITICAL: Set the variant ONCE before any other Mitsuba call. Scenes loaded in one
variant cannot be used in another.**

```python
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')  # or 'llvm_ad_rgb' for CPU
```

### For this project

```python
def ensure_variant():
    """Select best available differentiable variant."""
    if 'cuda_ad_rgb' in mi.variants():
        try:
            mi.set_variant('cuda_ad_rgb')
            return 'cuda_ad_rgb'
        except Exception:
            pass
    mi.set_variant('llvm_ad_rgb')
    return 'llvm_ad_rgb'
```

---

## 3. Scene Construction

### mi.load_dict() — The Core API

Scenes are built from nested Python dicts. Every dict requires a `"type"` key.

```python
scene = mi.load_dict({
    'type': 'scene',
    'integrator': {'type': 'path', 'max_depth': 6},
    'sensor': {
        'type': 'perspective',
        'fov': 45.0,
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0, 0, 5], target=[0, 0, 0], up=[0, 1, 0]
        ),
        'film': {
            'type': 'hdrfilm',
            'width': 512, 'height': 512,
            'rfilter': {'type': 'gaussian'},
        },
        'sampler': {'type': 'independent', 'sample_count': 64},
    },
    'light': {
        'type': 'point',
        'position': [0, 5, 5],
        'intensity': {'type': 'rgb', 'value': [100.0, 100.0, 100.0]},
    },
    'object': {
        'type': 'sphere',
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {'type': 'rgb', 'value': [0.8, 0.2, 0.2]},
        },
    },
})
```

### Rendering

```python
image = mi.render(scene, spp=128)  # mi.TensorXf of shape (H, W, C)
mi.util.write_bitmap("render.png", image)
```

### Reusing Objects

```python
shared_bsdf = mi.load_dict({
    'type': 'diffuse',
    'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]},
})
scene = mi.load_dict({
    'type': 'scene',
    'sphere': {'type': 'sphere', 'bsdf': shared_bsdf},
})
```

### Integrator Types

| Type | Description |
|------|-------------|
| `path` | Unidirectional path tracer (general purpose) |
| `direct` | Direct illumination only (fast, no indirect) |
| `prb` | Path Replay Backpropagation (recommended for diff. rendering, static geometry) |
| `prb_reparam` | PRB with reparameterization (handles silhouette/geometry gradients) |

```python
# For differentiable rendering, prefer prb:
'integrator': {'type': 'prb', 'max_depth': 6}
```

---

## 4. Materials (BSDFs)

### 4.1 Principled BSDF (recommended for face rendering)

Closest match to PyRedner's Material(diffuse, specular, roughness):

```python
{
    'type': 'principled',
    'base_color': {'type': 'rgb', 'value': [0.8, 0.2, 0.2]},
    'metallic': 0.0,       # 0 = dielectric (skin)
    'roughness': 0.4,      # microfacet roughness
    'specular': 0.5,       # Fresnel reflection coefficient
    'anisotropic': 0.0,
    'spec_trans': 0.0,     # 0 = opaque
}
```

With texture maps:

```python
{
    'type': 'principled',
    'base_color': {'type': 'bitmap', 'bitmap': mi.Bitmap(diffuse_np)},
    'roughness': {'type': 'bitmap', 'bitmap': mi.Bitmap(roughness_np)},
}
```

### 4.2 Roughplastic (alternative, explicit diffuse+specular separation)

```python
{
    'type': 'roughplastic',
    'distribution': 'ggx',
    'alpha': 0.3,          # roughness
    'int_ior': 1.5,
    'diffuse_reflectance': {'type': 'bitmap', 'bitmap': mi.Bitmap(diffuse_np)},
    'specular_reflectance': {'type': 'bitmap', 'bitmap': mi.Bitmap(specular_np)},
}
```

### 4.3 Diffuse (Lambertian, for albedo rendering)

```python
{'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}}
```

### 4.4 Roughconductor (metallic)

```python
{
    'type': 'roughconductor',
    'material': 'Al',
    'distribution': 'ggx',
    'alpha': 0.05,
}
```

### Creating Bitmap from tensor

```python
import numpy as np

# From numpy array [H, W, 3] float32
bitmap = mi.Bitmap(numpy_array)

# From torch tensor
numpy_data = tensor.detach().cpu().numpy().astype(np.float32)
bitmap = mi.Bitmap(numpy_data)

# Single channel [H, W, 1] for roughness
roughness_bitmap = mi.Bitmap(roughness_tensor.detach().cpu().numpy().astype(np.float32))
```

---

## 5. Cameras (Sensors)

### Perspective camera matching NextFace convention

```python
import math

def build_camera(focal, width, height, clip_near=10.0, spp=8):
    """Build Mitsuba perspective sensor dict matching PyRedner camera.

    Args:
        focal: focal length in pixels
        width: image width in pixels
        height: image height in pixels
        clip_near: near clipping plane
        spp: samples per pixel
    """
    # FOV formula matching original NextFace:
    fov = 360.0 * math.atan(width / (2.0 * focal)) / math.pi

    return {
        'type': 'perspective',
        'fov': fov,
        'fov_axis': 'x',
        'near_clip': clip_near,
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0, 0, 0],
            target=[0, 0, 1],
            up=[0, -1, 0],        # Y-down to match NextFace
        ),
        'film': {
            'type': 'hdrfilm',
            'width': width,
            'height': height,
            'pixel_format': 'rgba',
            'component_format': 'float32',
            'rfilter': {'type': 'tent'},
        },
        'sampler': {
            'type': 'independent',
            'sample_count': spp,
        },
    }
```

### ScalarTransform4f Operations

```python
# look_at
T = mi.ScalarTransform4f.look_at(origin=[0,0,0], target=[0,0,1], up=[0,-1,0])

# Translate, Rotate (axis + degrees), Scale
T = mi.ScalarTransform4f.translate([1, 0, 0])
T = mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=45.0)
T = mi.ScalarTransform4f.scale([2, 2, 2])

# Chaining (right-to-left mathematically)
T = mi.ScalarTransform4f.translate([1,0,0]) @ mi.ScalarTransform4f.rotate([0,1,0], 45)

# Fluent syntax (left-to-right, more readable)
T = mi.ScalarTransform4f().rotate([0,1,0], 45.0).translate([1,0,0])
```

### FOV from focal length

```python
# For pixel-based intrinsics:
def intrinsics_to_fov(fx_pixels, width):
    return 2.0 * math.degrees(math.atan(width / (2.0 * fx_pixels)))

# NextFace formula (equivalent):
# fov = 360 * atan(width / (2*focal)) / pi
```

---

## 6. Lighting (Emitters)

### Environment Map from tensor (key pattern for NextFace)

```python
def build_envmap(envmap_tensor):
    """Build Mitsuba envmap emitter from [H, W, 3] torch tensor."""
    numpy_data = envmap_tensor.detach().cpu().numpy().astype(np.float32)
    bitmap = mi.Bitmap(numpy_data)
    return {
        'type': 'envmap',
        'bitmap': bitmap,
    }
```

### Updating envmap during optimization

```python
params = mi.traverse(scene)
# Key is typically 'environment.data' — check with print(params)
new_env = mi.TensorXf(updated_np_array)   # Shape (H, W, 3)
params['environment.data'] = new_env
params.update()
```

### Constant background (for testing)

```python
{'type': 'constant', 'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}}
```

### Point light

```python
{'type': 'point', 'position': [0, 5, 5],
 'intensity': {'type': 'rgb', 'value': [100.0, 100.0, 100.0]}}
```

---

## 7. Mesh Handling

### Creating mesh from vertex/face buffers

Buffers are FLAT (not shaped): `[V*3]` for positions, `[F*3]` for faces.
Use `dr.ravel(mi.Point3f(...))` pattern.

```python
import drjit as dr
import numpy as np

def build_mesh(vertices_np, faces_np, normals_np=None, uvs_np=None):
    """Build Mitsuba mesh from numpy arrays.

    Args:
        vertices_np: [V, 3] float32
        faces_np: [F, 3] uint32
        normals_np: [V, 3] float32 (optional)
        uvs_np: [V, 2] float32 (optional)
    """
    mesh = mi.Mesh(
        "face_mesh",
        vertex_count=vertices_np.shape[0],
        face_count=faces_np.shape[0],
        has_vertex_normals=normals_np is not None,
        has_vertex_texcoords=uvs_np is not None,
    )

    mesh_params = mi.traverse(mesh)

    # Vertex positions — dr.ravel flattens Point3f to [V*3]
    mesh_params['vertex_positions'] = dr.ravel(
        mi.Point3f(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2])
    )

    # Face indices — dr.ravel flattens Vector3u to [F*3]
    mesh_params['faces'] = dr.ravel(
        mi.Vector3u(faces_np[:, 0], faces_np[:, 1], faces_np[:, 2])
    )

    if normals_np is not None:
        mesh_params['vertex_normals'] = dr.ravel(
            mi.Normal3f(normals_np[:, 0], normals_np[:, 1], normals_np[:, 2])
        )

    if uvs_np is not None:
        mesh_params['vertex_texcoords'] = dr.ravel(
            mi.Point2f(uvs_np[:, 0], uvs_np[:, 1])
        )

    mesh_params.update()
    return mesh
```

### Using mesh in scene

```python
scene = mi.load_dict({
    'type': 'scene',
    'my_mesh': mesh,          # Pass mi.Mesh object directly
    # ... sensors, emitters
})
```

### Reading back mesh geometry

```python
params = mi.traverse(scene)
flat_verts = params['my_mesh.vertex_positions']
verts = dr.unravel(mi.Point3f, flat_verts)
# verts.x, verts.y, verts.z are dr.Float arrays
```

### Critical notes

- Face winding: Mitsuba uses CCW. If faces appear inside-out, reverse winding
- Must call `mesh_params.update()` after setting all buffers
- For torch tensors, convert to numpy first: `t.detach().cpu().numpy().astype(np.float32)`

---

## 8. Differentiable Rendering

### Traversing scene parameters

```python
params = mi.traverse(scene)
print(params)  # Shows all modifiable parameter keys

# Common keys:
# 'shape.vertex_positions'      — [V*3] flat vertex buffer
# 'shape.vertex_normals'        — [V*3] flat normal buffer
# 'shape.bsdf.base_color.data'  — texture bitmap data
# 'shape.bsdf.roughness.data'   — roughness texture data
# 'environment.data'            — environment map data
```

### Enabling gradients

**Forgetting `dr.enable_grad()` is the most common cause of silent failures.**

```python
key = 'shape.vertex_positions'
dr.enable_grad(params[key])
params.update()  # Must propagate to scene graph
```

### Render with gradient tracking

```python
# params arg MUST be passed for gradients to flow
image = mi.render(scene, params=params, spp=64, seed=0, seed_grad=1)
```

### Backward pass

```python
loss = dr.mean(dr.sqr(image - target_image))
dr.backward(loss)
grad = dr.grad(params[key])
```

### Parameter update cycle

```python
# 1. Enable grad
dr.enable_grad(params[key])

# 2. Render
image = mi.render(scene, params=params, spp=32, seed=i, seed_grad=i+1)

# 3. Loss + backward
loss = dr.mean(dr.sqr(image - target))
dr.backward(loss)

# 4. Read grad, apply update
grad = dr.grad(params[key])
params[key] = mi.Float(params[key] - 0.01 * grad)

# 5. Update scene
params.update()
```

---

## 9. PyTorch Integration

### 9.1 Tensor Conversion

```python
# mi.TensorXf -> torch.Tensor (zero-copy on GPU with cuda variant)
torch_tensor = mitsuba_tensor.torch()

# torch.Tensor -> mi.TensorXf
mitsuba_tensor = mi.TensorXf(torch_tensor)

# mi.Float -> numpy
numpy_arr = mitsuba_float.numpy()

# numpy -> mi.Float
mitsuba_float = mi.Float(numpy_arr)
```

### 9.2 Image shape: HWC not NCHW

```python
image = mi.render(scene, spp=64)  # (H, W, C) — Mitsuba format

# Mitsuba -> PyTorch NCHW
t = image.torch()                                  # (H, W, 3)
t_nchw = t.permute(2, 0, 1).unsqueeze(0)           # (1, 3, H, W)

# PyTorch NCHW -> Mitsuba
t_hwc = t_nchw.squeeze(0).permute(1, 2, 0)         # (H, W, 3)
mi_tensor = mi.TensorXf(t_hwc.contiguous())
```

### 9.3 dr.wrap() — The Gradient Bridge (CRITICAL for NextFace)

`dr.wrap()` bridges PyTorch autograd and DrJit AD. This is how torch.optim.Adam
can optimize parameters that flow through Mitsuba rendering.

```python
@dr.wrap(source='torch', target='drjit')
def render_torch(vertex_positions, texture_data):
    """
    Input: PyTorch tensors (with requires_grad=True)
    Output: PyTorch tensor (gradients flow back through DrJit)
    """
    # Inside here, inputs are DrJit arrays
    params['shape.vertex_positions'] = vertex_positions
    params['shape.bsdf.base_color.data'] = texture_data
    params.update()
    image = mi.render(scene, params=params, spp=32)
    return image

# Usage in PyTorch optimization loop:
vertices = torch.randn(V * 3, requires_grad=True, device='cuda')
texture = torch.rand(H * W * 3, requires_grad=True, device='cuda')
optimizer = torch.optim.Adam([vertices, texture], lr=0.01)

for i in range(100):
    optimizer.zero_grad()
    image = render_torch(vertices, texture)  # returns torch tensor
    loss = torch.mean((image - target) ** 2)
    loss.backward()  # gradients flow through DrJit back to PyTorch
    optimizer.step()
```

**`dr.wrap()` notes:**
- `source='torch'` means the CALLER provides PyTorch tensors
- `target='drjit'` means the FUNCTION BODY uses DrJit operations
- Input tensors auto-converted: torch.Tensor -> mi.Float
- Output auto-converted: mi.TensorXf -> torch.Tensor
- Gradients flow correctly through boundary in both directions
- Tensors must be 1D (flat) — reshape before/after

### 9.4 Custom torch.autograd.Function (fallback)

```python
class MitsubaRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices_flat, scene, params, key, spp):
        ctx.scene = scene
        ctx.params = params
        ctx.key = key
        ctx.spp = spp

        dr_verts = mi.Float(vertices_flat.detach().cpu().numpy())
        dr.enable_grad(dr_verts)
        params[key] = dr_verts
        params.update()

        image = mi.render(scene, params=params, spp=spp)
        ctx.dr_verts = dr_verts
        ctx.image = image

        return torch.tensor(np.array(image), device=vertices_flat.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_np = grad_output.detach().cpu().numpy()
        dr.set_grad(ctx.image, mi.TensorXf(grad_np))
        dr.enqueue(dr.ADMode.Backward, ctx.image)
        dr.traverse(mi.Float, dr.ADMode.Backward)

        grad_verts = torch.tensor(
            np.array(dr.grad(ctx.dr_verts)),
            device=grad_output.device
        )
        return grad_verts, None, None, None, None
```

---

## 10. Inverse Rendering Optimization Loop

### Pure DrJit loop (no PyTorch)

```python
mi.set_variant('cuda_ad_rgb')

scene = mi.load_dict({...})
params = mi.traverse(scene)
key = 'sphere.bsdf.reflectance.value'

# Perturb initial value
params[key] = mi.Color3f(0.01, 0.2, 0.9)
params.update()

# DrJit Adam optimizer
opt = mi.ad.Adam(lr=0.05, params={key: params[key]})

for step in range(100):
    dr.enable_grad(opt[key])
    params[key] = opt[key]
    params.update()

    image = mi.render(scene, params=params, spp=64, seed=step, seed_grad=step+1)
    loss = dr.mean(dr.sqr(image - target_image))
    dr.backward(loss)

    opt.step()
    opt[key] = dr.clamp(opt[key], 0.0, 1.0)  # clamp to valid range

    print(f"Step {step}: loss={loss[0]:.4f}")
```

### Multi-view optimization

```python
sensors = [
    mi.load_dict({
        'type': 'perspective',
        'fov': 45.0,
        'to_world': mi.ScalarTransform4f.look_at(pos, [0,0,0], [0,1,0]),
        'film': {'type': 'hdrfilm', 'width': 256, 'height': 256},
        'sampler': {'type': 'independent', 'sample_count': 16},
    })
    for pos in [[5,0,0], [-5,0,0], [0,5,0], [0,-5,0]]
]

for step in range(100):
    dr.enable_grad(opt[key])
    params[key] = opt[key]
    params.update()

    total_loss = mi.Float(0.0)
    for i, sensor in enumerate(sensors):
        image = mi.render(scene, params=params, sensor=sensor, spp=16, seed=step)
        total_loss += dr.mean(dr.sqr(image - target_images[i]))

    dr.backward(total_loss / len(sensors))
    opt.step()
```

### Environment map optimization

```python
params = mi.traverse(scene)
env_key = 'environment.data'  # verify with print(params)

opt = mi.ad.Adam(lr=0.01, params={env_key: params[env_key]})

for step in range(200):
    dr.enable_grad(opt[env_key])
    params[env_key] = opt[env_key]
    params.update()

    image = mi.render(scene, params=params, spp=32, seed=step)
    loss = dr.mean(dr.sqr(image - target))
    dr.backward(loss)

    opt.step()
    opt[env_key] = dr.clamp(opt[env_key], 0.0, 1e6)  # keep non-negative
```

---

## 11. Common Gotchas

### 1. Variant not set
```python
# WRONG:
scene = mi.load_dict({...})  # ERROR!

# CORRECT:
mi.set_variant('cuda_ad_rgb')
scene = mi.load_dict({...})
```

### 2. Forgetting dr.enable_grad()
```python
# WRONG: gradients silently won't flow
image = mi.render(scene, params=params, spp=64)
dr.backward(dr.mean(image))
grad = dr.grad(params[key])  # Will be zero!

# CORRECT:
dr.enable_grad(params[key])
params.update()
image = mi.render(scene, params=params, spp=64)
dr.backward(dr.mean(image))
grad = dr.grad(params[key])  # Has valid gradients
```

### 3. Forgetting params.update()
```python
# WRONG:
params[key] = new_value
image = mi.render(scene, params=params)  # Uses OLD value!

# CORRECT:
params[key] = new_value
params.update()
image = mi.render(scene, params=params)
```

### 4. Forgetting params argument in mi.render()
```python
# WRONG: renders but gradients won't propagate
image = mi.render(scene, spp=64)

# CORRECT:
image = mi.render(scene, params=params, spp=64)
```

### 5. Tensor format: HWC not NCHW
```python
image = mi.render(scene)  # (H, W, C) — NOT (N, C, H, W)

# To match NextFace [N, H, W, 4]:
image_torch = image.torch().unsqueeze(0)  # add batch dim
```

### 6. Flat buffers for mesh params
```python
# WRONG:
params['vertex_positions'] = mi.Float(vertices_3d)  # [V, 3] fails!

# CORRECT:
params['vertex_positions'] = dr.ravel(
    mi.Point3f(v[:, 0], v[:, 1], v[:, 2])
)  # [V*3] flat
```

### 7. Scalar variant for differentiable rendering
```python
# WRONG: scalar_rgb does NOT support AD
mi.set_variant('scalar_rgb')
# dr.enable_grad() and dr.backward() will fail

# CORRECT:
mi.set_variant('llvm_ad_rgb')  # CPU
mi.set_variant('cuda_ad_rgb')  # GPU
```

### 8. Seed reuse in differentiable rendering
```python
# WRONG: correlated noise
image = mi.render(scene, params=params, spp=64, seed=0)

# CORRECT: different seeds for decorrelated gradient estimation
image = mi.render(scene, params=params, spp=64, seed=step, seed_grad=step+1)
```

### 9. Alpha channel
```python
# Film must request RGBA:
'film': {'type': 'hdrfilm', 'pixel_format': 'rgba'}
# Otherwise mi.render() returns [H, W, 3] without alpha
```

### 10. Image values are linear (no gamma)
```python
# Mitsuba renders in linear HDR. For sRGB targets, apply gamma:
image_srgb = dr.clamp(image, 0.0, 1.0) ** (1.0 / 2.2)
```

### 11. Scene incompatibility across variants
```python
# WRONG: load in one variant, render in another
mi.set_variant('scalar_rgb')
scene = mi.load_dict(...)
mi.set_variant('llvm_ad_rgb')
mi.render(scene)  # Error!

# CORRECT: set variant once, never change
```

---

## 12. PyRedner to Mitsuba 3 Migration

### API Mapping Table

| PyRedner | Mitsuba 3 | Notes |
|----------|-----------|-------|
| `pyredner.Camera(position, look_at, up, fov, resolution)` | `{'type': 'perspective', 'fov': fov, 'to_world': mi.ScalarTransform4f.look_at(...)}` | Dict-based |
| `pyredner.Material(diffuse, specular, roughness)` | `{'type': 'principled', 'base_color': ..., 'roughness': ...}` | Combined BSDF |
| `pyredner.Texture(tensor)` | `{'type': 'bitmap', 'bitmap': mi.Bitmap(numpy)}` | Bitmap wrapper |
| `pyredner.Object(vertices, indices, material, uvs, normals)` | `mi.Mesh(...)` + `dr.ravel()` + `params.update()` | Programmatic mesh |
| `pyredner.Scene(camera, materials, objects, envmap)` | `mi.load_dict({...})` | Dict assembly |
| `pyredner.EnvironmentMap(tensor)` | `{'type': 'envmap', 'bitmap': mi.Bitmap(numpy)}` | Envmap emitter |
| `pyredner.RenderFunction.apply(seed, *scene_args)` | `mi.render(scene, params=params, spp=N, seed=S)` | Core render |
| `pyredner.render_albedo(scenes)` | `path` integrator with `max_depth=0` or AOV | Different approach |
| `set_print_timing(False)` | `mi.set_log_level(mi.LogLevel.Warn)` | Silence logs |
| `pyredner.get_device()` | Variant determines device | `cuda_ad_rgb` vs `llvm_ad_rgb` |
| `pyredner.channels.alpha` | `'pixel_format': 'rgba'` in film | Film config |
| `pyredner.channels.diffuse_reflectance` | AOV integrator `'aovs': 'albedo:albedo'` | AOV pass |
| `pyredner.imwrite(image, path, gamma)` | `cv2.imwrite` with manual gamma | See below |
| `requires_grad_(True)` | `dr.enable_grad(params[key])` | Explicit AD enable |
| `loss.backward()` | `dr.backward(loss)` | DrJit backward |
| `tensor.grad` | `dr.grad(params[key])` | Read gradient |
| `torch.optim.Adam` | `mi.ad.Adam` or `torch.optim` via `dr.wrap()` | Both work |

### Key Behavioral Differences

| Behavior | PyRedner | Mitsuba 3 |
|----------|----------|-----------|
| Image format | (H, W, C) torch float | (H, W, C) mi.TensorXf |
| Gradient flow | PyTorch autograd, always on | Explicit `dr.enable_grad()` required |
| Silhouette gradients | Built-in via edge sampling | Requires `prb_reparam` integrator |
| Scene rebuild | Not needed for param changes | `params.update()` after changes |
| Environment map | Lat-long tensor | Same lat-long, via mi.Bitmap |
| Coordinate system | Right-handed, Y-up | Right-handed, Y-up |
| GPU memory | PyTorch CUDA tensors | DrJit CUDA arrays (separate allocator) |

### Migration Checklist

- [ ] Replace `pyredner.Camera()` with perspective sensor dict + `look_at`
- [ ] Replace `pyredner.Object()` with `mi.Mesh()` + traverse + `dr.ravel`
- [ ] Replace `pyredner.Material()` with principled/roughplastic BSDF dict
- [ ] Replace `pyredner.EnvironmentMap()` with envmap emitter dict
- [ ] Replace `requires_grad_(True)` with `dr.enable_grad(params[key])`
- [ ] Replace `loss.backward()` with `dr.backward(loss)` (or use `dr.wrap()`)
- [ ] Replace `tensor.grad` with `dr.grad(params[key])`
- [ ] Add `params.update()` after every parameter modification
- [ ] Add `seed_grad` argument to `mi.render()` for better gradient estimates
- [ ] Switch integrator to `prb` (or `prb_reparam` for geometry optimization)
- [ ] Verify image tensor shape: Mitsuba returns (H, W, C), not (N, C, H, W)
- [ ] Add gamma correction if comparing against sRGB reference images

---

## 13. NextFace-Specific Notes

### Coordinate system
- NextFace: Camera at origin (0,0,0) looking at +Z, up=(0,-1,0) (Y-down)
- Mitsuba default: Looks along -Z. Use `look_at(origin=[0,0,0], target=[0,0,1], up=[0,-1,0])`
- UV flip: `v = 1 - v` applied in pipeline.py line 34. Verify Mitsuba UV origin matches.

### Batch rendering
- NextFace builds scenes per-frame in a loop, renders each, then `torch.stack()`s
- Mitsuba: same pattern — loop over frames, build scene dict per frame, render, stack
- For N frames: N separate `mi.load_dict()` + `mi.render()` calls

### Rendering modes
- Forward render: `path` integrator with `max_depth` matching `self.bounces`
- Albedo render: `path` with `max_depth=0` or AOV integrator with `'aovs': 'albedo:albedo'`

### Alpha channel
- NextFace uses RGBA [N, H, W, 4], alpha used for masking in loss: `mask = images[..., 3:]`
- Mitsuba: set `'pixel_format': 'rgba'` in film to get alpha channel

### File I/O replacement
```python
import cv2
import numpy as np

def save_image(image_tensor, filename, gamma=2.2):
    """Replace pyredner.imwrite() — save with gamma correction."""
    img = image_tensor.detach().cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    img_rgb = np.power(img[..., :3], 1.0 / gamma)
    img_bgr = (img_rgb[..., ::-1] * 255).astype(np.uint8)
    cv2.imwrite(filename, img_bgr)
```

### Suppressing logs
```python
mi.set_log_level(mi.LogLevel.Warn)  # replaces set_print_timing(False)
```

---

## References

- [Mitsuba 3 Documentation](https://mitsuba.readthedocs.io/en/stable/)
- [Gradient-based Optimization](https://mitsuba.readthedocs.io/en/v3.5.1/src/inverse_rendering/gradient_based_opt.html)
- [PyTorch Interoperability](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/pytorch_mitsuba_interoperability.html)
- [Mesh I/O and Manipulation](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/mesh_io_and_manipulation.html)
- [Scene Format](https://mitsuba.readthedocs.io/en/latest/src/key_topics/scene_format.html)
- [Choosing Variants](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html)
- [BSDF Plugins](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html)
- [Emitter Plugins](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html)
- [Transformation Toolbox](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html)
- [Mitsuba 3 GitHub](https://github.com/mitsuba-renderer/mitsuba3)
