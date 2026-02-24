# Mitsuba 3 Porting Notes

Lessons learned from porting NextFace's rendering backend from PyRedner to Mitsuba 3.

---

## Scene Assembly

- Attach BSDF to mesh via `mesh.set_bsdf(mi.load_dict(material_dict))` before adding to scene dict
- `mi.Mesh` objects can be placed directly in scene dict values (no wrapper needed)
- Envmap emitter dict goes directly in scene dict (not nested under a shape)
- Scene assembly pattern: build component dicts/objects separately, combine in scene dict, call `mi.load_dict()`
- Path integrator uses `max_depth` (not `max_bounces`) -- set to `bounces + 1`

## Camera

- Camera dict keys: `type`, `fov`, `near_clip` (not `clip_near`), `to_world`, `film`
- FOV formula: `360 * atan(width / (2 * focal)) / pi` -- must match original exactly
- Up vector is `(0, -1, 0)` (Y-down convention)
- `mi.ScalarTransform4f.look_at()` for camera `to_world` transform
- Default `clip_near=10.0` -- all geometry must be at z > 10

## Mesh

- Create with `mi.Mesh(name, vertex_count, face_count, has_vertex_normals, has_vertex_texcoords)`
- Set buffers via `mi.traverse(mesh)`: positions/normals flattened [V,3] -> [V*3], faces [F,3] -> [F*3], UVs [V,2] -> [V*2]
- Face indices must be `mi.UInt32` (not signed int)
- UV flip (`v = 1 - v`) is done by the caller, not the mesh builder
- Face winding must give normals facing the camera for gradient flow (wrong winding renders OK but AD gradients are zero)

## Materials

- Principled BSDF: `base_color` and `roughness` accept bitmap textures, `specular` is float-only
- `roughplastic` BSDF `alpha` is also float-only in dict loading
- For single-channel [H,W,1] tensors, squeeze to [H,W] before creating `mi.Bitmap`
- `mi.Bitmap(numpy_array)` accepts [H,W,3] float32 arrays directly

## Environment Maps

- Envmap dict: `{"type": "envmap", "bitmap": mi.Bitmap(...)}`
- Mitsuba internally adds a wrap-around column: [H,W,3] -> [H,W+1,3] in scene params. Pad with first column when bridging gradients.
- With envmap present, alpha = 1 everywhere (envmap acts as background surface)
- Without envmap, alpha correctly represents mesh coverage

## Rendering

- `mi.render(scene, spp=N)` returns `mi.TensorXf` [H,W,C] -- convert via `np.array()` then `torch.from_numpy()`
- hdrfilm needs `pixel_format: rgba` to include alpha channel (default is RGB only)
- All rendered images are [N,H,W,4] (RGBA) -- alpha channel used for masking in loss computation

## Albedo Rendering (AOV)

- AOV integrator outputs [H,W,7]: `[path_R, path_G, path_B, alpha, albedo_R, albedo_G, albedo_B]`
- AOV `albedo` type extracts diffuse reflectance
- Exclude envmap from scene so alpha correctly represents mesh coverage
- AOV channels are zero when geometry is clipped -- easy to misdiagnose as "AOV broken"

## Differentiable Rendering (Gradient Bridge)

- Use custom `torch.autograd.Function`, NOT `dr.wrap_ad` (deprecated)
- Forward: convert torch tensors to DrJit params, set scene params, render via `mi.render()`
- Backward: re-render with DrJit AD enabled, seed output gradient, backpropagate through DrJit graph, extract gradients (double-render is standard)
- `dr.traverse(dr.ADMode.Backward)` -- no type arg
- `dr.ravel()` converts structured DrJit types (Color3f) to flat Float arrays for gradient extraction
- `Color3f` is a structured type (struct-of-3-arrays), construct via `mi.Color3f(r, g, b)` from Python floats
- TensorXf (bitmap data) must preserve shape: `mi.TensorXf(np_val.reshape(original.shape))` -- flat 1D triggers dimension errors
- In backward pass, create AD-tracked DrJit values BEFORE setting in params, then extract grads from those tracked references
- Scene parameter paths: `shape.vertex_positions`, `shape.bsdf.base_color.data`, `shape.bsdf.roughness.data`, `envmap.data`

## Variant Selection

- Priority order: `cuda_ad_rgb` > `llvm_ad_rgb` > `scalar_rgb`
- `llvm_ad_rgb` can be listed in `mi.variants()` but fail at `mi.set_variant()` if libLLVM.dylib is missing
- DrJIT requires libLLVM.dylib even for `scalar_rgb` -- on macOS: `brew install llvm`, then set `DRJIT_LIBLLVM_PATH=/opt/homebrew/opt/llvm/lib/libLLVM.dylib`
- `tests/conftest.py` auto-detects Homebrew LLVM so no manual env setup needed

## Testing Tips

- Test geometry must be at z > 10 (beyond `clip_near=10.0`) -- geometry at z=3 is invisible
- Use `sphere` (outward normals) not `rectangle` (single-sided, faces +Z) for test scenes visible from any direction
- Monte Carlo differentiable rendering has high variance -- `rtol=1.0` (same order of magnitude) is realistic for finite difference comparison
- Use larger epsilon (0.05) for finite differences to average out MC noise
- Higher spp (512+) needed for stable analytic gradients

## What Didn't Need Changing

- All 15 non-renderer files are pure PyTorch -- no changes needed
- Config parsing (.ini files) is pure Python
- SphericalHarmonics is pure PyTorch
- Checkpoint save/load is pure Python/pickle/numpy
- The only import swap needed was in `pipeline.py`: `from renderer import Renderer` -> `from renderer_mitsuba import Renderer`
