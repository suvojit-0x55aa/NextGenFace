# Porting Plan: NextFace PyRedner → Mitsuba 3

## Overview

NextFace is a 3D face reconstruction system that estimates face shape, albedo, and lighting from 2D photographs using differentiable rendering. The current implementation depends on PyRedner (Python 3.6, PyTorch 1.3) — both end-of-life. This plan ports the renderer to Mitsuba 3 with Python 3.10+ and UV package management.

## Scope

**Files requiring changes: 2 of 17**

| File | Change Type | Effort |
|------|------------|--------|
| `renderer.py` | Full rewrite (157 lines, 14 PyRedner API calls) | High |
| `image.py` | 1-line fix (replace `pyredner.imwrite` on line 19) | Trivial |

**Files unchanged: 15** — All pure PyTorch, no renderer dependency:
`camera.py`, `config.py`, `gaussiansmoothing.py`, `landmarksfan.py`, `landmarksmediapipe.py`, `meshnormals.py`, `morphablemodel.py`, `normalsampler.py`, `optimizer.py`, `pipeline.py`, `projection.py`, `replay.py`, `sphericalharmonics.py`, `textureloss.py`, `utils.py`

## Architecture

```
optimizer.py (unchanged)
  └── pipeline.py (unchanged)
        ├── camera.py (unchanged, pure PyTorch transforms)
        ├── morphablemodel.py (unchanged, PCA shape/albedo model)
        ├── sphericalharmonics.py (unchanged, SH → envmap)
        └── renderer.py (REWRITE)
              ├── build_camera() — FOV + perspective sensor dict
              ├── build_mesh() — vertices/faces → mi.Mesh
              ├── build_material() — textures → principled BSDF
              ├── build_envmap() — tensor → envmap emitter
              ├── build_scene() — assemble all components
              ├── render() — mi.render() + DrJit→PyTorch bridge
              └── render_albedo() — AOV/direct integrator mode
```

## Dependency Graph

```
US-001 (UV setup)
  └── US-002 (variant utility)
        └── US-003 (smoke test)
              ├── US-004 (camera) ──────────────────┐
              ├── US-005 (coordinates) ─────────────┤
              ├── US-006 (mesh) ────────────────────┤
              │     └── US-008 (UV mapping) ────────┤
              ├── US-007 (materials) ───────────────┤
              └── US-009 (envmap) ──────────────────┤
                    └── US-010 (SH integration) ────┤
                                                     ▼
                                              US-011 (scene assembly)
                                                │
                                         US-012 (forward render)
                                         US-013 (albedo render)
                                                │
                                         US-014 (gradient bridge) ◄── CRITICAL PATH
                                         US-015 (vertex gradients)
                                         US-016 (texture gradients)
                                                │
                                         US-017 (Renderer class API)
                                         US-018 (image.py fix)
                                                │
                                         US-019 (Step 1 verify)
                                         US-020 (Step 2 verify) ◄── FIRST FULL PIPELINE TEST
                                         US-021 (Step 3 verify)
                                                │
                                         US-022 (E2E single image)
                                         US-023 (E2E multi-image)
                                         US-024 (checkpoint roundtrip)
                                                │
                                         US-025 (remove pyredner)
                                         US-026 (60% test coverage)
                                         US-027 (final validation)
```

## Phases

### Phase 1: Foundation (US-001 → US-003)
Set up UV project, Mitsuba variant selection, smoke test. **Zero rendering code yet.**

### Phase 2: Component Building (US-004 → US-010)
Build each rendering component independently: camera, mesh, materials, envmap. Each can be unit tested in isolation.

### Phase 3: Assembly & Forward Path (US-011 → US-013)
Assemble components into full scenes. Verify forward rendering produces images.

### Phase 4: Differentiable Bridge (US-014 → US-016)
**Critical path.** Bridge DrJit's AD system with PyTorch autograd. Verify gradients for vertices and textures.

### Phase 5: API & Integration (US-017 → US-018)
Wrap everything in the Renderer class matching the original API. Fix image.py.

### Phase 6: Optimization Verification (US-019 → US-021)
Verify each optimization step converges with the new renderer.

### Phase 7: End-to-End & Cleanup (US-022 → US-027)
Full pipeline test, checkpoint roundtrip, pyredner removal, test coverage.

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| DrJit↔PyTorch gradient bridge complexity | HIGH | HIGH | US-014 is isolated early. Fallback: numerical gradients for prototyping |
| Coordinate system mismatch | MEDIUM | MEDIUM | US-005 catches this early with empirical tests |
| Mitsuba envmap orientation differs from PyRedner | MEDIUM | MEDIUM | US-009/010 test with known SH coefficients |
| Performance regression (Mitsuba slower than PyRedner) | LOW | MEDIUM | Acceptable if <2x. Mitsuba has better GPU utilization |
| UV mapping flipped/rotated | MEDIUM | LOW | US-008 checkerboard test catches this visually |
| Alpha channel semantics differ | LOW | LOW | US-012 explicitly tests alpha presence and values |

## Key Technical Decisions

1. **Mitsuba variant**: `cuda_ad_rgb` (GPU) or `llvm_ad_rgb` (CPU). Auto-selected at runtime.
2. **Material model**: Principled BSDF (closest to PyRedner's diffuse+specular+roughness).
3. **Gradient bridge**: Custom `torch.autograd.Function` (double-render approach). `dr.wrap_ad()` is deprecated.
4. **Scene construction**: `mi.load_dict()` for declarative scene building (matches Mitsuba idioms).
5. **Batch rendering**: Loop over frames (same as original), not Mitsuba's native batching.

## Testing Strategy

- **Integration tests** (primary): Verify components work together through the rendering pipeline
- **Unit tests** (~60% coverage): Focus on `renderer.py` and gradient bridge
- **Smoke tests**: GPU variant selection, import checks, minimal renders
- **Numerical sanity**: Monotonic brightness, coordinate correctness, energy conservation
- **Finite difference checks**: Gradient correctness for vertices and textures

## Completion Status

**PORT COMPLETE** — All 27 user stories implemented and passing as of 2026-02-24.

### New Files Created
| File | Purpose |
|------|---------|
| `NextFace/mitsuba_variant.py` | GPU/CPU variant auto-selection |
| `NextFace/camera_mitsuba.py` | Perspective camera builder |
| `NextFace/mesh_mitsuba.py` | Mesh construction from torch tensors |
| `NextFace/material_mitsuba.py` | Principled BSDF material builder |
| `NextFace/envmap_mitsuba.py` | Environment map emitter builder |
| `NextFace/scene_mitsuba.py` | Scene assembly (camera + mesh + material + envmap) |
| `NextFace/render_mitsuba.py` | Forward rendering (path tracing + albedo) |
| `NextFace/gradient_bridge.py` | DrJit-PyTorch AD bridge |
| `NextFace/renderer_mitsuba.py` | Drop-in Renderer class (same API as original) |

### Files Modified
| File | Change |
|------|--------|
| `NextFace/pipeline.py` | Import swap: `renderer` → `renderer_mitsuba` |
| `NextFace/image.py` | Replaced `pyredner.imwrite` with cv2-based implementation |

### Success Criteria

1. `grep -r 'pyredner\|import redner' NextFace/` returns zero results — **PASS**
2. All 27 user stories pass — **PASS**
3. E2E reconstruction produces visually reasonable output — **PASS**
4. Optimization converges (loss decreases) for all 3 steps — **PASS**
5. Test coverage >= 60% on renderer_mitsuba.py — **PASS** (94%)
