"""DrJit-PyTorch gradient bridge for differentiable Mitsuba 3 rendering.

Bridges DrJit AD (used by Mitsuba 3) with PyTorch autograd, enabling
gradient flow from rendered images back to scene parameters for
optimization with torch.optim.

Uses a custom torch.autograd.Function that:
  - Forward: sets scene params from torch tensors, renders via Mitsuba
  - Backward: re-renders with DrJit AD enabled, propagates gradients
    back to the original torch tensors
"""

import torch
import mitsuba as mi
import drjit as dr
import numpy as np


def _set_scene_param(params, path, torch_val):
    """Set a scene parameter from a detached torch tensor."""
    np_val = torch_val.detach().cpu().numpy().astype(np.float32)
    original = params[path]

    if isinstance(original, mi.Color3f):
        flat = np_val.flatten()
        params[path] = mi.Color3f(float(flat[0]), float(flat[1]), float(flat[2]))
    elif isinstance(original, mi.TensorXf):
        # Preserve shape for texture data (e.g., [H, W, C])
        params[path] = mi.TensorXf(np_val.reshape(original.shape))
    else:
        params[path] = type(original)(np_val.ravel())


def _torch_to_drjit(params, path, torch_val):
    """Convert a detached torch tensor to an AD-capable DrJit value."""
    np_val = torch_val.detach().cpu().numpy().astype(np.float32)
    original = params[path]

    if isinstance(original, mi.Color3f):
        flat = np_val.flatten()
        return mi.Color3f(float(flat[0]), float(flat[1]), float(flat[2]))
    elif isinstance(original, mi.TensorXf):
        return mi.TensorXf(np_val.reshape(original.shape))
    else:
        return type(original)(np_val.ravel())


def _drjit_grad_to_numpy(dr_val, shape):
    """Extract gradient from a DrJit value and reshape to match torch tensor."""
    g = dr.grad(dr_val)
    if isinstance(g, mi.TensorXf):
        # TensorXf has its own array conversion
        return np.array(g).reshape(shape).astype(np.float32)
    # Use dr.ravel for structured types (Color3f, Vector3f, etc.)
    try:
        flat = dr.ravel(g)
    except Exception:
        flat = g
    return np.array(flat).reshape(shape).astype(np.float32)


class _DiffRender(torch.autograd.Function):
    """Custom autograd function bridging Mitsuba differentiable rendering."""

    @staticmethod
    def forward(ctx, scene, spp, param_paths, *torch_values):
        ctx.scene = scene
        ctx.spp = spp
        ctx.param_paths = param_paths
        ctx.shapes = [v.shape for v in torch_values]
        ctx.save_for_backward(*torch_values)

        # Update scene params and render (no AD needed for forward)
        params = mi.traverse(scene)
        for path, val in zip(param_paths, torch_values):
            _set_scene_param(params, path, val)
        params.update()

        img = mi.render(scene, spp=spp)
        return torch.from_numpy(np.array(img).copy()).to(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        scene = ctx.scene
        spp = ctx.spp
        param_paths = ctx.param_paths
        shapes = ctx.shapes
        saved = ctx.saved_tensors

        # Re-render with DrJit AD to compute gradients.
        # Create AD-tracked DrJit values, set them in params, then update
        # once. This ensures TensorXf (texture) params are properly
        # connected to the AD graph alongside Float params.
        params = mi.traverse(scene)

        # Create AD-tracked DrJit values and set in params
        ad_values = {}
        for path, val in zip(param_paths, saved):
            drjit_val = _torch_to_drjit(params, path, val)
            dr.enable_grad(drjit_val)
            params[path] = drjit_val
            ad_values[path] = drjit_val
        params.update()

        # Render with AD tracking
        img = mi.render(scene, params=params, spp=spp)

        # Seed output gradient and backpropagate through DrJit AD graph
        grad_np = grad_output.detach().cpu().numpy().astype(np.float32)
        dr.set_grad(img, mi.TensorXf(grad_np))
        dr.enqueue(dr.ADMode.Backward, img)
        dr.traverse(dr.ADMode.Backward)

        # Collect parameter gradients from our AD-tracked values
        grads = [None, None, None]  # scene, spp, param_paths
        for path, shape in zip(param_paths, shapes):
            g_np = _drjit_grad_to_numpy(ad_values[path], shape)
            grads.append(torch.from_numpy(g_np))

        return tuple(grads)


def differentiable_render(scene, torch_params, spp=8):
    """Render a Mitsuba scene with gradient flow to PyTorch tensors.

    Args:
        scene: mi.Scene -- loaded Mitsuba scene.
        torch_params: dict[str, torch.Tensor] -- mapping of scene parameter
            paths (from mi.traverse()) to PyTorch tensors.
            Use requires_grad=True for parameters that need gradients.
        spp: int -- samples per pixel.

    Returns:
        torch.Tensor -- rendered image [H, W, C] with autograd support.
            Calling .backward() on a loss derived from this image will
            populate .grad on the input torch tensors.
    """
    param_paths = tuple(torch_params.keys())
    torch_values = tuple(torch_params[p] for p in param_paths)
    return _DiffRender.apply(scene, spp, param_paths, *torch_values)
