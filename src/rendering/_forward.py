"""Mitsuba 3 forward rendering utilities.

Replaces the original PyRedner renderPathTracing() by using mi.render()
to produce RGBA images as PyTorch tensors.
"""

import torch
import mitsuba as mi
import numpy as np


def render_scenes(scenes, spp=8, device=None):
    """Render a list of Mitsuba scenes and return RGBA images as a PyTorch tensor.

    Args:
        scenes: list[mi.Scene] — scenes to render (from build_scenes())
        spp: int — samples per pixel (overrides scene's sampler spp)
        device: torch.device or None — target device for output tensor.
                If None, defaults to CPU.

    Returns:
        torch.Tensor of shape [N, H, W, 4] (RGBA), float32, on `device`.
    """
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    images = []
    for scene in scenes:
        img = mi.render(scene, spp=spp)
        # img is a mi.TensorXf of shape [H, W, C]
        arr = np.array(img)

        # Ensure 4 channels (RGBA)
        if arr.ndim == 3 and arr.shape[2] == 3:
            # No alpha channel — add alpha = 1.0 for all pixels
            alpha = np.ones((*arr.shape[:2], 1), dtype=np.float32)
            arr = np.concatenate([arr, alpha], axis=2)
        elif arr.ndim == 3 and arr.shape[2] > 4:
            # Truncate to RGBA
            arr = arr[:, :, :4]

        t = torch.from_numpy(arr.copy()).to(device)
        images.append(t)

    return torch.stack(images, dim=0)


def render_albedo(scenes, spp=8, device=None):
    """Render albedo (diffuse reflectance) from AOV-configured Mitsuba scenes.

    Scenes must be built with albedo_mode=True (via build_scenes()) so that
    the AOV integrator outputs albedo channels alongside standard RGBA.

    The AOV output has shape [H, W, 7]: [R, G, B, A, albedo_R, albedo_G, albedo_B].
    This function extracts the albedo RGB (channels 4-6) and alpha (channel 3),
    returning [N, H, W, 4] RGBA where RGB = diffuse reflectance.

    Args:
        scenes: list[mi.Scene] — scenes built with albedo_mode=True.
        spp: int — samples per pixel.
        device: torch.device or None — target device for output tensor.

    Returns:
        torch.Tensor of shape [N, H, W, 4] (albedo RGBA), float32, on `device`.
    """
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    images = []
    for scene in scenes:
        img = mi.render(scene, spp=spp)
        arr = np.array(img)

        # AOV output: [H, W, 7] = [R, G, B, A, albedo_R, albedo_G, albedo_B]
        if arr.ndim == 3 and arr.shape[2] >= 7:
            albedo_rgb = arr[:, :, 4:7]
            alpha = arr[:, :, 3:4]
            rgba = np.concatenate([albedo_rgb, alpha], axis=2)
        elif arr.ndim == 3 and arr.shape[2] >= 4:
            # Fallback: no AOV channels found, use radiance as approximation
            rgba = arr[:, :, :4]
        else:
            # Minimal fallback
            h, w = arr.shape[:2]
            c = arr.shape[2] if arr.ndim == 3 else 1
            pad = np.zeros((h, w, 4 - c), dtype=np.float32)
            rgba = np.concatenate([arr.reshape(h, w, c), pad], axis=2)

        t = torch.from_numpy(rgba.copy()).to(device)
        images.append(t)

    return torch.stack(images, dim=0)
