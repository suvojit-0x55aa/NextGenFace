"""Mitsuba 3 forward rendering utilities.

Replaces the original PyRedner renderPathTracing() by using mi.render()
to produce RGBA images as PyTorch tensors.
"""

import torch
import mitsuba as mi
import drjit as dr
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
