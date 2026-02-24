"""Mitsuba 3 environment map emitter from PyTorch tensors.

Converts a PyTorch [H, W, 3] environment map tensor into a Mitsuba 3
envmap emitter dict suitable for scene assembly.
"""

import torch
import numpy as np
import mitsuba as mi


def build_envmap(envmap_tensor):
    """Build a Mitsuba envmap emitter dict from a PyTorch tensor.

    Args:
        envmap_tensor: [H, W, 3] float32 torch tensor representing
            the environment map radiance.

    Returns:
        dict: Mitsuba envmap emitter dict suitable for mi.load_dict()
            scene assembly.
    """
    if envmap_tensor.dim() != 3 or envmap_tensor.shape[2] != 3:
        raise ValueError(
            f"Expected [H, W, 3] tensor, got shape {list(envmap_tensor.shape)}"
        )

    arr = envmap_tensor.detach().cpu().to(torch.float32).numpy()
    bitmap = mi.Bitmap(arr)

    return {
        "type": "envmap",
        "bitmap": bitmap,
    }
