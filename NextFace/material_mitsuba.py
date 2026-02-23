"""Mitsuba 3 material construction from texture tensors.

Converts PyTorch texture tensors (diffuse, specular, roughness) into a Mitsuba 3
BSDF dict suitable for scene assembly. Uses a principled BSDF to combine
diffuse reflectance, specular reflectance, and roughness.
"""

import torch
import numpy as np
import mitsuba as mi


def _tensor_to_bitmap(tensor):
    """Convert a PyTorch [H, W, C] tensor to a Mitsuba Bitmap.

    Args:
        tensor: [H, W, C] float32 torch tensor (C=1 or C=3).

    Returns:
        mi.Bitmap: Mitsuba bitmap object.
    """
    arr = tensor.detach().cpu().to(torch.float32).numpy()
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]  # [H, W, 1] -> [H, W] for single-channel
    return mi.Bitmap(arr)


def _bitmap_texture(tensor):
    """Create a Mitsuba bitmap texture dict from a PyTorch tensor.

    Args:
        tensor: [H, W, C] float32 torch tensor.

    Returns:
        dict: Mitsuba texture dict with type 'bitmap'.
    """
    return {
        "type": "bitmap",
        "bitmap": _tensor_to_bitmap(tensor),
    }


def build_material(diffuse, specular=None, roughness=None):
    """Build a Mitsuba BSDF dict from texture tensors.

    Uses principled BSDF mapping:
    - diffuse → base_color (texture)
    - roughness → roughness (texture)
    - specular → specular (scalar float, mean of RGB channels)

    The principled BSDF's specular parameter is float-only, so per-pixel
    specular variation is approximated by the mean reflectance. For face
    reconstruction this is acceptable since specular variation is small.

    Args:
        diffuse: [H, W, 3] float32 torch tensor for diffuse albedo.
        specular: [H, W, 3] float32 torch tensor for specular reflectance (optional).
        roughness: [H, W, 1] float32 torch tensor for roughness (optional).

    Returns:
        dict: Mitsuba BSDF dict suitable for mi.load_dict() scene assembly.
    """
    bsdf = {
        "type": "principled",
        "base_color": _bitmap_texture(diffuse),
    }

    if specular is not None:
        # Principled BSDF specular is float-only; use mean reflectance
        spec_mean = float(specular.detach().cpu().to(torch.float32).mean())
        bsdf["specular"] = max(0.0, min(1.0, spec_mean))

    if roughness is not None:
        bsdf["roughness"] = _bitmap_texture(roughness)

    return bsdf
