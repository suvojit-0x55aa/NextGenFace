"""Mitsuba 3 perspective camera builder.

Constructs a Mitsuba camera dict from focal length, image width, and image height,
matching the original NextFace/PyRedner camera conventions.
"""

import math


def build_camera(focal, width, height, clip_near=10.0):
    """Build a Mitsuba perspective camera dict.

    Args:
        focal: Focal length in pixels (scalar or 0-d tensor).
        width: Image width in pixels.
        height: Image height in pixels.
        clip_near: Near clipping plane distance (default 10.0).

    Returns:
        dict: Mitsuba sensor dict suitable for mi.load_dict().
    """
    focal_val = float(focal.detach() if hasattr(focal, 'detach') else focal)
    width_val = int(width)
    height_val = int(height)

    # FOV formula matching original: fov = 360 * atan(width / (2*focal)) / pi
    fov = 360.0 * math.atan(width_val / (2.0 * focal_val)) / math.pi

    import mitsuba as mi

    # Camera at origin, looking at +Z, up vector (0, -1, 0)
    to_world = mi.ScalarTransform4f.look_at(
        origin=[0.0, 0.0, 0.0],
        target=[0.0, 0.0, 1.0],
        up=[0.0, -1.0, 0.0],
    )

    return {
        "type": "perspective",
        "fov": fov,
        "near_clip": clip_near,
        "to_world": to_world,
        "film": {
            "type": "hdrfilm",
            "width": width_val,
            "height": height_val,
            "pixel_format": "rgba",
        },
    }
