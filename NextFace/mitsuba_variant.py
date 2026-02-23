"""Mitsuba 3 variant selection utility.

Selects the correct Mitsuba variant (cuda_ad_rgb or llvm_ad_rgb)
based on GPU availability. Must be imported before any other mitsuba usage.
"""

import mitsuba as mi

_variant_set = False
_active_variant: str | None = None

# Preferred variants in priority order (AD = automatic differentiation support)
_VARIANT_PRIORITY = [
    "cuda_ad_rgb",
    "llvm_ad_rgb",
    "scalar_rgb",
]


def ensure_variant() -> str:
    """Select and set the appropriate Mitsuba variant.

    Tries variants in priority order: cuda_ad_rgb > llvm_ad_rgb > scalar_rgb.
    Calls mi.set_variant() as a side effect. Idempotent: safe to call multiple times.

    Returns:
        The variant string that was set.

    Raises:
        RuntimeError: If no supported variant can be activated.
    """
    global _variant_set, _active_variant

    if _variant_set and _active_variant is not None:
        return _active_variant

    variant = _choose_and_activate_variant()
    _active_variant = variant
    _variant_set = True
    return variant


def _choose_and_activate_variant() -> str:
    """Try each variant in priority order, returning the first that activates."""
    available = mi.variants()
    for variant in _VARIANT_PRIORITY:
        if variant in available:
            try:
                mi.set_variant(variant)
                return variant
            except (ImportError, RuntimeError):
                continue
    raise RuntimeError(
        f"No supported Mitsuba variant could be activated. "
        f"Available: {available}, tried: {_VARIANT_PRIORITY}"
    )
