"""Tests for US-002: Mitsuba 3 variant selection utility."""

import mitsuba as mi


def test_us002_variant_returns_string():
    """ensure_variant() returns a valid variant string."""
    from rendering._variant import ensure_variant

    result = ensure_variant()
    assert isinstance(result, str)
    assert result in ("cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb")


def test_us002_variant_idempotent():
    """Calling ensure_variant() twice returns the same result without error."""
    from rendering._variant import ensure_variant

    first = ensure_variant()
    second = ensure_variant()
    assert first == second


def test_us002_variant_is_active():
    """After ensure_variant(), mi.variant() matches the returned value."""
    from rendering._variant import ensure_variant

    variant = ensure_variant()
    assert mi.variant() == variant


def test_us002_variant_priority():
    """Variant selection respects priority: cuda_ad_rgb > llvm_ad_rgb > scalar_rgb."""
    from rendering._variant import _VARIANT_PRIORITY

    assert _VARIANT_PRIORITY[0] == "cuda_ad_rgb"
    assert _VARIANT_PRIORITY[1] == "llvm_ad_rgb"
    assert _VARIANT_PRIORITY[2] == "scalar_rgb"
