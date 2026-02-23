"""Shared pytest configuration and fixtures."""

import os
import shutil


def _find_libllvm() -> str | None:
    """Find libLLVM.dylib path for DrJIT on macOS."""
    # Explicit env var takes priority
    if os.environ.get("DRJIT_LIBLLVM_PATH"):
        return os.environ["DRJIT_LIBLLVM_PATH"]

    # Homebrew LLVM (Apple Silicon and Intel)
    candidates = [
        "/opt/homebrew/opt/llvm/lib/libLLVM.dylib",
        "/usr/local/opt/llvm/lib/libLLVM.dylib",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    # Try to find via `brew --prefix llvm`
    brew = shutil.which("brew")
    if brew:
        import subprocess

        try:
            prefix = subprocess.check_output(
                [brew, "--prefix", "llvm"], text=True, timeout=5
            ).strip()
            candidate = os.path.join(prefix, "lib", "libLLVM.dylib")
            if os.path.isfile(candidate):
                return candidate
        except (subprocess.SubprocessError, OSError):
            pass

    return None


# Set DRJIT_LIBLLVM_PATH early so mitsuba/drjit can find LLVM
_llvm_path = _find_libllvm()
if _llvm_path:
    os.environ.setdefault("DRJIT_LIBLLVM_PATH", _llvm_path)
