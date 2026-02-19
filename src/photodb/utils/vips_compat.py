"""
Compatibility shim for pyvips on macOS with Homebrew.

pyvips uses cffi to dlopen libvips.42.dylib, but Homebrew's library path
(/opt/homebrew/lib on ARM, /usr/local/lib on Intel) is not in the default
dlopen search path. This module sets DYLD_FALLBACK_LIBRARY_PATH before
importing pyvips.

Usage:
    Import this module instead of importing pyvips directly:

        from ..utils.vips_compat import pyvips
"""

import os
import sys

if sys.platform == "darwin":
    _fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    _brew_paths = "/opt/homebrew/lib:/usr/local/lib"
    if _brew_paths not in _fallback:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
            f"{_brew_paths}:{_fallback}" if _fallback else _brew_paths
        )

import pyvips  # noqa: E402

__all__ = ["pyvips"]
