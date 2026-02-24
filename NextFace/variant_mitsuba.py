"""Shim - redirects to new location."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from rendering._variant import *
from rendering._variant import ensure_variant, _VARIANT_PRIORITY, _variant_set, _active_variant
