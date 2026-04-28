"""Early process setup before Ursina / Panda3D (faulthandler, shadow PRC)."""

from __future__ import annotations

import faulthandler
import os

faulthandler.enable(all_threads=True)

# Before any other Panda3D import: shadow buffer depth (default 16; try 24 if bands appear).
# Shadows on by default; set ASTEROID_GAME_SHADOWS=0 if the GPU driver crashes (e.g. Intel Arc + Wayland HW path).
if os.environ.get("ASTEROID_GAME_SHADOWS", "1") != "0":
    from panda3d.core import loadPrcFileData

    _shadow_depth_bits = os.environ.get("ASTEROID_GAME_SHADOW_DEPTH_BITS", "16").strip()
    if _shadow_depth_bits.isdigit():
        loadPrcFileData("", f"shadow-depth-bits {_shadow_depth_bits}")
