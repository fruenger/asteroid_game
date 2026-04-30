"""Mutable cross-module state for the Ursina app (set during bootstrap)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from game_state import AsteroidGameState

game_paused: bool = False
game_gs: AsteroidGameState | None = None
laser_spawned: bool = False
cheat_through: bool = False
prev_message: str = ""

idle_exit_timeout_sec: float = 0.0
last_activity_wall_time: float = 0.0
# Incremented on locale changes (language bar / Astro launcher ``ASTRO_LANG``) — UI invalidates caches.
i18n_epoch: int = 0
# Filled by game_app.bootstrap(); handlers read this namespace.
runtime: Any = None
