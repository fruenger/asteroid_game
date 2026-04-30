"""
Idle-timeout resolution for Ursina CLI + env.

Native ``asteroid_game_touch`` uses ``--idle-timeout`` / ``--no-idle-timeout`` and ``ASTEROID_IDLE_TIMEOUT_SEC`` directly;
the desktop Ursina launcher shares the **same meanings** via this module plus ``GAME_IDLE_EXIT_SEC``.
"""

from __future__ import annotations

import os
import sys


def idle_exit_timeout_sec_from_env_only() -> float:
    """0 = disabled."""
    for k in ("GAME_IDLE_EXIT_SEC", "ASTEROID_IDLE_TIMEOUT_SEC"):
        raw = os.environ.get(k, "").strip()
        if not raw:
            continue
        try:
            return max(0.0, float(raw))
        except ValueError:
            continue
    return 0.0


def resolved_idle_exit_timeout_sec() -> float:
    """CLI overrides env (same precedence as native ``asteroid_game_touch``)."""
    overridden = False
    val = 0.0
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--no-idle-timeout":
            overridden = True
            val = 0.0
            i += 1
            continue
        if a.startswith("--idle-timeout="):
            raw = a.split("=", 1)[1].strip()
            overridden = True
            try:
                val = max(0.0, float(raw))
            except ValueError:
                val = 0.0
            i += 1
            continue
        if a == "--idle-timeout":
            overridden = True
            if i + 1 < len(argv):
                try:
                    val = max(0.0, float(argv[i + 1]))
                except ValueError:
                    val = 0.0
                i += 2
            else:
                i += 1
            continue
        i += 1

    if overridden:
        return val
    return idle_exit_timeout_sec_from_env_only()
