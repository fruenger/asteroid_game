"""Orbit summary text and catalog file append — no Ursina (shared by desktop + asteroid_game_touch)."""

from __future__ import annotations

import datetime
import os
from typing import Any


def orbital_elements_block(orbit: Any) -> str:
    import astropy.units as u

    from game_i18n import tr

    return tr("catalog.orbit_block").format(
        a_au=orbit.a.to_value(u.AU),
        ecc=float(orbit.ecc),
        inc_deg=orbit.inc.to_value(u.deg),
        Om_deg=orbit.raan.to_value(u.deg),
        om_deg=orbit.argp.to_value(u.deg),
        p_yr=orbit.period.to_value(u.yr),
        rp_au=orbit.r_p.to_value(u.AU),
        ra_au=orbit.r_a.to_value(u.AU),
    )


def default_object_name() -> str:
    return "OST/%s" % datetime.datetime.now().strftime(r"%Y-%m-%d %H-%M-%S")


def append_catalog_entry(
    obj_name: str,
    discoverer: str,
    path: str | None = None,
    *,
    a_au: float | None = None,
    period_yr: float | None = None,
) -> None:
    path = path or os.environ.get("GAME_USER_SAVES_PATH", "user_saves.dat")
    ts = datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
    on = obj_name.replace("\n", " ").replace(",", " ")[:200]
    dn = discoverer.replace("\n", " ").replace(",", " ")[:200]
    with open(path, "a", encoding="utf-8") as f:
        if a_au is not None and period_yr is not None:
            f.write("%s, %s, %s, %.6f, %.6f\n" % (ts, on, dn, float(a_au), float(period_yr)))
        else:
            f.write("%s, %s, %s\n" % (ts, on, dn))


def parse_catalog_line(line: str) -> tuple[str, str, str, float | None, float | None]:
    """One line from user_saves: ``ts, name, team`` or ``ts, name, team, a_AU, P_yr``."""
    line = line.strip()
    if not line:
        return ("", "", "", None, None)
    parts = [p.strip() for p in line.split(", ")]
    if len(parts) >= 5:
        try:
            a = float(parts[3])
            p = float(parts[4])
            return (parts[0], parts[1], parts[2], a, p)
        except (ValueError, IndexError):
            pass
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2], None, None)
    return ("", "", "", None, None)


def load_catalog_entries(path: str | None = None) -> list[tuple[str, str, str, float | None, float | None]]:
    path = path or os.environ.get("GAME_USER_SAVES_PATH", "user_saves.dat")
    out: list[tuple[str, str, str, float | None, float | None]] = []
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            ts, on, dn, a, p = parse_catalog_line(line)
            if on or dn or ts:
                out.append((ts, on, dn, a, p))
    return out
