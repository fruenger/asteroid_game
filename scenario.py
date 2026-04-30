"""
Procedural game scenario: ephemeris night window, mock observations, poliastro orbit, rig-snapped target.

Ursina-free — safe to import from tests or tooling without Panda3D.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

import rig_kinematics as rk

from observation_window import (
    compute_night_visibility_window_detailed,
    helio_to_topocentric_manual_dir,
    observations_from_orbit,
    utc_time_to_game_time,
)
from orbit_catalog import classical_orbit_from_seed
from orbit_api import preliminary_orbit

# visibility_precache.json "version": bump gen_visibility_precache.py when asteroid window/grid logic changes.
MIN_VISIBILITY_PRECACHE_VERSION = 2


def _snap_object_dir_to_telescope_rig(manual_unit: np.ndarray) -> np.ndarray:
    """Game target must lie on directions reachable by ``rig_kinematics`` / Ursina rig (coarse grid)."""
    v = np.asarray(manual_unit, dtype=np.float64).reshape(3)
    vn = float(np.linalg.norm(v))
    if vn < 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    v = v / vn

    grid_ra = np.arange(-180.0, 180.0 + 1e-6, 5.0)
    grid_dec = np.arange(-85.0, 85.0 + 1e-6, 5.0)
    ra_flat = np.repeat(grid_ra, len(grid_dec))
    dec_flat = np.tile(grid_dec, len(grid_ra))
    cand = rk.optical_axis_world_unit_batch(ra_flat, dec_flat)
    k = int(np.argmax(cand @ v))
    return cand[k].copy()


@dataclass(frozen=True)
class GameScenario:
    """Values needed by game.py and AsteroidGameState (orbit kept for catalog / step 6)."""

    object_dir_cartesian: np.ndarray
    min_time: float
    max_time: float
    orbit: Any
    obs_time: datetime.time
    obs_location: EarthLocation
    today: datetime.date
    reference_date: datetime.date
    session_seed: int
    catalog_display_date: datetime.date
    visibility_t_open_mjd: float
    visibility_t_close_mjd: float
    synthetic_window: bool
    longitude_deg: float
    latitude_deg: float


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_date(key: str) -> datetime.date | None:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    return datetime.date.fromisoformat(raw)


def _load_visibility_precache() -> dict[str, Any] | None:
    p = os.environ.get("GAME_VISIBILITY_PRECACHE", "").strip()
    if not p:
        p = str(Path(__file__).resolve().parent / "assets" / "visibility_precache.json")
    path = Path(p)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _precache_entry(data: dict[str, Any], seed: int) -> dict[str, Any] | None:
    for e in data.get("entries", []):
        if int(e.get("session_seed", -1)) == int(seed):
            return e
    return None


def _window_from_precache_entry(entry: dict[str, Any]) -> tuple[Time, Time]:
    t_open = Time(float(entry["visibility_t_open_mjd"]), format="mjd", scale="utc")
    t_close = Time(float(entry["visibility_t_close_mjd"]), format="mjd", scale="utc")
    return t_open, t_close


def _precache_row_duration_hours(entry: dict[str, Any]) -> float:
    t_open = Time(float(entry["visibility_t_open_mjd"]), format="mjd", scale="utc")
    t_close = Time(float(entry["visibility_t_close_mjd"]), format="mjd", scale="utc")
    return float((t_close - t_open).to(u.h).value)


def _precache_entries_eligible(entries: list[Any], min_span_h: float) -> list[dict[str, Any]]:
    """Rows with duration >= ``min_span_h`` and not flagged ``below_min_span_hours`` in JSON."""
    out: list[dict[str, Any]] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get("below_min_span_hours"):
            continue
        try:
            dur = _precache_row_duration_hours(e)
        except (KeyError, TypeError, ValueError):
            continue
        if dur >= min_span_h:
            out.append(e)
    return out


def _pick_precache_row_stable(eligible: list[dict[str, Any]], picker_seed: int) -> dict[str, Any]:
    """Deterministic pick: sort by ``session_seed``, index = SHA256(picker_seed) mod n."""
    el = sorted(eligible, key=lambda x: int(x.get("session_seed", -1)))
    if not el:
        raise ValueError("precache eligible list empty")
    h = hashlib.sha256(str(int(picker_seed)).encode("utf-8")).digest()
    idx = int.from_bytes(h[:8], "little", signed=False) % len(el)
    return el[idx]


def _precache_alias_enabled() -> bool:
    return os.environ.get("GAME_PRECACHE_ALIAS", "1").strip().lower() not in ("0", "false", "no")


def _ephemeris_diag_enabled() -> bool:
    return os.environ.get("GAME_SILENCE_EPHEMERIS_LOG", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    )


def _ephemeris_log(lines: list[str]) -> None:
    for ln in lines:
        print(ln, file=sys.stderr, flush=True)


def generate_game_scenario(*, verbose: bool = True) -> GameScenario:
    catalog_display_date = datetime.date.today()
    lat_deg = _env_float("GAME_LATITUDE_DEG", 52.0)
    lon_deg = _env_float("GAME_LONGITUDE_DEG", 13.0)
    height_m = _env_float("GAME_HEIGHT_M", 0.0)
    obs_location = EarthLocation(lon=lon_deg * u.deg, lat=lat_deg * u.deg, height=height_m * u.m)

    precache_path = os.environ.get("GAME_VISIBILITY_PRECACHE", "").strip()
    if not precache_path:
        precache_path = str(Path(__file__).resolve().parent / "assets" / "visibility_precache.json")
    precache_file = _load_visibility_precache()
    precache_windows = precache_file
    if precache_file is not None and int(precache_file.get("version", 0)) < MIN_VISIBILITY_PRECACHE_VERSION:
        if verbose:
            print(
                "[scenario] visibility precache version "
                f"{precache_file.get('version', 0)!r} < {MIN_VISIBILITY_PRECACHE_VERSION} "
                "— ignoring window MJDs only (run tools/gen_visibility_precache.py to refresh). "
                "reference_date in file is still used.",
                file=sys.stderr,
                flush=True,
            )
        precache_windows = None

    ref_date: datetime.date
    ref_date_source: str
    ref_date = _parse_date("GAME_REFERENCE_DATE")
    if ref_date is not None:
        ref_date_source = "GAME_REFERENCE_DATE"
    elif precache_file is not None:
        rs = str(precache_file.get("reference_date", "")).strip()
        if rs:
            ref_date = datetime.date.fromisoformat(rs)
            ref_date_source = f"visibility_precache.json (field reference_date) via {precache_path}"
        else:
            ref_date = catalog_display_date
            ref_date_source = (
                f"datetime.date.today()={catalog_display_date.isoformat()} "
                "(precache had no reference_date field)"
            )
    else:
        ref_date = catalog_display_date
        ref_date_source = (
            f"datetime.date.today()={catalog_display_date.isoformat()} "
            f"(no precache at {precache_path})"
        )

    seed_env = os.environ.get("GAME_SESSION_SEED", "").strip()
    if seed_env:
        session_seed = int(seed_env)
        session_seed_source = "GAME_SESSION_SEED"
    else:
        session_seed = int(np.random.randint(0, 2**31 - 1))
        session_seed_source = "random (set GAME_SESSION_SEED for reproducibility)"

    min_alt_deg = _env_float("GAME_MIN_TARGET_ALTITUDE_DEG", 10.0)
    min_span_h = _env_float("GAME_MIN_VISIBILITY_HOURS", 1.0)
    step_minutes = _env_float("GAME_VISIBILITY_STEP_MINUTES", 10.0)
    allow_synthetic = os.environ.get("GAME_ALLOW_SYNTHETIC_WINDOW", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    max_attempts = int(_env_float("GAME_SEED_RETRY_MAX", 96))

    sun_cap_used = float("nan")

    t_open: Time | None = None
    t_close: Time | None = None
    classical: Any = None

    attempt = 0
    cur_seed = int(session_seed)
    initial_seed = int(cur_seed)
    while attempt < max_attempts:
        attempt += 1
        used_precache = False
        used_synthetic = False
        window_resolution = ""
        orbit_seed = int(cur_seed)

        t_open = None
        t_close = None
        precache_tried_reason = ""
        if precache_windows is not None:
            pc_ref = str(precache_windows.get("reference_date", "")).strip()
            if not pc_ref:
                precache_tried_reason = "precache JSON has no reference_date; cannot match session ref"
            elif datetime.date.fromisoformat(pc_ref) != ref_date:
                precache_tried_reason = (
                    f"precache reference_date={pc_ref} != session reference_date={ref_date.isoformat()} "
                    "(windows not loaded from file)"
                )
            else:
                row: dict[str, Any] | None = None
                ent = _precache_entry(precache_windows, cur_seed)
                if ent is not None:
                    t_open, t_close = _window_from_precache_entry(ent)
                    dur_pc = float((t_close - t_open).to(u.h).value)
                    if dur_pc >= min_span_h:
                        row = ent
                        orbit_seed = int(cur_seed)
                    else:
                        precache_tried_reason = (
                            f"precache exact row seed={cur_seed} dur_h={dur_pc:.3f} < min_span_h={min_span_h:g}"
                        )
                        t_open = None
                        t_close = None

                if row is None and _precache_alias_enabled():
                    eligible = _precache_entries_eligible(precache_windows.get("entries", []), min_span_h)
                    if eligible:
                        row = _pick_precache_row_stable(eligible, cur_seed)
                        orbit_seed = int(row["session_seed"])
                        t_open, t_close = _window_from_precache_entry(row)
                        dur_pc = float((t_close - t_open).to(u.h).value)
                        used_precache = True
                        scap = row.get("sun_cap_used")
                        sun_cap_used = float(scap) if scap is not None else float("nan")
                        used_synthetic = False
                        alias_note = "" if orbit_seed == cur_seed else f" alias loop_seed={cur_seed}→orbit_seed={orbit_seed}"
                        window_resolution = (
                            f"precache ({precache_path}): picked row session_seed={orbit_seed} among "
                            f"{len(eligible)} eligible (min_span_h>={min_span_h:g}){alias_note}; "
                            f"dur_h={dur_pc:.3f} sun_cap={scap!r}"
                        )
                    elif ent is None:
                        precache_tried_reason = (
                            f"no precache row for session_seed={cur_seed}; "
                            f"no eligible rows (min_span_h={min_span_h:g})"
                        )
                    else:
                        precache_tried_reason += "; no eligible rows for alias"

                elif row is not None:
                    used_precache = True
                    scap = row.get("sun_cap_used")
                    sun_cap_used = float(scap) if scap is not None else float("nan")
                    used_synthetic = False
                    dur_pc = float((t_close - t_open).to(u.h).value)
                    window_resolution = (
                        f"precache exact ({precache_path}): session_seed={orbit_seed} "
                        f"dur_h={dur_pc:.3f}>=min_span_h={min_span_h:g} sun_cap={scap!r}"
                    )
                elif not _precache_alias_enabled():
                    if ent is None:
                        precache_tried_reason = f"no precache entry for session_seed={cur_seed} (GAME_PRECACHE_ALIAS=0)"
                    else:
                        precache_tried_reason += " (GAME_PRECACHE_ALIAS=0, no remap)"

        classical, cat_idx = classical_orbit_from_seed(orbit_seed, ref_date)

        if t_open is None or t_close is None:
            t_open, t_close, sun_cap_used, used_synthetic = compute_night_visibility_window_detailed(
                classical,
                ref_date,
                obs_location,
                h_min_deg=min_alt_deg,
                step_minutes=step_minutes,
            )
            mode = "synthetic_LMT_fallback" if used_synthetic else "grid_longest_contiguous"
            if not window_resolution:
                extra = f"; {precache_tried_reason}" if precache_tried_reason else ""
                window_resolution = (
                    f"compute_night_visibility_window_detailed ({mode}) orbit_seed={orbit_seed} "
                    f"catalog_index={cat_idx} step_minutes={step_minutes:g} h_min_deg={min_alt_deg:g}{extra}"
                )
            elif not used_precache:
                window_resolution += (
                    f" → then compute_night_visibility_window_detailed ({mode}) catalog_index={cat_idx}"
                )

        dur_h = float((t_close - t_open).to(u.h).value)
        bad_span = dur_h < min_span_h
        bad_synth = used_synthetic and not allow_synthetic
        if not bad_span and not bad_synth:
            session_seed = orbit_seed
            break
        if verbose and bad_synth:
            print(
                "[scenario] synthetic visibility window rejected (set GAME_ALLOW_SYNTHETIC_WINDOW=1); retrying seed…"
            )
        cur_seed = (cur_seed + 12_345) % (2**31)
    else:
        raise RuntimeError(
            "generate_game_scenario: could not find a visibility window after "
            f"{max_attempts} attempts (min_span_h={min_span_h}, allow_synthetic={allow_synthetic}). "
            "Try GAME_SESSION_SEED, GAME_REFERENCE_DATE, or GAME_ALLOW_SYNTHETIC_WINDOW=1."
        )

    assert t_open is not None and t_close is not None
    times_3 = [
        t_open + f * (t_close - t_open) for f in (0.25, 0.5, 0.75)
    ]
    observations = observations_from_orbit(classical, times_3, obs_location)
    orbit = preliminary_orbit(observations)
    if verbose:
        print(
            f"[scenario] seed={session_seed} ref={ref_date} precache={used_precache} "
            f"synthetic={used_synthetic} sun_cap={sun_cap_used} dur_h={dur_h:.3f}"
        )
        print(orbit.a, orbit.r_a, orbit.r_p, orbit.ecc, orbit.inc, orbit.period)

    min_time = float(utc_time_to_game_time(t_open, obs_location))
    max_time = float(utc_time_to_game_time(t_close, obs_location))

    if _ephemeris_diag_enabled():
        if isinstance(sun_cap_used, float) and math.isnan(sun_cap_used):
            cap_repr: str | float = "nan"
        else:
            cap_repr = float(sun_cap_used)
        retry_note = ""
        if attempt > 1:
            retry_note = (
                f" (seed_retry attempts={attempt} initial_seed={initial_seed}→final_seed={session_seed}; "
                f"GAME_SEED_RETRY_MAX={max_attempts} GAME_MIN_VISIBILITY_HOURS={min_span_h:g} "
                f"GAME_ALLOW_SYNTHETIC_WINDOW={allow_synthetic})"
            )
        elif int(session_seed) != int(initial_seed):
            retry_note = (
                f" (precache_alias: first_loop_seed={initial_seed} → effective_session_seed={session_seed} "
                f"for orbit+Fenster)"
            )
        _ephemeris_log(
            [
                "[asteroid_ephemeris] ── session scenario / Nachtfenster / Referenzdatum ──",
                f"[asteroid_ephemeris] reference_date={ref_date.isoformat()}  ← {ref_date_source}",
                f"[asteroid_ephemeris] catalog_display_date={catalog_display_date.isoformat()} (UI/Katalog „heute“ vs. Physik siehe Hilfe)",
                f"[asteroid_ephemeris] observer lat={lat_deg:g}° lon={lon_deg:g}° height_m={height_m:g}",
                f"[asteroid_ephemeris] session_seed={session_seed}  ← {session_seed_source}{retry_note}",
                f"[asteroid_ephemeris] precache_loaded={'yes' if precache_file is not None else 'no'} path={precache_path} "
                f"GAME_PRECACHE_ALIAS={_precache_alias_enabled()}",
                f"[asteroid_ephemeris] window: {window_resolution}",
                f"[asteroid_ephemeris] used_precache_json={used_precache} synthetic_window={used_synthetic} "
                f"sun_cap_used={cap_repr!r} dur_h={dur_h:.4f}",
                f"[asteroid_ephemeris] game min_time/max_time from utc_time_to_game_time(t_open/t_close, loc): "
                f"{min_time:.6f} … {max_time:.6f}  (Leertaste nutzt UTC-Vergleich game_time_to_utc im State)",
                f"[asteroid_ephemeris] visibility UTC t_open={t_open.iso}  t_close={t_close.iso}",
                "[asteroid_ephemeris] runtime: Sonne+Ziel pro Tick via game_time_to_utc(time_now, reference_date, loc); "
                "stille Logs: GAME_SILENCE_EPHEMERIS_LOG=1",
                "[asteroid_ephemeris] ────────────────────────────────────────────────",
            ]
        )

    t_mid = t_open + 0.5 * (t_close - t_open)
    manual0 = helio_to_topocentric_manual_dir(orbit, t_mid, obs_location)
    object_dir_cartesian = _snap_object_dir_to_telescope_rig(manual0)

    dtm_mid = t_mid.to_datetime(timezone=datetime.timezone.utc)
    obs_time = datetime.time(dtm_mid.hour, dtm_mid.minute, dtm_mid.second)

    return GameScenario(
        object_dir_cartesian=object_dir_cartesian,
        min_time=min_time,
        max_time=max_time,
        orbit=orbit,
        obs_time=obs_time,
        obs_location=obs_location,
        today=catalog_display_date,
        reference_date=ref_date,
        session_seed=int(session_seed),
        catalog_display_date=catalog_display_date,
        visibility_t_open_mjd=float(t_open.mjd),
        visibility_t_close_mjd=float(t_close.mjd),
        synthetic_window=bool(used_synthetic),
        longitude_deg=float(lon_deg),
        latitude_deg=float(lat_deg),
    )
