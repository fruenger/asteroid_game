"""
Topocentric asteroid direction and night visibility windows (Astropy + Poliastro).

Ursina-free. Uses ``builtin`` solar system ephemeris (no network).
"""

from __future__ import annotations

import datetime as dt
from typing import List, Tuple

import numpy as np
import astropy.units as u
from astropy.coordinates import (
    AltAz,
    EarthLocation,
    ICRS,
    SkyCoord,
    CartesianRepresentation,
    get_body_barycentric,
    get_sun,
)
from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import propagate as _poliastro_propagate

from celestial_settings import apply_horizontal_yaw_y_up


def _asteroid_altaz_manual_flip_azimuth_180(manual: np.ndarray) -> np.ndarray:
    """
    In Astropy's AltAz unit form (x=cos h cos A, y=sin h, z=cos h sin A), adding **180° to azimuth**
    is ``(-x, y, -z)`` — flips the horizontal direction without negating **y**, so altitude / above
    horizon stay correct (unlike ``(-x,-y,z)`` which breaks day/night).

    ``get_sun`` → same manual path does not need this; poliastro SSB + Sun→asteroid → ``AltAz`` can
    land mirrored in the game's (x,z) plane relative to the dome/stars. Applied only to asteroid.
    """
    a = np.asarray(manual, dtype=np.float64).reshape(3)
    return np.array([-float(a[0]), float(a[1]), -float(a[2])], dtype=np.float64)


def _propagate_orbit(orbit: Orbit, time_utc: Time) -> Orbit:
    """
    Poliastro Kepler propagation. Mock / ``preliminary_orbit`` orbits can be numerically stiff;
    ``Orbit.propagate`` uses a tight default iteration cap — relax on failure.
    """
    dt = time_utc - orbit.epoch
    try:
        return _poliastro_propagate(orbit, dt, rtol=1e-9, numiter=96)
    except RuntimeError:
        return _poliastro_propagate(orbit, dt, rtol=1e-5, numiter=512)


def _sun_to_asteroid_cartesian_au(r) -> CartesianRepresentation:
    """Poliastro ``orbit.r`` is Sun→asteroid in km; Astropy SSB vectors use AU."""
    if hasattr(r, "to"):
        v = r.to(u.AU)
        return CartesianRepresentation(v[0], v[1], v[2])
    arr = np.asarray(r, dtype=np.float64).ravel()
    q = arr * u.km
    q = q.to(u.AU)
    return CartesianRepresentation(q[0], q[1], q[2])


def _asteroid_icrs_skycoord(orbit: Orbit, time_utc: Time) -> SkyCoord:
    """
    ICRS position of the asteroid from the solar system barycentre.

    Poliastro ``orbit.r`` is Sun→asteroid. ``get_body_barycentric('sun')`` is SSB→Sun; their sum
    is SSB→asteroid. That is what ``SkyCoord(..., frame=ICRS())`` expects: **absolute** Cartesian
    coordinates, **not** Earth→asteroid (a difference vector must not be passed as if it were a
    position from the origin — that breaks ``transform_to(AltAz)`` and looked like wrong
    meridian / reversed diurnal motion).

    ``AltAz(location=...)`` then gives the correct **topocentric** line of sight (parallax).
    """
    ot = _propagate_orbit(orbit, time_utc)
    dr = _sun_to_asteroid_cartesian_au(ot.r)
    with solar_system_ephemeris.set("builtin"):
        sun_ssb = get_body_barycentric("sun", time_utc)
    ast_ssb = sun_ssb + dr
    return SkyCoord(ast_ssb, obstime=time_utc, frame=ICRS())


def helio_to_topocentric_manual_dir(orbit: Orbit, time_utc: Time, location: EarthLocation) -> np.ndarray:
    """
    Unit direction in the same AltAz→manual convention as ``scenario.generate_game_scenario``
    (before rig snap): (cos h cos A, sin h, cos h sin A) with ``apply_horizontal_yaw_y_up`` applied.

    Uses absolute barycentric ``SkyCoord`` → ``AltAz`` like ``get_sun`` (see ``_asteroid_icrs_skycoord``).

    Applies ``_asteroid_altaz_manual_flip_azimuth_180``: horizontal mirror vs ``get_sun`` only, so the
    marker matches southern culmination / diurnal sense with the star dome without touching sun.
    """
    sc_ast = _asteroid_icrs_skycoord(orbit, time_utc)
    with solar_system_ephemeris.set("builtin"):
        aa = AltAz(obstime=time_utc, location=location)
        topo = sc_ast.transform_to(aa)
    alt_rad = float(topo.alt.to_value(u.rad))
    az_rad = float(topo.az.to_value(u.rad))
    manual = np.array(
        [
            float(np.cos(alt_rad) * np.cos(az_rad)),
            float(np.sin(alt_rad)),
            float(np.cos(alt_rad) * np.sin(az_rad)),
        ],
        dtype=np.float64,
    )
    manual = _asteroid_altaz_manual_flip_azimuth_180(manual)
    return apply_horizontal_yaw_y_up(manual)


def _asteroid_alt_deg(orbit: Orbit, ti: Time, location: EarthLocation) -> float:
    sc_ast = _asteroid_icrs_skycoord(orbit, ti)
    with solar_system_ephemeris.set("builtin"):
        aa = AltAz(obstime=ti, location=location)
        topo = sc_ast.transform_to(aa)
    return float(topo.alt.deg)


def _sun_alt_deg(ti: Time, location: EarthLocation) -> float:
    with solar_system_ephemeris.set("builtin"):
        sun = get_sun(ti)
        sa = sun.transform_to(AltAz(obstime=ti, location=location))
    return float(sa.alt.deg)


def compute_night_visibility_window_detailed(
    orbit: Orbit,
    ref_date: dt.date,
    location: EarthLocation,
    h_min_deg: float,
    sun_max_alt_deg: float = -12.0,
    step_minutes: float = 10.0,
) -> Tuple[Time, Time, float, bool]:
    """
    Like ``compute_night_visibility_window`` but also returns ``sun_cap_used`` (the sun-altitude
    cap that produced the mask, or ``float('nan')`` for the synthetic fallback) and
    ``used_synthetic_fallback`` (True when the 1 h LMT band was used).
    """
    t0 = Time(f"{ref_date.isoformat()} 12:00:00", scale="utc")
    n = int(round(24 * 60.0 / step_minutes))
    masks: List[bool] = []
    sun_cap_used = float("nan")

    def _fill_mask(sun_cap: float) -> None:
        masks.clear()
        for i in range(n + 1):
            ti = t0 + i * step_minutes * u.minute
            try:
                a_alt = _asteroid_alt_deg(orbit, ti, location)
                s_alt = _sun_alt_deg(ti, location)
            except Exception:
                masks.append(False)
                continue
            masks.append(a_alt >= h_min_deg and s_alt <= sun_cap)

    for cap in (float(sun_max_alt_deg), -3.0, 0.0):
        _fill_mask(cap)
        if any(masks):
            sun_cap_used = float(cap)
            break

    if not any(masks):
        lon_h = float(location.lon.to(u.hourangle).value)
        utc_approx = Time(f"{ref_date.isoformat()} 12:00:00", scale="utc") + (22.0 - lon_h) * u.hour
        return utc_approx - 0.5 * u.hour, utc_approx + 0.5 * u.hour, sun_cap_used, True

    best_lo = 0
    best_len = 0
    lo = 0
    while lo <= n:
        if not masks[lo]:
            lo += 1
            continue
        hi = lo
        while hi <= n and masks[hi]:
            hi += 1
        ln = hi - lo
        if ln > best_len:
            best_len = ln
            best_lo = lo
        lo = hi

    i_open = best_lo
    i_close = best_lo + best_len - 1
    t_open = t0 + i_open * step_minutes * u.minute
    t_close = t0 + i_close * step_minutes * u.minute
    return t_open, t_close, sun_cap_used, False


def compute_night_visibility_window(
    orbit: Orbit,
    ref_date: dt.date,
    location: EarthLocation,
    h_min_deg: float,
    sun_max_alt_deg: float = -12.0,
    step_minutes: float = 10.0,
) -> tuple[Time, Time]:
    """
    Coarse grid from ref_date 12:00 UTC through the following 24 h; return first/last times
    (UTC) inside the **longest** contiguous span where Sun is at or below the active sun cap
    and asteroid altitude is at least ``h_min_deg``.

    Sun caps are tried in order: ``sun_max_alt_deg`` (default nautical dusk **-12°**), then **-3°**,
    then **0°** (centre at or below the horizon). **Positive** caps are not used: allowing the Sun
    a few degrees *above* the horizon (e.g. +6°) produced spurious **daytime** windows (e.g.
    09:30–10:30 local) when the asteroid was still geometrically high.

    If nothing qualifies, return a 1 h synthetic band around 22:00 local mean time on ref_date.
    """
    t_open, t_close, _, _ = compute_night_visibility_window_detailed(
        orbit, ref_date, location, h_min_deg, sun_max_alt_deg, step_minutes
    )
    return t_open, t_close


def _local_mean_time_hours(t: Time, location: EarthLocation) -> float:
    """Decimal hour 0–24, local mean time (UTC + longitude in hours)."""
    d = t.to_datetime(timezone=dt.timezone.utc)
    utc_dec = d.hour + d.minute / 60.0 + d.second / 3600.0 + d.microsecond / 3.6e9
    lon_h = float(location.lon.to(u.hourangle).value)
    return (utc_dec + lon_h) % 24.0


def _day2range(hours: float) -> float:
    """Same as ``orbit_api.day2range`` (avoid import cycles)."""
    return ((float(hours) + 12.0) % 24.0) / 24.0


def utc_time_to_game_time(t: Time, location: EarthLocation) -> float:
    """Map UTC instant to game ``time_now`` in [0,1) via local mean time + ``day2range``."""
    lm = _local_mean_time_hours(t, location)
    return float(_day2range(lm))


def game_time_to_utc(time_now: float, ref_date: dt.date, location: EarthLocation) -> Time:
    """
    Approximate inverse of ``utc_time_to_game_time`` for a given ``ref_date`` anchor.

    UTC time-of-day is determined uniquely from ``time_now`` and longitude; the calendar day is
    chosen among a small range around ``ref_date`` so the result matches the ephemeris/night
    convention (grid starting ref_date 12:00 UTC).
    """
    g = float(time_now) % 1.0
    lm = (g * 24.0 + 12.0) % 24.0
    lon_h = float(location.lon.to(u.hourangle).value)
    utc_h = (lm - lon_h + 48.0) % 24.0
    anchor = Time(f"{ref_date.isoformat()} 18:00:00", scale="utc")
    best: Time | None = None
    best_score = float("inf")
    best_err = float("inf")
    for day_off in range(-1, 6):
        day = ref_date + dt.timedelta(days=day_off)
        dtm = dt.datetime.combine(day, dt.time(0, 0, 0), tzinfo=dt.timezone.utc) + dt.timedelta(
            seconds=utc_h * 3600.0
        )
        tt = Time(dtm, scale="utc")
        g2 = utc_time_to_game_time(tt, location)
        err = abs(((g2 - g + 0.5) % 1.0) - 0.5)
        score = abs(float((tt - anchor).to_value(u.day)))
        if err < best_err - 1e-12 or (abs(err - best_err) <= 1e-12 and score < best_score):
            best_err = err
            best_score = score
            best = tt
    assert best is not None
    return best


def solar_topocentric_manual_dir(time_utc: Time, location: EarthLocation) -> np.ndarray:
    """Sun unit vector in the same manual AltAz convention as ``helio_to_topocentric_manual_dir``."""
    with solar_system_ephemeris.set("builtin"):
        sun = get_sun(time_utc)
        aa = AltAz(obstime=time_utc, location=location)
        topo = sun.transform_to(aa)
    alt_rad = float(topo.alt.to_value(u.rad))
    az_rad = float(topo.az.to_value(u.rad))
    manual = np.array(
        [
            float(np.cos(alt_rad) * np.cos(az_rad)),
            float(np.sin(alt_rad)),
            float(np.cos(alt_rad) * np.sin(az_rad)),
        ],
        dtype=np.float64,
    )
    return apply_horizontal_yaw_y_up(manual)


def game_time_in_visibility_window(
    time_now: float,
    t_open: Time,
    t_close: Time,
    ref_date: dt.date,
    location: EarthLocation,
) -> bool:
    """True iff ``game_time_to_utc(time_now, …)`` lies strictly between ``t_open`` and ``t_close`` (UTC)."""
    tutc = game_time_to_utc(float(time_now) % 1.0, ref_date, location)
    return float(t_open.mjd) < float(tutc.mjd) < float(t_close.mjd)


def observations_from_orbit(
    orbit: Orbit,
    times: List[Time],
    location: EarthLocation,
) -> List[dict]:
    """Build three ``preliminary_orbit``-style observation dicts (ICRS RA/Dec, ISO UTC)."""
    out: List[dict] = []
    for ti in times:
        sc_ast = _asteroid_icrs_skycoord(orbit, ti)
        icrs = sc_ast.transform_to(ICRS())
        ra_deg = float(icrs.ra.to_value(u.deg))
        dec_deg = float(icrs.dec.to_value(u.deg))
        dtm = ti.to_datetime(timezone=dt.timezone.utc)
        out.append(
            {
                "ra": ra_deg,
                "dec": dec_deg,
                "time": dtm.strftime("%Y-%m-%dT%H:%M:%S"),
                "location": location,
            }
        )
    return out
