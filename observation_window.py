"""
Topocentric asteroid direction and night visibility windows (Astropy + Poliastro).

Ursina-free. Uses ``builtin`` solar system ephemeris (no network).
"""

from __future__ import annotations

import datetime as dt
from typing import List

import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, ICRS, SkyCoord, CartesianRepresentation, get_sun
from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time
from poliastro.twobody import Orbit

from celestial_settings import apply_horizontal_yaw_y_up


def helio_to_topocentric_manual_dir(orbit: Orbit, time_utc: Time, location: EarthLocation) -> np.ndarray:
    """
    Unit direction in the same AltAz→manual convention as ``scenario.generate_game_scenario``
    (before rig snap): (cos h cos A, sin h, cos h sin A) with ``apply_horizontal_yaw_y_up`` applied.
    """
    ot = orbit.propagate(time_utc - orbit.epoch)
    r = ot.r
    sc_ast = SkyCoord(
        CartesianRepresentation(r[0], r[1], r[2]),
        obstime=time_utc,
        frame=ICRS(),
    )
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
    return apply_horizontal_yaw_y_up(manual)


def _asteroid_alt_deg(orbit: Orbit, ti: Time, location: EarthLocation) -> float:
    ot = orbit.propagate(ti - orbit.epoch)
    r = ot.r
    sc_ast = SkyCoord(
        CartesianRepresentation(r[0], r[1], r[2]),
        obstime=ti,
        frame=ICRS(),
    )
    with solar_system_ephemeris.set("builtin"):
        aa = AltAz(obstime=ti, location=location)
        topo = sc_ast.transform_to(aa)
    return float(topo.alt.deg)


def _sun_alt_deg(ti: Time, location: EarthLocation) -> float:
    with solar_system_ephemeris.set("builtin"):
        sun = get_sun(ti)
        sa = sun.transform_to(AltAz(obstime=ti, location=location))
    return float(sa.alt.deg)


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
    t0 = Time(f"{ref_date.isoformat()} 12:00:00", scale="utc")
    n = int(round(24 * 60.0 / step_minutes))
    masks: List[bool] = []

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
            break

    if not any(masks):
        lon_h = float(location.lon.to(u.hourangle).value)
        utc_approx = Time(f"{ref_date.isoformat()} 12:00:00", scale="utc") + (22.0 - lon_h) * u.hour
        return utc_approx - 0.5 * u.hour, utc_approx + 0.5 * u.hour

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


def observations_from_orbit(
    orbit: Orbit,
    times: List[Time],
    location: EarthLocation,
) -> List[dict]:
    """Build three ``preliminary_orbit``-style observation dicts (ICRS RA/Dec, ISO UTC)."""
    out: List[dict] = []
    for ti in times:
        ot = orbit.propagate(ti - orbit.epoch)
        r = ot.r
        sc_ast = SkyCoord(
            CartesianRepresentation(r[0], r[1], r[2]),
            obstime=ti,
            frame=ICRS(),
        )
        with solar_system_ephemeris.set("builtin"):
            aa = AltAz(obstime=ti, location=location)
            topo = sc_ast.transform_to(aa)
        icrs = SkyCoord(alt=topo.alt, az=topo.az, obstime=ti, location=location, frame=AltAz).transform_to(
            ICRS()
        )
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
