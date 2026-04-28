"""
Compute-only API for the B1 native host (no Ursina / Panda).
Callable from embedded Python via C API or JSON wrappers.
"""

from __future__ import annotations

import json
from typing import Any, List, Union

import numpy as np
from astropy.coordinates import (
    EarthLocation,
    get_body_barycentric,
    get_sun,
    solar_system_ephemeris,
)
from astropy.time import Time
import astropy.units as u
from poliastro.bodies import Sun
from poliastro.twobody import Orbit

from celestial_settings import apply_horizontal_yaw_y_up, diurnal_sign


def day2range(hours: float) -> float:
    """Map clock hour (0–24 style) to game day parameter in [0, 1). Same as game.py."""
    return ((float(hours) + 12.0) % 24.0) / 24.0


def range2day(i: float) -> float:
    """Inverse of day2range domain mapping used in game.py."""
    return ((float(i) + 0.5) % 1.0) * 24.0


def time_str(time_of_day: float) -> str:
    """Format game time-of-day in [0,1] as HH:MM (game.py convention)."""
    hour = int(np.floor((float(time_of_day) + 0.5) * 24.0) % 24)
    minute = int(np.floor(((float(time_of_day) + 0.5) * 24.0 * 60.0) % 60.0))
    return f"{hour:02d}:{minute:02d}"


def vector_magnitude(vec) -> float:
    """Euclidean length (game.py `normalize` was this, not a normalized vector)."""
    return float(np.sqrt(np.sum(np.asarray(vec, dtype=np.float64) ** 2)))


def dispatch_compute_json(json_in: str) -> str:
    """
    Single JSON entry for hosts / tests (no Ursina).

    ops:
      {"op":"day2range","hours": 14.5}
      {"op":"range2day","t": 0.35}
      {"op":"time_str","t": 0.35}
      {"op":"vec_mag","v": [1,2,3]}
    """
    try:
        d = json.loads(json_in)
        op = d.get("op")
        if op == "day2range":
            return json.dumps({"ok": True, "t": day2range(float(d["hours"]))})
        if op == "range2day":
            return json.dumps({"ok": True, "hours": range2day(float(d["t"]))})
        if op == "time_str":
            return json.dumps({"ok": True, "s": time_str(float(d["t"]))})
        if op == "vec_mag":
            return json.dumps({"ok": True, "mag": vector_magnitude(d["v"])})
        return json.dumps({"ok": False, "error": f"unknown op: {op!r}"})
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)})


def ra_dec_to_unitvec(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    return np.array(
        [
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ]
    )


def _location_from_obj(loc: Union[EarthLocation, dict[str, Any]]) -> EarthLocation:
    if isinstance(loc, EarthLocation):
        return loc
    return EarthLocation(
        lat=float(loc["lat_deg"]) * u.deg,
        lon=float(loc["lon_deg"]) * u.deg,
        height=float(loc.get("height_m", 0.0)) * u.m,
    )


def sun_direction(t: float, latitude: float, sun_declination: float) -> np.ndarray:
    """
    Unit vector toward the sun (same geometry as game.py).

    t: time of day in [0, 1], 0.5 = noon.
    latitude, sun_declination: degrees.
    """
    lat_rad = np.radians(latitude)
    dec_rad = np.radians(sun_declination)
    hour_angle = diurnal_sign() * (t - 0.5) * 2.0 * np.pi
    sin_alt = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_angle)
    alt = np.arcsin(np.clip(sin_alt, -1.0, 1.0))
    az = np.arctan2(
        -np.cos(dec_rad) * np.cos(lat_rad) * np.sin(hour_angle),
        -np.sin(dec_rad) - np.sin(lat_rad) * np.sin(alt),
    )
    x = np.cos(alt) * np.cos(az)
    y = -np.sin(alt)
    z = -np.cos(alt) * np.sin(az)
    x = -x
    z = -z
    return apply_horizontal_yaw_y_up(np.array([x, y, z]))


def get_dome_intersect(
    sample_radius: float, origin: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Dome ray intersection (ported from game.py)."""
    v_r0 = np.asarray(origin)
    v_a = np.asarray(direction)
    a2 = np.dot(v_a, v_a)
    t = np.sqrt(
        np.dot(v_r0, v_a) ** 2 / a2**2
        + sample_radius**2 / a2
        - np.dot(v_r0, v_r0) / a2
    ) - np.dot(v_r0, v_a) / a2
    return v_r0 + v_a * t


def preliminary_orbit(
    obs: List[dict],
    rng_seed: int | None = None,
) -> Orbit:
    """
    obs = list of three dicts:
        ra, dec (deg), time (ISO UTC string), location (EarthLocation or dict lat_deg, lon_deg, height_m)
    """
    rho_hat = []
    for ob in obs:
        rho_hat.append(ra_dec_to_unitvec(float(ob["ra"]), float(ob["dec"])))
    rho_hat = np.array(rho_hat)

    R = []
    with solar_system_ephemeris.set("builtin"):
        for ob in obs:
            t = Time(ob["time"], scale="utc")
            earth = get_body_barycentric("earth", t)
            eloc = _location_from_obj(ob["location"])
            obsvec = eloc.get_gcrs_posvel(t)[0].xyz.to(u.km).value
            Rvec = earth.xyz.to(u.km).value + obsvec
            R.append(Rvec)
    R = np.array(R)

    if rng_seed is not None:
        np.random.seed(int(rng_seed))

    tau1 = (Time(obs[0]["time"]).tdb.mjd - Time(obs[1]["time"]).tdb.mjd) * 86400.0
    tau3 = (Time(obs[2]["time"]).tdb.mjd - Time(obs[1]["time"]).tdb.mjd) * 86400.0
    drho_dt = (rho_hat[2] - rho_hat[0]) / (tau3 - tau1)
    r_mag = np.random.uniform(1.0, 3.0) * 1.496e8
    r = R[1] + r_mag * rho_hat[1]
    v = drho_dt * r_mag
    r = r * u.km
    v = v * (u.km / u.s)
    return Orbit.from_vectors(Sun, r, v, epoch=Time(obs[1]["time"]))


def smoke_scalar() -> float:
    """
    Cheap astropy-only check for embed smoke tests.
    Returns a value in [0, 1) derived from Sun RA at a fixed epoch.
    """
    t = Time("2024-06-15T12:00:00", scale="utc")
    ra = float(get_sun(t).ra.deg)
    return (ra % 360.0) / 360.0


def compute_orbit_json(json_in: str) -> str:
    """
    Input JSON:
      {"observations": [ {"ra", "dec", "time", "location": {lat_deg, lon_deg, height_m?}}, x3 ]}
    Output JSON:
      {"a_km", "ecc", "inc_deg", "period_s", "ok": true} or {"ok": false, "error": "..."}
    """
    try:
        data = json.loads(json_in)
        seed = data.get("seed")
        obs_raw = data["observations"]
        obs = []
        for o in obs_raw:
            obs.append(
                {
                    "ra": float(o["ra"]),
                    "dec": float(o["dec"]),
                    "time": str(o["time"]),
                    "location": o["location"],
                }
            )
        orb = preliminary_orbit(obs, rng_seed=int(seed) if seed is not None else None)
        out = {
            "ok": True,
            "a_km": float(orb.a.to(u.km).value),
            "ecc": float(orb.ecc.value),
            "inc_deg": float(orb.inc.to(u.deg).value),
            "period_s": float(orb.period.to(u.s).value),
        }
        return json.dumps(out)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)})


def frame_visual_for_c(t_day: float, latitude_deg: float = 52.0, sun_dec_deg: float = 15.0):
    """
    Single Python call from b1_host: sun unit vector + RGB derived from it (GLES cube tint
    and secondary object placement).
    Returns: (sx, sy, sz, r, g, b)
    """
    s = sun_direction(float(t_day), float(latitude_deg), float(sun_dec_deg))
    sx, sy, sz = float(s[0]), float(s[1]), float(s[2])
    r = 0.35 + 0.45 * (sx * 0.5 + 0.5)
    g = 0.3 + 0.5 * (sy * 0.5 + 0.5)
    b = 0.4 + 0.35 * (sz * 0.5 + 0.5)
    return (sx, sy, sz, r, g, b)


def sun_direction_json(json_in: str) -> str:
    """Input: {"t": float, "latitude_deg": float, "sun_declination_deg": float}"""
    try:
        d = json.loads(json_in)
        v = sun_direction(
            float(d["t"]),
            float(d["latitude_deg"]),
            float(d["sun_declination_deg"]),
        )
        return json.dumps({"ok": True, "x": float(v[0]), "y": float(v[1]), "z": float(v[2])})
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)})


def frame_angle_from_orbit(json_in: str) -> float:
    """
    Runs compute_orbit_json and maps result to [0,1) for driving visuals (deterministic from ecc).
    """
    s = compute_orbit_json(json_in)
    d = json.loads(s)
    if not d.get("ok"):
        return 0.0
    return (float(d["ecc"]) + float(d["inc_deg"]) / 180.0) * 0.5 % 1.0
