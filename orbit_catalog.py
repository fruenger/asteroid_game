"""Deterministic main-belt-style classical orbits: catalog index + on-the-fly sample."""

from __future__ import annotations

import datetime as dt
import hashlib
import zlib

import numpy as np
import astropy.units as u
from astropy.time import Time
from poliastro.bodies import Sun
from poliastro.twobody import Orbit

AU_KM = 149597870.7
N_CATALOG = 512


def _elements_from_index(idx: int) -> tuple[float, float, float, float, float]:
    rng = np.random.default_rng(42_000 + int(idx) % N_CATALOG)
    a_au = float(rng.uniform(1.85, 3.45))
    ecc = float(rng.uniform(0.02, 0.28))
    inc = float(rng.uniform(0.5, 18.0))
    raan = float(rng.uniform(0.0, 360.0))
    argp = float(rng.uniform(0.0, 360.0))
    return a_au, ecc, inc, raan, argp


def catalog_index_for_seed(session_seed: int) -> int:
    h = hashlib.sha256(str(int(session_seed)).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little", signed=False) % N_CATALOG


def classical_orbit_from_seed(session_seed: int, ref_date: dt.date) -> tuple[Orbit, int]:
    """
    Heliocentric two-body orbit (Sun). Returns ``(orbit, catalog_index)``.
    True anomaly from ``session_seed`` (distinct from geometry index).
    """
    idx = catalog_index_for_seed(session_seed)
    a_au, ecc, inc, raan, argp = _elements_from_index(idx)
    nu_rng = np.random.default_rng(zlib.adler32(str(int(session_seed)).encode("utf-8")) & 0xFFFFFFFF)
    nu_deg = float(nu_rng.uniform(0.0, 360.0))
    epoch = Time(f"{ref_date.isoformat()} 12:00:00", scale="utc")
    a_km = a_au * AU_KM
    orb = Orbit.from_classical(
        Sun,
        a_km * u.km,
        ecc * u.one,
        inc * u.deg,
        raan * u.deg,
        argp * u.deg,
        nu_deg * u.deg,
        epoch=epoch,
    )
    return orb, idx


def sample_classical_orbit(ref_date: dt.date, rng: np.random.Generator) -> Orbit:
    """Fallback when no session seed is desired: random main-belt-like orbit."""
    a_au = float(rng.uniform(1.85, 3.45))
    ecc = float(rng.uniform(0.02, 0.28))
    inc = float(rng.uniform(0.5, 18.0))
    raan = float(rng.uniform(0.0, 360.0))
    argp = float(rng.uniform(0.0, 360.0))
    nu_deg = float(rng.uniform(0.0, 360.0))
    epoch = Time(f"{ref_date.isoformat()} 12:00:00", scale="utc")
    a_km = a_au * AU_KM
    return Orbit.from_classical(
        Sun,
        a_km * u.km,
        ecc * u.one,
        inc * u.deg,
        raan * u.deg,
        argp * u.deg,
        nu_deg * u.deg,
        epoch=epoch,
    )
