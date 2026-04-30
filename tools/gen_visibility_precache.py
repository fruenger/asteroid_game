#!/usr/bin/env python3
"""
Build ``visibility_precache.json``: for each session seed, night visibility window (MJD open/close).

Run from ``asteroid_game`` (this package root) so imports resolve, e.g.:

  ./venv/bin/python tools/gen_visibility_precache.py --reference-date 2026-04-08 --seeds 0-511 --out assets/visibility_precache.json

Flags: ``-v`` / ``--verbose`` per-seed resolution logs; ``-q`` / ``--quiet`` no banner/progress bar.

**asteroid_game_touch session seed:** the native host does not pass a seed. ``generate_game_scenario`` uses the
environment variable ``GAME_SESSION_SEED`` if set before launch; otherwise a random integer. Example:
``export GAME_SESSION_SEED=42`` then ``./build/asteroid_game_touch --venv ... --asteroid-root ...``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import astropy.units as u
from astropy.coordinates import EarthLocation

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from observation_window import compute_night_visibility_window_detailed  # noqa: E402
from orbit_catalog import classical_orbit_from_seed  # noqa: E402

_LOG_PREFIX = "[gen_visibility_precache]"


def _log(msg: str, *, file=sys.stderr) -> None:
    print(f"{_LOG_PREFIX} {msg}", file=file, flush=True)


def _progress_line(current: int, total: int, seed: int, width: int = 36) -> str:
    """Single-line progress bar (0-based current index, inclusive end)."""
    if total <= 0:
        return ""
    done = current + 1
    frac = done / total
    n = int(round(width * frac))
    n = min(width, max(0, n))
    bar = "#" * n + "·" * (width - n)
    return f"[{bar}] {done}/{total} ({100.0 * frac:.1f}%)  seed={seed}"


def _parse_seeds(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        lo, hi = int(a.strip()), int(b.strip())
        return list(range(lo, hi + 1))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate visibility_precache.json (version 2: ICRS barycentric asteroid + AltAz grid)",
        epilog=(
            "Window logic matches runtime: classical_orbit_from_seed(seed, ref_date) then "
            "compute_night_visibility_window_detailed (grid from ref 12:00 UTC + 24h; if no mask matches, "
            "synthetic 1h band at ~22h LMT). Only one code path — no precache lookup here."
        ),
    )
    ap.add_argument("--reference-date", required=True, help="YYYY-MM-DD (must match session ephemeris anchor)")
    ap.add_argument("--seeds", default="0-511", help='Range "0-511" or comma list')
    ap.add_argument("--lat-deg", type=float, default=52.0)
    ap.add_argument("--lon-deg", type=float, default=13.0)
    ap.add_argument("--height-m", type=float, default=0.0)
    ap.add_argument("--h-min-deg", type=float, default=10.0)
    ap.add_argument("--step-minutes", type=float, default=10.0)
    ap.add_argument("--min-span-hours", type=float, default=1.0)
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if any seed fails min span")
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log each seed: catalog index, grid vs synthetic fallback, sun cap, duration vs min_span",
    )
    ap.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress bar and startup banner (errors and final path still printed)",
    )
    args = ap.parse_args()

    ref = date.fromisoformat(args.reference_date)
    loc = EarthLocation(
        lon=float(args.lon_deg) * u.deg,
        lat=float(args.lat_deg) * u.deg,
        height=float(args.height_m) * u.m,
    )
    seeds = _parse_seeds(args.seeds)
    entries: list[dict] = []
    failures: list[int] = []

    if not args.quiet:
        _log(
            f"start reference_date={ref.isoformat()} n_seeds={len(seeds)} "
            f"location=lat={args.lat_deg:g}° lon={args.lon_deg:g}° height={args.height_m:g}m "
            f"h_min={args.h_min_deg:g}° step={args.step_minutes:g}min min_span={args.min_span_hours:g}h "
            f"strict={args.strict} out={args.out}"
        )
        _log(
            "path: for each seed → classical_orbit_from_seed (epoch ref 12:00 UTC) → "
            "compute_night_visibility_window_detailed (same as asteroid_game at runtime when not using precache)"
        )

    n = len(seeds)
    for i, seed in enumerate(seeds):
        if not args.quiet and not args.verbose:
            line = _progress_line(i, n, int(seed))
            sys.stderr.write("\r" + line + "   ")
            sys.stderr.flush()

        orb, cat_idx = classical_orbit_from_seed(int(seed), ref)
        t_open, t_close, sun_cap_used, synth = compute_night_visibility_window_detailed(
            orb,
            ref,
            loc,
            h_min_deg=float(args.h_min_deg),
            step_minutes=float(args.step_minutes),
        )
        dur_h = float((t_close - t_open).to(u.h).value)
        span_ok = dur_h >= float(args.min_span_hours)

        if synth:
            mode = "synthetic_LMT_band"
            why = (
                "no grid cell satisfied asteroid_alt>=h_min AND sun_alt<=cap for caps -12°,-3°,0° "
                "(see observation_window.compute_night_visibility_window_detailed)"
            )
        else:
            mode = "grid_longest_run"
            why = (
                f"longest contiguous span on {args.step_minutes:g}min grid from {ref.isoformat()} 12:00 UTC "
                f"+24h; sun_cap_used={float(sun_cap_used):g}°"
            )

        if args.verbose:
            cap_s = "nan" if math.isnan(float(sun_cap_used)) else f"{float(sun_cap_used):g}"
            _log(
                f"seed={int(seed)} catalog_index={int(cat_idx)} mode={mode} | {why} | "
                f"dur_h={dur_h:.4f} min_span_h={float(args.min_span_hours):g} span_ok={span_ok} "
                f"t_open_mjd={float(t_open.mjd):.8f} t_close_mjd={float(t_close.mjd):.8f} cap={cap_s}"
            )

        if not span_ok:
            failures.append(int(seed))
        cap_json: float | None
        if isinstance(sun_cap_used, float) and math.isnan(sun_cap_used):
            cap_json = None
        else:
            cap_json = float(sun_cap_used)
        entry = {
            "session_seed": int(seed),
            "visibility_t_open_mjd": float(t_open.mjd),
            "visibility_t_close_mjd": float(t_close.mjd),
            "sun_cap_used": cap_json,
        }
        if not span_ok:
            entry["below_min_span_hours"] = True
        entries.append(entry)

    if not args.quiet:
        sys.stderr.write("\n")
        sys.stderr.flush()

    if args.strict and failures:
        _log(
            f"strict FAIL: {len(failures)} seeds below min_span_hours={args.min_span_hours:g}: "
            f"{failures[:40]!r}{'...' if len(failures) > 40 else ''}"
        )
        return 2

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": 2,
        "reference_date": ref.isoformat(),
        "min_span_hours": float(args.min_span_hours),
        "entries": entries,
    }
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    below = len(failures)
    _log(
        f"done wrote {len(entries)} entries to {out_path} "
        f"(below_min_span={below}; strict was {'enforced' if args.strict else 'off'})",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
