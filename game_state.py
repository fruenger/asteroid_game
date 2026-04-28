"""
Ursina-free game state and step logic for Asteroid (Phase A).

Drives the same phase flow as game.py using explicit time, key input, and optional
per-frame hints (telescope axis, dome ray) so a native host can supply geometry later.
"""

from __future__ import annotations

import os
import time
from dataclasses import InitVar, dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

import b1_kinematics as _b1k
from celestial_settings import apply_horizontal_yaw_y_up, diurnal_sign
from game_strings import TIME_WINDOW_REJECT_TOAST

# Pure NumPy copies of orbit_api.time_str / sun_direction — avoids importing poliastro here
# so Phase A tests run without the full ephemeris stack. Keep in sync with orbit_api.py.


def time_str(time_of_day: float) -> str:
    hour = int(np.floor((float(time_of_day) + 0.5) * 24.0) % 24)
    minute = int(np.floor(((float(time_of_day) + 0.5) * 24.0 * 60.0) % 60.0))
    return f"{hour:02d}:{minute:02d}"


def sun_direction(t: float, latitude: float, sun_declination: float) -> np.ndarray:
    """
    Local solar unit vector (Y up). Standard altitude:
    sin(alt) = sin φ sin δ + cos φ cos δ cos(H). Meridian azimuth in (x, z) is then flipped
    (negate x, z) so that after ``apply_horizontal_yaw_y_up`` the Sun matches the star field
    (southern culmination near noon on the northern mid-latitudes).
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

MAX_SPEED = 10.0
ACCEL = 20.0
DOME_RAY_CLEAR_DISTANCE = 100.0


def align_dot_threshold() -> float:
    """
    Minimum dot(object, optical_axis) to advance step 2 → 3.
    ``GAME_ALIGN_MAX_ANGLE_DEG`` — half-angle tolerance in degrees (default 0.5).
    """
    raw = os.environ.get("GAME_ALIGN_MAX_ANGLE_DEG", "").strip()
    try:
        deg = float(raw) if raw else 0.5
    except ValueError:
        deg = 0.5
    deg = max(0.05, min(deg, 30.0))
    return float(np.cos(np.radians(deg)))


def telescope_target_alignment_misalignment_deg(
    optical_world_unit: np.ndarray,
    object_dir_cartesian: np.ndarray,
) -> tuple[float, float]:
    """
    Returns (angular_separation_deg, dot_product) between normalized boresight and target.
    ``angular_separation_deg`` is arccos(dot) in degrees, in [0, 180].
    """
    u = np.asarray(optical_world_unit, dtype=np.float64).reshape(3)
    obj = np.asarray(object_dir_cartesian, dtype=np.float64).reshape(3)
    un = float(np.linalg.norm(u))
    on = float(np.linalg.norm(obj))
    if un < 1e-12 or on < 1e-12:
        return float("nan"), float("nan")
    u = u / un
    obj = obj / on
    dot = float(np.clip(np.dot(obj, u), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(dot)))
    return ang, dot


OBSERVATORY_OPEN_DURATION = 10.0
STAGE4_PAUSE_SEC = 5.0
EXPOSURE_COMMIT_SEC = 3.0
IMAGE_TICK_INTERVAL = 0.1
STEP5_BLINK_INTERVAL = 0.5


HELP_TEXTS: dict[int, str] = {
    0: (
        "In den Tiefen des Sonnensystems wurde ein noch unbekanntes kleines Objekt mit einem "
        "robotischen Teleskop gesichtet. Heute Nacht ist es unsere Aufgabe, dessen Nachverfolgung "
        "aufzunehmen. Dazu müssen wir zunächst warten, bis es über dem Horizont steht. Das ist heute "
        "Nacht zwischen %s und %s Uhr. Warte, bis wir im Zeitfenster sind, und drücke dann die Taste "
        "„Beobachtung beginnen”, um die Zeit zu pausieren."
    ),
    1: (
        "Nun muss die Sternwarte geöffnet und die Technik im Inneren hochgefahren werden. Das kann "
        "durch Betätigen der „Sternwarte öffnen“-Taste bewerkstelligt werden."
    ),
    2: (
        "Sobald die Sternwarte vollständig geöffnet ist, steuere das Teleskop an die vorgegebene "
        "Himmelsposition. Diese ist am Himmel mit einem kleinen roten Punkt markiert. Zur besseren "
        "Orientierung benutzen wir heute unseren Navigierlaser (blauer Strahl). Die Teleskopachsen "
        "steuerst du mit den Pfeiltasten. Das ist nun Deine Aufgabe!"
    ),
    3: (
        "Nun schaut das Teleskop auf den Himmelsbereich unserer Wahl. Allerdings muss auch die Kuppel "
        "korrekt ausgerichtet sein. Dazu muss der Kuppelspalt so ausgerichtet werden, dass wir durch "
        "das Teleskop freie Sicht auf den Himmel haben. Mit den Pfeiltasten kannst du die Kuppel "
        "drehen, damit wir mit den Beobachtungen beginnen können."
    ),
    4: (
        "Mache drei Bilder nacheinander und warte, bis das jeweilige Bild ein gutes "
        "Signal-Rausch-Verhältnis hat, sodass man rund 20 Sterne klar erkennen kann. Wenn du mit "
        "dem jeweiligen Bild zufrieden bist, drücke die Taste „Aufnahme stoppen”."
    ),
    5: (
        "Tippe auf das sich bewegende Objekt. Wenn du das Objekt auf deinen Bildern noch nicht "
        "erkennen kannst, kannst du mit „Von vorne starten” drei neue Aufnahmen anfertigen."
    ),
    6: (
        "Die Beobachtung war erfolgreich! Damit das neu entdeckte Objekt in unseren "
        "OST-Asteroidenkatalog aufgenommen werden kann, benötigt es noch einen Namen. Bitte trage "
        "auch das Entdeckerteam ein. Mit [Tab] springst du zwischen den Feldern. Wenn alles passt, "
        "speicherst du den Eintrag über die Taste „Speichern“."
    ),
    7: (
        "In der Tabelle sind alle bisher mit dem OST entdeckten Asteroiden aufgeführt. Die große "
        "Halbachse und die Umlaufzeit stammen von der jeweiligen Bahnberechnung. Mit „Zurück zur "
        "App-Übersicht“ beendest du deine Beobachtungssession."
    ),
}


def wrap_text(text: str, width: int) -> str:
    words = text.split()
    lines: list[str] = []
    line: list[str] = []
    for w in words:
        if sum(len(x) for x in line) + len(line) + len(w) <= width:
            line.append(w)
        else:
            lines.append(" ".join(line))
            line = [w]
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


@dataclass
class KeysHeld:
    space: bool = False
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    r: bool = False
    dome_ccw: bool = False  # e.g. [ — Kuppel Azimut (Schritt 3)
    dome_cw: bool = False  # e.g. ]


@dataclass
class KeysInput:
    """Per frame: held keys (continuous). Discrete keys use handle_discrete_input."""

    held: KeysHeld = field(default_factory=KeysHeld)


@dataclass
class FrameHints:
    """
    Optional geometry from the renderer (Ursina today, GLES host later).
    Step 2 → 3: if ``defer_telescope_target_alignment`` is True (Ursina), the host must call
    ``apply_telescope_target_alignment`` with the rendered optical axis after scene joints update.
    If False (B1), alignment uses ``b1_kinematics`` inside ``tick`` after RA/Dec integration.
    If dome_ray_exit_distance is None during step 3, automatic advance to step 4 is disabled
    (unless cheat_through).
    """

    optical_axis_world_unit: Optional[np.ndarray] = None
    defer_telescope_target_alignment: bool = False
    dome_ray_exit_distance: Optional[float] = None
    step5_pick_pixel_xy: Optional[tuple[float, float]] = None


def _build_star_field_and_trajectory(
    imsize: int, n_stars: int, rng: np.random.Generator
) -> tuple[list[np.ndarray], np.ndarray]:
    traj_angle = float(rng.uniform(0, 2.0 * np.pi))
    image_locations = np.array(
        [[np.cos(traj_angle), np.sin(traj_angle)], [-np.cos(traj_angle), -np.sin(traj_angle)]]
    )
    image_locations = np.insert(image_locations, 1, np.mean(image_locations, axis=1), axis=1)
    image_locations *= rng.uniform(4.0, 20.0)
    image_locations += np.expand_dims(rng.uniform(imsize / 4.0, imsize * 3.0 / 4.0, 2), axis=1)
    image_locations = image_locations.astype(int)

    image_perfect_bg = np.ones((imsize, imsize), dtype=float) / 50.0
    image_perfect_bg[rng.integers(0, imsize, n_stars), rng.integers(0, imsize, n_stars)] = (
        rng.exponential(0.5, n_stars)
    )

    perfect_list: list[np.ndarray] = []
    for x, y in image_locations.T:
        image_perfect = image_perfect_bg.copy()
        image_perfect[x, y] = 1.0
        image_perfect = gaussian_filter(image_perfect, sigma=1.5)
        perfect_list.append(image_perfect)

    return perfect_list, image_locations


@dataclass
class AsteroidGameState:
    min_time: float
    max_time: float
    object_dir_cartesian: np.ndarray
    sun_dec_deg: float = 0.5
    latitude_deg: float = 52.0
    imsize: int = 250
    cheat_through: bool = False

    step: int = 0
    time_now: float = 0.0
    time_stopped: bool = False
    _time_wall_anchor: Optional[float] = field(default=None, init=False, repr=False)
    _time_now_at_anchor: float = field(default=0.0, init=False, repr=False)
    _start_before_window_enabled: bool = field(default=True, init=False, repr=False)
    _start_before_window_hours: float = field(default=2.5, init=False, repr=False)
    paused_help: bool = False
    # none | help | controls — which overlay b1_host shows when paused_help
    help_overlay_kind: str = "none"
    _help_pause_start_wall_t: Optional[float] = field(default=None, init=False, repr=False)
    """Subtract from wall_t in _advance_time so celestial time does not run while paused_help."""
    _help_wall_slip_accum: float = field(default=0.0, init=False, repr=False)

    toast_message: str = ""
    toast_until_wall_t: float = 0.0

    step6_object_name: str = ""
    step6_discoverer: str = ""
    step6_focus_idx: int = 0
    step6_orbital_text_block: str = ""
    """Semi-major axis (AU) and period (yr) for the current scenario — stored with the catalog line."""
    step6_orbit_a_au: float = 0.0
    step6_orbit_p_yr: float = 0.0
    step6_catalog_ready: bool = False

    ra_deg: float = 0.0
    dec_deg: float = 0.0
    dome_az_deg: float = 0.0
    telescope_speed_ra: float = 0.0
    telescope_speed_dec: float = 0.0
    dome_speed_az: float = 0.0

    observatory_open_t: float = 0.0
    observatory_fully_open: bool = False
    laser_and_marker_ready: bool = False

    stage4_panel_on_wall_t: float = 0.0
    _stage4_exposure_unlock_wall_t: float = 0.0
    image_panel_enabled: bool = False
    step4_exposure_active: bool = field(default=False, init=False, repr=False)
    _step4_space_was_held: bool = field(default=False, init=False, repr=False)

    exposure_time: float = 0.0
    last_image_tick_wall_t: float = 0.0
    last_step5_blink_wall_t: float = 0.0
    all_images: list[np.ndarray] = field(default_factory=list)
    image_shown: int = 0

    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    perfect_stack: InitVar[Optional[list[np.ndarray]]] = None
    locations_stack: InitVar[Optional[np.ndarray]] = None

    image_array: np.ndarray = field(init=False)
    all_images_perfect: list[np.ndarray] = field(init=False)
    image_locations: np.ndarray = field(init=False)

    def __post_init__(
        self,
        perfect_stack: Optional[list[np.ndarray]],
        locations_stack: Optional[np.ndarray],
    ) -> None:
        od = np.asarray(self.object_dir_cartesian, dtype=np.float64).reshape(3)
        n = np.linalg.norm(od)
        self.object_dir_cartesian = od / (n if n > 1e-12 else 1.0)
        self.image_array = np.zeros((self.imsize, self.imsize), dtype=float)
        if perfect_stack is not None and locations_stack is not None:
            self.all_images_perfect = [np.asarray(x, dtype=np.float64).copy() for x in perfect_stack]
            self.image_locations = np.asarray(locations_stack, dtype=np.int64).copy()
        else:
            self.all_images_perfect, self.image_locations = _build_star_field_and_trajectory(
                self.imsize, 100, self.rng
            )
        st = os.environ.get("GAME_START_BEFORE_WINDOW", "1").strip().lower()
        self._start_before_window_enabled = st in ("1", "true", "yes")
        try:
            self._start_before_window_hours = float(os.environ.get("GAME_START_HOURS_BEFORE_WINDOW", "2.5"))
        except ValueError:
            self._start_before_window_hours = 2.5

    def _apply_friction(self, speed: float, dt: float) -> float:
        return speed - (ACCEL / 2.0) * dt * np.sign(speed)

    def _advance_time(self, wall_t: float) -> None:
        if self.time_stopped:
            return
        rate = 1.0 / 100.0
        wt = float(wall_t) - float(self._help_wall_slip_accum)
        if not self._start_before_window_enabled:
            self.time_now = (wt * rate) % 1.0
            return
        if self._time_wall_anchor is None:
            self._time_wall_anchor = wt
            self._time_now_at_anchor = (float(self.min_time) - self._start_before_window_hours / 24.0) % 1.0
        self.time_now = (self._time_now_at_anchor + (wt - self._time_wall_anchor) * rate) % 1.0

    def infotext_message(self) -> str:
        if self.step == 0:
            return (
                "Das Beobachtungsfenster ist zwischen %s und %s Uhr heute Nacht. Warte und "
                "druecke [LEERTASTE], um die Beobachtung zu beginnen."
                % (time_str(self.min_time), time_str(self.max_time))
            )
        if self.step == 1:
            return "Öffne das Observatorium mit [Z]."
        if self.step == 2:
            return "Richte das Teleskop auf das Ziel aus. Kuppel bei Bedarf mit [ und ]."
        if self.step == 3:
            return "Richte die Kuppel mit [←][→] oder [ und ] auf das Objekt aus."
        if self.step == 4:
            return "Die Kamera belichtet ... Drücke [LEERTASTE], um ein Bild zu machen"
        if self.step == 5:
            return (
                "Wähle das sich bewegende Objekt mit der [linken Maustaste] aus. (Mit [R] kannst "
                "du auf die drei Aufnahmen noch einmal von vorn starten falls du auf deinen "
                "Bildern das Objekt noch nicht erkennen kannst.)"
            )
        if self.step == 6:
            return (
                "Trage rechts Objektname und Entdeckerteam ein — [Tab] wechselt das Feld, [Enter] "
                "speichert. Die ausführliche Anleitung steht links in der Hilfebox."
            )
        if self.step == 7:
            return (
                "Katalogübersicht — [Enter] beendet die Anwendung. Kurzhilfe: Taste [H]."
            )
        return ""

    def help_body_text(self) -> str:
        if self.step == 0:
            return HELP_TEXTS[0] % (time_str(self.min_time), time_str(self.max_time))
        return HELP_TEXTS.get(self.step, "")

    def sun_unit_vector(self) -> np.ndarray:
        return sun_direction(self.time_now, self.latitude_deg, self.sun_dec_deg)

    def stage4_can_expose(self) -> bool:
        """True when step 4 imaging logic should run (after intro pause)."""
        return self.step == 4 and self._stage4_exposure_unlock_wall_t == 0.0

    def _reset_stage4_imaging(self, wall_t: float) -> None:
        """Clear partial or completed captures; stay on step 4 with a fresh stack."""
        self.all_images = []
        self.exposure_time = 0.0
        self.image_array = np.zeros((self.imsize, self.imsize), dtype=float)
        self.last_image_tick_wall_t = float(wall_t)
        self._stage4_exposure_unlock_wall_t = 0.0
        self.image_panel_enabled = True
        self.step4_exposure_active = False
        self._step4_space_was_held = False

    def step4_primary_button_label(self) -> str:
        if self.step != 4:
            return ""
        if self._stage4_exposure_unlock_wall_t > 0.0:
            return "Aufnahme starten"
        if self.step4_exposure_active:
            if len(self.all_images) == 2:
                return "Letzte Aufnahme speichern"
            return "Aufnahme speichern"
        if len(self.all_images) == 0:
            return "Aufnahme starten"
        return "Nächste Aufnahme starten"

    def try_step5_pick(self, pixel_xy: tuple[float, float]) -> bool:
        """
        ``pixel_xy`` is ``(row, col)`` in image index space (first axis vertical, second horizontal),
        same as ``image_locations`` and ``game_app`` after the Ursina local transform.
        Returns True if the pick hits the asteroid (advances to step 6).
        """
        if self.step != 5 or len(self.all_images) != 3:
            return False
        local_point = np.array(pixel_xy, dtype=np.float64)
        dist = float(
            np.min(
                np.sqrt(
                    np.sum(
                        (np.expand_dims(local_point, axis=1) - self.image_locations) ** 2,
                        axis=0,
                    )
                )
            )
        )
        if dist / float(self.imsize) < 0.05:
            self.step = 6
            self.image_panel_enabled = True
            self._reset_step6_form()
            return True
        return False

    def _reset_step6_form(self) -> None:
        from game_catalog_common import default_object_name

        self.step6_object_name = default_object_name()
        self.step6_discoverer = "Name d. Entdeckerteams"
        self.step6_focus_idx = 0
        self.step6_catalog_ready = True

    def step6_key_action(self, kind: str, text: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {"handled": False, "process_exit": False}
        if self.step != 6:
            return out
        max_len = 30
        if kind == "char" and text:
            for ch in text:
                if ch in ("\r", "\n", "\t"):
                    continue
                if ord(ch) < 32:
                    continue
                if self.step6_focus_idx == 0:
                    if len(self.step6_object_name) < max_len:
                        self.step6_object_name += ch
                elif self.step6_focus_idx == 1:
                    if len(self.step6_discoverer) < max_len:
                        self.step6_discoverer += ch
            out["handled"] = True
            return out
        if kind == "backspace":
            if self.step6_focus_idx == 0 and self.step6_object_name:
                self.step6_object_name = self.step6_object_name[:-1]
            elif self.step6_focus_idx == 1 and self.step6_discoverer:
                self.step6_discoverer = self.step6_discoverer[:-1]
            out["handled"] = True
            return out
        if kind == "tab":
            self.step6_focus_idx = (int(self.step6_focus_idx) + 1) % 2
            out["handled"] = True
            return out
        if kind == "save":
            from game_catalog_common import append_catalog_entry

            a = float(self.step6_orbit_a_au)
            p = float(self.step6_orbit_p_yr)
            if a > 0.0 and p > 0.0:
                append_catalog_entry(self.step6_object_name, self.step6_discoverer, a_au=a, period_yr=p)
            else:
                append_catalog_entry(self.step6_object_name, self.step6_discoverer)
            self.step = 7
            self.image_panel_enabled = False
            out["handled"] = True
            out["process_exit"] = False
            return out
        return out

    def step7_key_action(self, kind: str) -> dict[str, Any]:
        out: dict[str, Any] = {"handled": False, "process_exit": False}
        if self.step != 7:
            return out
        if kind in ("finish", "enter", "return"):
            out["handled"] = True
            out["process_exit"] = True
        return out

    def apply_telescope_target_alignment(self, optical_world_unit: np.ndarray) -> list[str]:
        """
        Ursina: call once per frame after ``ra_pivot`` / ``dec_pivot`` match ``ra_deg`` / ``dec_deg``.
        Uses the same world-space axis as the drawn laser (Panda), not ``b1_kinematics`` alone.
        """
        if self.step != 2 or self.paused_help:
            return []
        _ang, dot = telescope_target_alignment_misalignment_deg(
            optical_world_unit, self.object_dir_cartesian
        )
        if np.isnan(dot):
            return []
        if dot > align_dot_threshold():
            self.step = 3
            return ["telescope_aligned"]
        return []

    def tick(
        self,
        dt: float,
        wall_t: float,
        keys: KeysInput,
        hints: Optional[FrameHints] = None,
    ) -> dict[str, Any]:
        """
        One simulation step. `wall_t` should match `time.time()` in production so
        `(wall_t/100)%1` matches game.py's day parameter.
        """
        hints = hints or FrameHints()
        out: dict[str, Any] = {"events": []}

        if self.toast_until_wall_t > 0.0 and float(wall_t) >= self.toast_until_wall_t:
            self.toast_message = ""
            self.toast_until_wall_t = 0.0
        toast_out = ""
        if self.toast_message and float(wall_t) < self.toast_until_wall_t:
            toast_out = self.toast_message

        if self.paused_help:
            if toast_out:
                out["toast"] = toast_out
            return out

        if self.step == 4 and self._stage4_exposure_unlock_wall_t > 0.0:
            if wall_t < self._stage4_exposure_unlock_wall_t:
                out["time_display"] = time_str(self.time_now)
                out["infotext"] = self.infotext_message()
                out["sun_direction"] = self.sun_unit_vector()
                out["step4_primary_button_label"] = self.step4_primary_button_label()
                if toast_out:
                    out["toast"] = toast_out
                return out
            self._stage4_exposure_unlock_wall_t = 0.0

        self._advance_time(wall_t)

        if self.step == 2:
            h = keys.held
            self.telescope_speed_ra = float(
                np.clip(
                    self.telescope_speed_ra
                    + ACCEL * dt * float(h.right)
                    - ACCEL * dt * float(h.left),
                    -MAX_SPEED,
                    MAX_SPEED,
                )
            )
            self.telescope_speed_ra = self._apply_friction(self.telescope_speed_ra, dt)
            self.ra_deg += self.telescope_speed_ra * dt

            self.telescope_speed_dec = float(
                np.clip(
                    self.telescope_speed_dec
                    + ACCEL * dt * float(h.up)
                    - ACCEL * dt * float(h.down),
                    -MAX_SPEED,
                    MAX_SPEED,
                )
            )
            self.telescope_speed_dec = self._apply_friction(self.telescope_speed_dec, dt)
            self.dec_deg += self.telescope_speed_dec * dt

            h2 = keys.held
            dome_l2 = float(h2.dome_ccw)
            dome_r2 = float(h2.dome_cw)
            self.dome_speed_az = float(
                np.clip(
                    self.dome_speed_az
                    + ACCEL * dt * dome_l2
                    - ACCEL * dt * dome_r2,
                    -MAX_SPEED,
                    MAX_SPEED,
                )
            )
            self.dome_speed_az = self._apply_friction(self.dome_speed_az, dt)
            self.dome_az_deg += self.dome_speed_az * dt

        if self.step == 3:
            h = keys.held
            dome_l = float(h.left or h.dome_ccw)
            dome_r = float(h.right or h.dome_cw)
            self.dome_speed_az = float(
                np.clip(
                    self.dome_speed_az
                    + ACCEL * dt * dome_l
                    - ACCEL * dt * dome_r,
                    -MAX_SPEED,
                    MAX_SPEED,
                )
            )
            self.dome_speed_az = self._apply_friction(self.dome_speed_az, dt)
            self.dome_az_deg += self.dome_speed_az * dt

        if self.step == 2:
            self.observatory_open_t += dt
            if self.observatory_open_t >= OBSERVATORY_OPEN_DURATION:
                self.observatory_fully_open = True
            if self.observatory_open_t >= OBSERVATORY_OPEN_DURATION and not self.laser_and_marker_ready:
                self.laser_and_marker_ready = True
                out["events"].append("laser_and_marker_ready")

            if not hints.defer_telescope_target_alignment:
                u = np.asarray(_b1k.optical_axis_world_unit(self.ra_deg, self.dec_deg), dtype=np.float64).reshape(3)
                un = float(np.linalg.norm(u))
                if un > 1e-12:
                    u = u / un
                    obj = np.asarray(self.object_dir_cartesian, dtype=np.float64).reshape(3)
                    on = float(np.linalg.norm(obj))
                    if on > 1e-12:
                        obj = obj / on
                    if float(np.dot(obj, u)) > align_dot_threshold():
                        self.step = 3
                        out["events"].append("telescope_aligned")

        if self.step == 3:
            dist = hints.dome_ray_exit_distance
            if self.cheat_through or (dist is not None and dist > DOME_RAY_CLEAR_DISTANCE):
                self.step = 4
                self._stage4_exposure_unlock_wall_t = wall_t + STAGE4_PAUSE_SEC
                self.stage4_panel_on_wall_t = wall_t
                self.image_panel_enabled = True
                self.exposure_time = 0.0
                self.last_image_tick_wall_t = wall_t
                self.step4_exposure_active = False
                self._step4_space_was_held = False
                out["events"].append("entered_stage4")

        if self.step == 4 and self._stage4_exposure_unlock_wall_t == 0.0:
            if keys.held.r:
                self._reset_stage4_imaging(wall_t)
                out["events"].append("stage4_imaging_reset")
            else:
                space_now = bool(keys.held.space)
                edge = space_now and not self._step4_space_was_held
                if edge and (not self.step4_exposure_active) and len(self.all_images) < 3:
                    self.step4_exposure_active = True
                    self.exposure_time = 0.0
                    self.last_image_tick_wall_t = wall_t

                if self.step4_exposure_active:
                    self.exposure_time += dt
                    if wall_t - self.last_image_tick_wall_t > IMAGE_TICK_INTERVAL:
                        if len(self.all_images) < len(self.all_images_perfect):
                            image_perfect = self.all_images_perfect[len(self.all_images)]
                            lam = self.rng.poisson(image_perfect * 1.0, size=(self.imsize, self.imsize))
                            self.image_array = self.image_array + lam.astype(float)
                        self.last_image_tick_wall_t = wall_t

                    if keys.held.space and self.exposure_time >= EXPOSURE_COMMIT_SEC:
                        self.all_images.append(self.image_array.copy())
                        self.image_array = np.zeros((self.imsize, self.imsize))
                        self.last_image_tick_wall_t = wall_t
                        self.exposure_time = 0.0
                        self.step4_exposure_active = False
                        if len(self.all_images) == 3:
                            self.step = 5
                            self.last_step5_blink_wall_t = wall_t
                            self.image_panel_enabled = True
                            out["events"].append("imaging_complete")

                self._step4_space_was_held = space_now

        if self.step == 5:
            if keys.held.r:
                self._reset_stage4_imaging(wall_t)
                self.step = 4
                self.last_step5_blink_wall_t = wall_t
                out["events"].append("imaging_restarted")
            elif wall_t - self.last_step5_blink_wall_t > STEP5_BLINK_INTERVAL and self.all_images:
                self.image_shown = (self.image_shown + 1) % 3
                self.last_step5_blink_wall_t = wall_t

            pix = hints.step5_pick_pixel_xy
            if pix is not None and len(self.all_images) == 3:
                local_point = np.array(pix, dtype=float)
                dist = float(
                    np.min(
                        np.sqrt(
                            np.sum(
                                (np.expand_dims(local_point, axis=1) - self.image_locations) ** 2,
                                axis=0,
                            )
                        )
                    )
                )
                if dist / self.imsize < 0.05:
                    self.step = 6
                    self.image_panel_enabled = True
                    self._reset_step6_form()
                    out["events"].append("asteroid_picked")

        out["time_display"] = time_str(self.time_now)
        out["infotext"] = self.infotext_message()
        out["sun_direction"] = self.sun_unit_vector()
        if toast_out:
            out["toast"] = toast_out
        if self.step == 4:
            out["step4_primary_button_label"] = self.step4_primary_button_label()
        return out

    def handle_discrete_input(self, key: str, wall_t: float) -> dict[str, Any]:
        """Matches game.py `input(key)` for non-held actions."""
        out: dict[str, Any] = {"events": [], "handled": False}

        if key == "space" and self.step == 0:
            self._advance_time(wall_t)
            if self.min_time < self.time_now < self.max_time:
                self.time_stopped = True
                self.step = 1
                out["events"].append("time_window_ok")
            else:
                self.toast_message = TIME_WINDOW_REJECT_TOAST
                self.toast_until_wall_t = float(wall_t) + 2.5
                out["events"].append("time_window_reject")
            out["handled"] = True
            return out

        if key == "z" and self.step == 1:
            self.step = 2
            self.observatory_open_t = 0.0
            self.observatory_fully_open = False
            self.laser_and_marker_ready = False
            out["events"].append("observatory_opening_started")
            out["handled"] = True
            return out

        if key == "h":
            self.paused_help = True
            self.help_overlay_kind = "help"
            self._help_pause_start_wall_t = float(wall_t)
            out["events"].append("open_help")
            out["handled"] = True
            return out

        if key == "f1":
            self.paused_help = True
            self.help_overlay_kind = "controls"
            self._help_pause_start_wall_t = float(wall_t)
            out["events"].append("open_controls")
            out["handled"] = True
            return out

        return out

    def _shift_wall_deadlines(self, delta: float) -> None:
        """Move wall-clock deadlines forward by ``delta`` (seconds overlay was open)."""
        if delta <= 0.0:
            return
        if self.toast_until_wall_t > 0.0:
            self.toast_until_wall_t += delta
        if self._stage4_exposure_unlock_wall_t > 0.0:
            self._stage4_exposure_unlock_wall_t += delta
        if self.stage4_panel_on_wall_t > 0.0:
            self.stage4_panel_on_wall_t += delta
        if self.last_image_tick_wall_t > 0.0:
            self.last_image_tick_wall_t += delta
        if self.last_step5_blink_wall_t > 0.0:
            self.last_step5_blink_wall_t += delta

    def close_help_overlay(self, wall_t: Optional[float] = None) -> None:
        if not self.paused_help:
            return
        wt = float(wall_t) if wall_t is not None else time.time()
        ps = self._help_pause_start_wall_t
        if ps is not None:
            slip = wt - float(ps)
            if slip > 0.0:
                self._help_wall_slip_accum += slip
                self._shift_wall_deadlines(slip)
        self.paused_help = False
        self.help_overlay_kind = "none"
        self._help_pause_start_wall_t = None
