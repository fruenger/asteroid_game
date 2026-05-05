# asteroid_game — Python game package

Python layer for the **Asteroid** outreach build: **scenario generation**, **game state machine** (steps 1–8), **Ursina** presentation, and the **touch bridge** used by the native OpenGL ES host. The same `AsteroidGameState.tick()` drives both the desktop game and `asteroid_game_touch` when the embedded Python session is active.

---

## What this package does

1. **Observation narrative** — time window check, dome and telescope interaction, simulated Poisson imaging, asteroid pick, catalog form, final catalog table.
2. **Consistency** — Ursina (`game_app.py`) and the C++ host consume **one** implementation of steps, keys, and overlays (`paused_help`, toast, panel bytes).
3. **Localization** — `de` / `en` / `es` via `game_i18n` and `locales/*.yaml` (see [`locales/README.md`](locales/README.md)).

---

## Tech stack

| Area | Libraries / notes |
| ---- | ----------------- |
| Desktop runtime | **Ursina** (Panda3D) |
| Headless / native | **NumPy**, **SciPy** (imaging); optional **Astropy** + **Poliastro** (`orbit_api`, `observation_window`) |
| State & flow | `game_state.py` — no Ursina import |
| Scenario | `scenario.py` — procedural mock geometry and time window |

---

## Architecture (high level)

```text
game.py
  └─ game_boot  →  game_app.bootstrap()  →  Ursina update/input
                        │
                        ├─ GameRuntime / scene (game_scene.py)
                        └─ AsteroidGameState.tick()  ← single step engine
```

Native host (sibling directory `../native/asteroid_game_touch/`):

```text
touch_game_bridge.session_init()
  └─ tick(dt, wall_t, …)  →  AsteroidGameState.tick()  →  dict → GLES host
```

Aligning **telescope vs. target** in step 3 uses `rig_kinematics` in both worlds; the Ursina path can defer alignment to the render thread; the native path uses immediate optical-axis checks from the same math.

---

## Key modules

### Entry & lifecycle

| Module | Responsibility |
| ------ | -------------- |
| `game.py` | Entry: `game_boot`, Ursina hooks `update` / `input`, `game_app.run_forever()` |
| `game_app.py` | `GameRuntime`, `frame_update`, `handle_input`, infotext / panel wiring |
| `game_boot.py` | Faulthandler, early Panda3D `loadPrcFileData` |
| `game_globals.py` | Shared mutable flags (`game_gs`, `runtime`, pause, cheats) |

### Settings & camera

| Module | Responsibility |
| ------ | -------------- |
| `game_settings.py` | Env-driven fullscreen, shaders, debug; `apply_os_fullscreen_hint`, camera from env |
| `game_cam_env.py` | Default pitch / yaw conventions (parity with native docs) |

### Scene & UI (Ursina)

| Module | Responsibility |
| ------ | -------------- |
| `game_scene.py` | Telescope, dome, skybox, lights, transitions; `build_scene()` |
| `game_ui_*.py` | Shaders, on-screen text, help/control overlays |
| `game_stage_events.py` | Laser / marker / stage transitions |
| `game_catalog_step6.py` | Catalog-oriented UI helpers after imaging |

### Simulation & astronomy

| Module | Responsibility |
| ------ | -------------- |
| `game_state.py` | **`AsteroidGameState`** — phases, imaging, dome/telescope speeds, **`paused_help`**, outputs for overlay/panel |
| `scenario.py` | **`GameScenario`** — visibility window, object direction, orbit hints; Ursina-free |
| `orbit_api.py` | Compute API for orbit JSON, sun direction, dome ray tests; Astropy/Poliastro |
| `celestial_settings.py` | `GAME_CELESTIAL_DIURNAL_SIGN`, `GAME_CELESTIAL_HORIZ_OFFSET_DEG` (sky / sun parity with GLES shaders) |
| `game_synthetic_imaging.py` | Perfect-image stack fed into Poisson exposures |
| `rig_kinematics.py` | Mount math; **`GAME_RA_OFFSET_DEG`** (Ursina default 0°, native bridge often forces 180° for mesh alignment) |
| `touch_game_bridge.py` | **`tick`**, **`input_key`**, locale, panel grayscale bytes for GLES |

---

## Environment variables (representative)

Full lists live in source comments and in **`asteroid_game_touch`** usage strings. Common knobs:

| Variable | Role |
| -------- | ---- |
| `GAME_START_BEFORE_WINDOW` / `GAME_START_HOURS_BEFORE_WINDOW` | Start wall-clock-relative **before** the visibility window (default-like behavior in `game_state`) |
| `GAME_MAX_TARGET_ALTITUDE_DEG` / `GAME_MIN_TARGET_ALTITUDE_DEG` | Clamp target altitude for the scenario |
| `GAME_ALIGN_MAX_ANGLE_DEG` | Telescope–target alignment tolerance (step 3 → 4) |
| `GAME_RA_OFFSET_DEG` | Rig vs. Panda mesh offset (coordinate parity with native) |
| `GAME_CELESTIAL_*` | Diurnal sign and horizon azimuth tweak for sky/object consistency |
| `ASTRO_LANG` | Locale; mirrors Astro Mini Games when used under the launcher |
| `ASTEROID_GAME_STARTUP_HELP=0` | Skip one-shot help at launch (kiosk) |

---

## Testing

From **`asteroid_game/`**:

```bash
python3 -m pytest tests/ -q
# or
python3 -m unittest discover -s tests -v
```

Some tests need the **full venv** (Poliastro / `orbit_api`); `test_game_state.py` is designed to run with minimal deps.

---

## Related paths

| Path | Description |
| ---- | ----------- |
| [`../README.md`](../README.md) | Repository overview |
| [`../native/asteroid_game_touch/README.md`](../native/asteroid_game_touch/README.md) | Native GLES host; **[Astro launcher / `config.yaml`](../native/asteroid_game_touch/README.md#astro-mini-games-launcher-integration)** |
| [`assets/skybox/README.txt`](assets/skybox/README.txt) | Cubemap layout for optional sky |
| [`assets/fonts/`](assets/fonts/) | Optional bundled **DejaVu** + **Material Icons** for native UI |
