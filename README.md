# asteroid_game

A tiny asteroid-finding game for public outreach (Ursina / Panda3D).

- **`game.py`** — entry only: **`game_boot`**, then **`game_app.bootstrap()`**, Ursina hooks **`update` / `input`**, **`game_app.run_forever()`**.
- **`game_app.py`** — Ursina window, **`GameRuntime`** (scenario, scene, UI refs), **`frame_update`** / **`handle_input`**.
- **`game_boot.py`** — faulthandler + shadow **`loadPrcFileData`** before Panda3D.
- **`game_cam_env.py`** — **`GAME_CAM_DEFAULT_INIT_PITCH_DEG`** (**-30**) and **`initial_camera_yaw_pitch_deg`** (no Ursina; tests / B1 docs parity).
- **`game_settings.py`** — env flags (shadows, fullscreen, borderless, debug), scene shader choice, **`apply_os_fullscreen_hint`**, **`apply_initial_editor_camera_from_env`** (uses **`game_cam_env`**; Ursina **`EditorCamera`** after bootstrap).
- **`game_globals.py`** — shared mutable state (**`game_paused`**, **`game_gs`**, cheat/laser flags, **`runtime`**).
- **`game_strings.py`** — static **`STEUERUNG_TEXT`**.
- **`game_ui_shaders.py`** — **`load_shader`** (GLSL files).
- **`game_ui_onscreen.py`** — **`OnScreenMessage`**, **`blink_opacity`**.
- **`game_ui_help.py`** — help / Steuerung **`TextField`** overlays and fade helpers.
- **`game_scene.py`** — telescope, dome, skybox, lights, **`TransitionMask`**; **`build_scene()`** → **`SceneEntities`**.
- **`game_synthetic_imaging.py`** — procedural perfect-image stack for **`AsteroidGameState`**.
- **`game_stage_events.py`** — **`make_stageup_event`** (laser/marker, stage-4 mask).
- **`game_catalog_step6.py`** — catalog UI after a successful step-5 pick.
- Step flow and imaging are driven by **`game_state.AsteroidGameState`**. **`get_dome_intersect`** / **`time_str`** from **`orbit_api`**; **`wrap_text`** from **`game_state`**.
- **`scenario.py`** — procedural **`GameScenario`** (mock asteroid direction, **`preliminary_orbit`**, time window); no Ursina import. Env **`GAME_MAX_TARGET_ALTITUDE_DEG`** (default **75**) und **`GAME_MIN_TARGET_ALTITUDE_DEG`** (default **10**) begrenzen die Zielhöhe (gleichmäßig in Grad zwischen min und max, nicht mehr ``arcsin`` auf [0.5,1]).
- **`b1_kinematics.py`** / **`b1_game_bridge.py`** — B1 native host: headless **`AsteroidGameState`** for embedded CPython. **`b1_kinematics`**: **`B1_RA_OFFSET_DEG`** defaults to **0** (Ursina rig); **`b1_game_bridge.session_init`** sets **`B1_RA_OFFSET_DEG=180`** if unset so GLES mount matches **`b1_host`**. Step **2→3** uses **`b1_kinematics.optical_axis_world_unit`** after each tick’s RA/Dec integration; **`GAME_ALIGN_MAX_ANGLE_DEG`** (default **0.5**) sets the angular tolerance vs. the red target.
- **`orbit_api.py`** — compute-only API (Astropy, Poliastro, NumPy) for the **B1 native host**; no Ursina. Includes `sun_direction`, `get_dome_intersect` (tests / legacy), time helpers (`day2range`, `range2day`, `time_str`, `vector_magnitude`), **`dispatch_compute_json`** (JSON ops), orbit JSON helpers, optional **`seed`** in `compute_orbit_json`.
- **`celestial_settings.py`** — **`GAME_CELESTIAL_DIURNAL_SIGN`** (default **-1**): Tagesdrehung (Stundenwinkel, Sternfeld). **`GAME_CELESTIAL_HORIZ_OFFSET_DEG`** (default **180**): Zusätzlicher Azimut um die Welt-**+Y**-Achse für Sonne, Zielrichtung (`scenario`) und Sternhimmel, damit Kulmination zur geografisch erwarteten Seite zeigt (Kuppel/„Home“-Blick vs. Süden zur Mittagssonne auf der Nordhalbkugel). **0** = früheres Mapping ohne diesen Offset. **`b1_sky_gradient`** liest dieselbe Variable. Die Sonnenrichtung nutzt die übliche Höhenformel sin(alt)=sin φ sin δ + cos φ cos δ cos H plus Meridian-Anpassung; der Ursina-Himmel (`u_sun_dir`) folgt derselben Vektorrechnung wie das DirectionalLight.
- **`game_state.py`** — **Phase A:** Ursina-free step logic (`AsteroidGameState`), same flow as `game.py`; `tick(dt, wall_t, keys, hints)` plus `handle_discrete_input`. Uses NumPy/SciPy only; `time_str` / `sun_direction` duplicated here (keep in sync with `orbit_api`) so tests do not require Poliastro. Env **`GAME_START_BEFORE_WINDOW`** (default **1**) und **`GAME_START_HOURS_BEFORE_WINDOW`** (default **2.5**) setzen die Uhr beim ersten Tick auf einige Stunden *vor* dem Sichtbarkeitsfenster. **`GAME_ALIGN_MAX_ANGLE_DEG`** — siehe oben.
- **Startup:** the help panel opens once at launch (same as before the `game_state` refactor). Set **`ASTEROID_GAME_STARTUP_HELP=0`** to skip it (kiosk / repeat players).
- **Debug (Ursina):** **`GAME_DEBUG_ALIGN=1`** or **`ASTEROID_DEBUG_ALIGN=1`** — gelber Text: Winkel zwischen Laserachse und Zielrichtung, dot-Produkt, Schwellwinkel, **`paused_help`** / **`game_paused`** (wenn „ja“, läuft keine Zielerkennung).
- **`tests/`** — `python -m unittest discover -s tests -v` from this directory (`test_game_state.py` runs without the venv; `test_orbit_api.py` / `test_scenario` / `test_b1_*` need the venv).
- **`../native/b1_host/`** — SDL2 + OpenGL ES 2 + embedded CPython; see `README.md` there for build and Pi deployment.
- **Repo root** — [`../README.md`](../README.md), [`../docs/astro_launcher_integration.md`](../docs/astro_launcher_integration.md) (Astro Mini Games launcher), [`../contrib/`](../contrib/) (example YAML + systemd unit).
