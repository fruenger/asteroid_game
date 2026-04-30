 asteroid_game locales (de / en / es)
========================================

- **Source YAML:** asteroid_game/locales/de.yaml, en.yaml, es.yaml
- **Lookup keys:** nested YAML maps flatten to dot notation, e.g. toast.time_window_reject, help.steps.3,
  infotext.step.1, catalog.orbit_block, controls.steuerung
- **API (game_i18n.py):**
  - tr("dotted.key") — optional {placeholders} for Python .format(**kwargs)
  - set_locale("en") — persists alongside other Astro apps to
    ~/.local/share/astro_mini_games/locale.yaml and sets ASTRO_LANG
  - ensure_locale_env() — called on first tr(); also used in touch_game_bridge.session_init()
  - Optional: ASTRO_LAUNCHER_ROOT → path to astro_mini_games checkout for fallback i18n.locale from config.yaml

Ursina: ``game_language_bar.attach_language_switcher`` — Flaggen-PNGs unter ``asteroid_game/assets/local_flags/`` (README dort).

Native ``asteroid_game_touch``: dieselben Dateien unter ``<asteroid-root>/assets/local_flags/``; Sprachwechsel wie zuvor über ``set_locale_from_host``.

Strings not yet routed through locales (examples)
------------------------------------------------

- touch_game_bridge._align_debug_line (GAME_DEBUG_ALIGN) — [Align] debug line
- Ephemeris stderr [asteroid_ephemeris] banners
- native/asteroid_game_touch C++ help/usage text (separate if needed). **Touch-Chrome-Buttons** und Idle-Banner kommen aus ``native_chrome.*`` in den YAML (``tick["chrome_ui"]``); Platzhalter ``{step}``, ``{seconds}`` im Overlay-Header bzw. Idle-Zeile.
