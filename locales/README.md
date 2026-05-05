# Localization (`locales/`)

The game ships **three catalogs**: `de.yaml`, `en.yaml`, `es.yaml`.

---

## Keys

YAML maps are flattened to **dot keys** for `tr()`, for example:

| Example key | Typical use |
| ----------- | ----------- |
| `toast.time_window_reject` | Feedback when the chosen time is outside the window |
| `help.steps.3` | Step-specific help body |
| `infotext.step.1` | Short status line with `{min_time}` / `{max_time}` placeholders |
| `catalog.orbit_block`, `catalog_ui.*` | Catalog and form copy |
| `controls.steuerung` | Controls overlay |
| `native_chrome.*` | Strings for the **native** bottom bar / idle banner (`tick["chrome_ui"]`); placeholders such as `{step}`, `{seconds}` |

---

## API (`game_i18n.py`)

| Function | Purpose |
| -------- | ------- |
| **`tr("dotted.key")`** | Translate; optional **`.format(**kwargs)`** placeholders in YAML |
| **`set_locale("en")`** | Switch language; persists with other Astro apps under `~/.local/share/astro_mini_games/locale.yaml` and sets **`ASTRO_LANG`** |
| **`ensure_locale_env()`** | Called on first **`tr()`**; also from **`touch_game_bridge.session_init()`** for the GLES host |
| **`ASTRO_LAUNCHER_ROOT`** (optional) | Path to an **astro_mini_games** checkout so `i18n.locale` from `config.yaml` can seed the default locale |

---

## Flag assets

- **Ursina:** language bar uses PNGs under **`asteroid_game/assets/local_flags/`** (see that folder’s readme).
- **Native host:** same PNGs under **`<asteroid-root>/assets/local_flags/`**; host calls **`set_locale_from_host`** when the user taps **ES | EN | DE**.

---

## Not yet in YAML (examples)

- **`touch_game_bridge._align_debug_line`** — raw `[Align]` debug line when **`GAME_DEBUG_ALIGN`** is on.
- Ephemeris **`[asteroid_ephemeris]`** stderr banners.
- **`asteroid_game_touch --help`** usage text (C++), unless duplicated into YAML on purpose.
