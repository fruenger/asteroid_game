"""
Locale bundles for Asteroid (`asteroid_game/locales/{de,en,es}.yaml`).

Resolution order mirrors **Astro Mini Games** ``shared/i18n``:
1. ``~/.local/share/astro_mini_games/locale.yaml`` (``locale: de|en|es``),
2. environment ``ASTRO_LANG``,
3. optional launcher config (``ASTRO_LAUNCHER_ROOT`` → ``config.yaml`` → ``i18n.locale``),
4. ``de``.

Embedded **asteroid_game_touch** should read the same user file / ``ASTRO_LANG`` so kiosk and Ursina builds stay aligned.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

import yaml

_LOG = logging.getLogger(__name__)

SUPPORTED_LOCALES: tuple[str, ...] = ("de", "en", "es")
ASTRO_LANG_ENV = "ASTRO_LANG"
_ENV_LAUNCHER_ROOT = "ASTRO_LAUNCHER_ROOT"
_DEFAULT_LOCALE = "de"

_catalogs: dict[str, dict[str, str]] = {}
_locale_callbacks: list[Callable[[str], None]] = []
_initialized_env = False


def _astro_user_locale_path() -> Path:
    return Path.home() / ".local" / "share" / "astro_mini_games" / "locale.yaml"


def locales_dir() -> Path:
    return Path(__file__).resolve().parent / "locales"


def locale_flag_png_path(code: str) -> Path | None:
    """PNG for locale buttons (`assets/local_flags/`), mirrors astro_mini_games launcher fallbacks."""
    base = Path(__file__).resolve().parent / "assets" / "local_flags"
    c = (code or "").strip().lower()
    if c not in SUPPORTED_LOCALES:
        return None
    candidates: list[Path] = [base / f"{c}.png"]
    if c == "en":
        candidates.extend([base / "gb.png", base / "en_GB.png", base / "uk.png"])
    for p in candidates:
        if p.is_file():
            return p
    return None


def _read_user_saved_locale() -> str | None:
    path = _astro_user_locale_path()
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        loc = data.get("locale")
        if loc is None and isinstance(data.get("i18n"), dict):
            loc = data["i18n"].get("locale")
        loc = str(loc or "").strip().lower()
        return loc if loc in SUPPORTED_LOCALES else None
    except (OSError, yaml.YAMLError) as e:
        _LOG.debug("Could not read user locale file: %s", e)
        return None


def get_launcher_config_locale() -> str:
    """``i18n.locale`` from ``$ASTRO_LAUNCHER_ROOT/config.yaml`` (optional)."""
    root = os.environ.get(_ENV_LAUNCHER_ROOT, "").strip()
    if not root:
        return _DEFAULT_LOCALE
    cfg = Path(root).expanduser() / "config.yaml"
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        loc = str((data.get("i18n") or {}).get("locale", _DEFAULT_LOCALE)).lower().strip()
        if loc in SUPPORTED_LOCALES:
            return loc
    except (OSError, yaml.YAMLError) as e:
        _LOG.debug("Could not read launcher config locale %s: %s", cfg, e)
    return _DEFAULT_LOCALE


def resolve_effective_locale() -> str:
    """User YAML → ``ASTRO_LANG`` → launcher config → ``de``."""
    saved = _read_user_saved_locale()
    if saved:
        return saved
    env = os.environ.get(ASTRO_LANG_ENV, "").strip().lower()
    if env in SUPPORTED_LOCALES:
        return env
    return get_launcher_config_locale()


def ensure_locale_env(*, notify: bool = False) -> str:
    """Assign ``ASTRO_LANG``, preload catalog(s). Call early in ``bootstrap()`` / native bridge."""
    global _initialized_env
    prev = os.environ.get(ASTRO_LANG_ENV, "").strip().lower()
    loc = resolve_effective_locale()
    os.environ[ASTRO_LANG_ENV] = loc
    _load_catalog(loc)
    _initialized_env = True
    if notify and prev in SUPPORTED_LOCALES and loc != prev:
        _notify_callbacks(loc)
    return loc


def _flatten(prefix: str, node: Any, out: dict[str, str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            nk = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                _flatten(nk, v, out)
            elif v is not None:
                out[nk] = str(v)
    elif node is not None and prefix:
        out[prefix] = str(node)


def _load_catalog(locale: str) -> None:
    path = locales_dir() / f"{locale}.yaml"
    fallback_path = locales_dir() / f"{_DEFAULT_LOCALE}.yaml"
    flat: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        _flatten("", raw, flat)
    except (OSError, yaml.YAMLError) as e:
        _LOG.debug("Locale file missing or invalid %s: %s", path, e)
    if locale != _DEFAULT_LOCALE:
        try:
            with open(fallback_path, "r", encoding="utf-8") as f:
                raw_fb = yaml.safe_load(f) or {}
            fb: dict[str, str] = {}
            _flatten("", raw_fb, fb)
            for k, v in fb.items():
                flat.setdefault(k, v)
        except (OSError, yaml.YAMLError):
            pass
    _catalogs[locale] = flat


def get_locale() -> str:
    v = os.environ.get(ASTRO_LANG_ENV, "").strip().lower()
    if v in SUPPORTED_LOCALES:
        return v
    return _DEFAULT_LOCALE


def register_locale_callback(cb: Callable[[str], None]) -> None:
    if cb not in _locale_callbacks:
        _locale_callbacks.append(cb)


def unregister_locale_callback(cb: Callable[[str], None]) -> None:
    try:
        _locale_callbacks.remove(cb)
    except ValueError:
        pass


def _notify_callbacks(code: str) -> None:
    for cb in list(_locale_callbacks):
        try:
            cb(code)
        except Exception as e:
            _LOG.warning("locale callback failed: %s", e)


def set_locale(code: str) -> str:
    """Persist to user ``locale.yaml`` (shared with Astro), update ``ASTRO_LANG``, reload catalogs."""
    code_l = (code or _DEFAULT_LOCALE).strip().lower()
    if code_l not in SUPPORTED_LOCALES:
        code_l = _DEFAULT_LOCALE
    prev = get_locale()
    _persist_user_locale(code_l)
    os.environ[ASTRO_LANG_ENV] = code_l
    _load_catalog(code_l)
    if prev != code_l:
        _notify_callbacks(code_l)
    return code_l


def _persist_user_locale(locale: str) -> None:
    path = _astro_user_locale_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                {"locale": locale},
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
    except OSError as e:
        _LOG.warning("Could not write user locale file %s: %s", path, e)


def tr(key: str, **kwargs: Any) -> str:
    """Translate flattened dotted ``key``. Format with kwargs when placeholders exist."""
    if not _initialized_env:
        ensure_locale_env()
    loc = get_locale()
    if loc not in _catalogs:
        _load_catalog(loc)
    catalog = _catalogs.get(loc, {})
    s = catalog.get(key)
    if s is None:
        _LOG.debug("Missing translation key: %s locale=%s", key, loc)
        if _DEFAULT_LOCALE not in _catalogs:
            _load_catalog(_DEFAULT_LOCALE)
        s = _catalogs.get(_DEFAULT_LOCALE, {}).get(key, key)
    if kwargs:
        try:
            return str(s).format(**kwargs)
        except (KeyError, ValueError):
            return str(s)
    return str(s)


def pick_lang_field(value: Any, locale: str | None = None) -> str:
    """Resolve ``dict[de/en/es]|str`` (launcher-style option fields)."""
    loc = (locale or get_locale()).strip().lower()
    if loc not in SUPPORTED_LOCALES:
        loc = _DEFAULT_LOCALE
    if isinstance(value, dict):
        v = value.get(loc) or value.get(_DEFAULT_LOCALE)
        if v is not None:
            return str(v)
        for x in value.values():
            return str(x) if x is not None else ""
        return ""
    if value is None:
        return ""
    return str(value)
