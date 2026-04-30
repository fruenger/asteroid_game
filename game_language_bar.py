"""Top-right locale toggles — same precedence as Astro Mini Games (`ASTRO_LANG`, user locale YAML)."""

from __future__ import annotations

from game_i18n import (
    SUPPORTED_LOCALES,
    ensure_locale_env,
    get_locale,
    locale_flag_png_path,
    set_locale,
)


def attach_language_switcher(*, top: float = 0.46, x_right: float = 0.38) -> None:
    """Add compact buttons parented to ``camera.ui`` (after ``Ursina()`` / ``camera.ui`` exists)."""
    ensure_locale_env()
    from ursina import Button, camera, color, load_texture

    spacing = 0.076
    buttons: dict[str, Button] = {}
    x0 = x_right

    def _refresh() -> None:
        active = get_locale()
        for code, bt in buttons.items():
            bt.color = color.azure.tint(-0.1) if code == active else color.black.tint(0.5)
            te = getattr(bt, "text_entity", None)
            if te is not None:
                te.color = color.white

    for i, code in enumerate(SUPPORTED_LOCALES):
        png = locale_flag_png_path(code)
        if png is not None:
            tex = load_texture(str(png))
            bt = Button(
                parent=camera.ui,
                text="",
                texture=tex,
                scale=(0.065, 0.048),
                position=(x0 - i * spacing, top),
                origin=(0.5, 0.5),
                color=color.black.tint(0.5),
            )
        else:
            bt = Button(
                parent=camera.ui,
                text=code.upper(),
                scale=(0.05, 0.045),
                position=(x0 - i * spacing, top),
                origin=(0.5, 0.5),
            )
            if bt.text_entity is not None:
                bt.text_entity.scale *= 1.35

        def _pick(c=code) -> None:  # noqa: B023
            set_locale(c)
            _refresh()

        bt.on_click = _pick
        buttons[code] = bt
    _refresh()
