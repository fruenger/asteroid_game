"""Post-game catalog / orbit summary UI (step 6)."""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from ursina import Button, Entity, Text, TextField, Texture, camera, color

from game_catalog_common import append_catalog_entry, default_object_name, orbital_elements_block
from game_i18n import tr


def run_step6(*, orbit, game_gs, infotext) -> None:
    infotext.enabled = False
    infotext.message = ""

    Text(tr("catalog_ui.object_name"), x=-0.4 * camera.aspect_ratio_getter(), y=0.4, origin=(-0.5, 0.5))
    obj_name = TextField(character_limit=30, max_lines=1, x=-0.4 * camera.aspect_ratio_getter(), y=0.36, origin=(-0.5, 0.5))
    obj_name.bg.scale_x = 0.5
    obj_name.bg.scale_y = 0.05
    obj_name.add_text(default_object_name())
    obj_name.active = False

    Text(tr("catalog_ui.discoverer_team"), x=-0.4 * camera.aspect_ratio_getter(), y=0.25, origin=(-0.5, 0.5))
    discoverer_name = TextField(character_limit=30, max_lines=1, x=-0.4 * camera.aspect_ratio_getter(), y=0.21, origin=(-0.5, 0.5))
    discoverer_name.text_entity.color = color.green
    discoverer_name.bg.scale_x = 0.5
    discoverer_name.bg.scale_y = 0.05
    discoverer_name.add_text(tr("catalog_ui.discoverer_placeholder"))
    discoverer_name.active = True

    Text(tr("catalog_ui.orbital_elements"), x=-0.4 * camera.aspect_ratio_getter(), y=0.05, origin=(-0.5, 0.5))
    orbital_elements = TextField(character_limit=30, max_lines=8, x=-0.4 * camera.aspect_ratio_getter(), y=0.0, origin=(-0.5, 0.5))
    orbital_elements.bg.scale_x = 0.5
    orbital_elements.bg.scale_y = 0.25
    orbital_elements.add_text(orbital_elements_block(orbit))
    orbital_elements.active = False

    img_view = Entity(model="quad", parent=camera.ui, x=0.7, y=0.0, scale=(0.8, 0.8), origin=(0.5, 0.0))
    coadded_image = np.sum(game_gs.all_images, axis=0)
    mx = float(np.max(coadded_image))
    if mx <= 0.0:
        mx = 1.0
    row_c, col_c, pad = game_gs.step6_identified_marker_pixels()
    pil = Image.fromarray(((coadded_image / mx) * 255).astype(np.uint8), mode="L").convert("RGBA")
    dw = ImageDraw.Draw(pil)
    r = float(pad)
    bbox = (float(col_c - r), float(row_c - r), float(col_c + r), float(row_c + r))
    dw.ellipse(bbox, outline=(255, 0, 0, 255), width=3)
    img_view.texture = Texture(pil)
    save_button = Button(
        tr("catalog_ui.add_to_catalog"),
        highlight_scale=1.1,
        pressed_scale=0.95,
        scale=(0.25, 0.1),
        x=-0.4 * camera.aspect_ratio_getter(),
        y=-0.35,
        origin=(-0.5, 0.5),
    )

    def save_button_fct():
        append_catalog_entry(obj_name.text, discoverer_name.text)
        python = sys.executable
        os.execl(python, python, *sys.argv)

    save_button.on_click = save_button_fct
