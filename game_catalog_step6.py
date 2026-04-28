"""Post-game catalog / orbit summary UI (step 6)."""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np
from PIL import Image
from ursina import Button, Entity, Text, TextField, Texture, camera, color

from game_catalog_common import append_catalog_entry, default_object_name, orbital_elements_block


def run_step6(*, orbit, game_gs, infotext) -> None:
    infotext.enabled = False
    infotext.message = ""

    Text("Objektname", x=-0.4 * camera.aspect_ratio_getter(), y=0.4, origin=(-0.5, 0.5))
    obj_name = TextField(character_limit=30, max_lines=1, x=-0.4 * camera.aspect_ratio_getter(), y=0.36, origin=(-0.5, 0.5))
    obj_name.bg.scale_x = 0.5
    obj_name.bg.scale_y = 0.05
    obj_name.add_text(default_object_name())
    obj_name.active = False

    Text("Endeckerteam", x=-0.4 * camera.aspect_ratio_getter(), y=0.25, origin=(-0.5, 0.5))
    discoverer_name = TextField(character_limit=30, max_lines=1, x=-0.4 * camera.aspect_ratio_getter(), y=0.21, origin=(-0.5, 0.5))
    discoverer_name.text_entity.color = color.green
    discoverer_name.bg.scale_x = 0.5
    discoverer_name.bg.scale_y = 0.05
    discoverer_name.add_text("Name d. Entdeckerteams")
    discoverer_name.active = True

    Text("Bahnelemente", x=-0.4 * camera.aspect_ratio_getter(), y=0.05, origin=(-0.5, 0.5))
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
    img_view.texture = Texture(Image.fromarray(((coadded_image / mx) * 255).astype(np.uint8), mode="L").convert("RGBA"))
    save_button = Button(
        "Add to Catalog", highlight_scale=1.1, pressed_scale=0.95, scale=(0.25, 0.1), x=-0.4 * camera.aspect_ratio_getter(), y=-0.35, origin=(-0.5, 0.5)
    )

    def save_button_fct():
        append_catalog_entry(obj_name.text, discoverer_name.text)
        python = sys.executable
        os.execl(python, python, *sys.argv)

    save_button.on_click = save_button_fct
