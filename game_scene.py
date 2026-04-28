"""3D scene: telescope rig, dome, skybox, lighting, transition mask."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ursina import (
    DirectionalLight,
    Entity,
    Mesh,
    Shader,
    Vec3,
    camera,
    color,
    invoke,
)

from celestial_settings import diurnal_sign, horizontal_celestial_offset_rad
from game_state import sun_direction
from game_settings import SCENE_SHADER, USE_SHADOWS, dlog, shadow_map_resolution
from game_ui_shaders import load_shader


@dataclass
class SceneEntities:
    telescope_base: Entity
    base_pivot: Entity
    ra_pivot: Entity
    dec_pivot: Entity
    telescope_base2: Entity
    telescope_ra: Entity
    telescope_dec: Entity
    telecope_optical_axis: Entity
    domewall: Entity
    dome_pivot: Entity
    shutter_pivot: Entity
    shutter: Entity
    flap_pivot: Entity
    flap: Entity
    bottom: Entity
    light_box: Entity
    light: DirectionalLight
    tr_mask: "TransitionMask"
    sky: "Skybox"


class Skybox(Entity):
    def __init__(self):
        super().__init__(
            model="models/skybox.glb",
            shader=Shader(vertex=load_shader("shaders/sky.vert"), fragment=load_shader("shaders/sky.frag")),
            unlit=True,
            parent=camera,
        )
        self.setBin("background", 0)
        self.scale = camera.clip_plane_far * 0.8 / 100

    def update(self):
        self.world_rotation = Vec3(0, 0, 0)


class TransitionMask(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(model="quad", color=color.black, parent=camera.ui, scale=(camera.aspect_ratio_getter(), 1), *args, **kwargs)
        self.position = Vec3(0, 0, 0.01)
        self.alpha_setter(0)

    def fade_in(self, duration=1.0, update_frq=20.0):
        for alpha_value in np.linspace(0.0, 1.0, int(duration * update_frq)):
            invoke(self.alpha_setter, alpha_value, delay=alpha_value * duration)

    def fade_out(self, duration=1.0, update_frq=20.0):
        for inverse_alpha_value in np.linspace(0.0, 1.0, int(duration * update_frq)):
            invoke(self.alpha_setter, 1.0 - inverse_alpha_value, delay=inverse_alpha_value * duration)
            self.alpha_setter(1)


def build_scene() -> SceneEntities:
    telescope_base = Entity(model="models/telescope_mount_base.glb", shader=SCENE_SHADER, color=color.gray, position=Vec3(0, 0, 0))

    base_pivot = Entity(position=Vec3(-1, 3, 0), rotation=Vec3(0, 0, 52 - 90), parent=telescope_base)

    ra_pivot = Entity(position=Vec3(0, 0, 0), parent=base_pivot)
    dec_pivot = Entity(position=Vec3(1.8, 1.15, 0), parent=ra_pivot)

    sky = Skybox()
    sky.set_shader_input("u_time", 0)
    sky.set_shader_input("u_diurnal_sign", diurnal_sign())
    sky.set_shader_input("u_horiz_yaw_rad", horizontal_celestial_offset_rad())
    s0 = sun_direction(0.5, 52.0, 0.5)
    sky.set_shader_input("u_sun_dir", (float(s0[0]), float(s0[1]), float(s0[2])))
    sky.hide(0b0001)
    dlog("skybox ready")

    telescope_base2 = Entity(
        model="models/telescope_mount_ra.glb", shader=SCENE_SHADER, color=color.gray, position=Vec3(-1, 3, 0), rotation=Vec3(0, 0, 52 - 90)
    )
    telescope_ra = Entity(model="models/telescope_mount_dec.glb", shader=SCENE_SHADER, color=color.gray, parent=ra_pivot)

    telescope_dec = Entity(
        model="models/telescope_tube.glb", shader=SCENE_SHADER, color=color.gray, parent=dec_pivot, rotation=(0, 90, 0), position=(2.2, -0.5, 0)
    )

    telecope_optical_axis = Entity(
        position=Vec3(2.2, 0.0, 0), scale=Vec3(1, 10, 1), color=color.red, parent=dec_pivot
    )

    dlog("telescope rig entities ready")

    domewall = Entity(
        model="models/observatory_domewall.glb",
        shader=SCENE_SHADER,
        color=color.gray,
        scale=Vec3(15, 15, 15),
        collider="mesh",
    )
    dome_pivot = Entity(
        model="models/observatory_dome.glb",
        position=Vec3(0, 0, 0),
        shader=SCENE_SHADER,
        color=color.gray,
        scale=Vec3(15, 15, 15),
        collider="mesh",
    )
    dlog("dome_pivot (mesh collider) ready")

    shutter_pivot = Entity(position=Vec3(0, 0, 0), parent=dome_pivot)
    shutter = Entity(model="models/observatory_shutter.glb", parent=shutter_pivot, shader=SCENE_SHADER, color=color.gray)
    flap_pivot = Entity(position=Vec3(1, 0, 0), parent=dome_pivot, shader=SCENE_SHADER, color=color.gray)
    flap = Entity(model="models/observatory_flap.glb", position=Vec3(-1, 0, 0), parent=flap_pivot, shader=SCENE_SHADER, color=color.gray)
    bottom = Entity(model="plane", position=Vec3(0, -7.5, 0), scale=Vec3(1000, 1, 1000), color=color.gray, shader=SCENE_SHADER)

    dlog("dome parts + ground plane ready")

    light_box = Entity(model="cube", scale=100.0, color=color.rgb(0, 0, 0, 0))
    _shadow_res = shadow_map_resolution()
    light = DirectionalLight(shadows=USE_SHADOWS, shadow_map_resolution=_shadow_res)
    light.update_bounds(entity=light_box)
    dlog(f"directional light ready (shadows={USE_SHADOWS}, shadow_map={int(_shadow_res[0])}x{int(_shadow_res[1])})")

    tr_mask = TransitionMask()
    dlog("transition mask ready")

    return SceneEntities(
        telescope_base=telescope_base,
        base_pivot=base_pivot,
        ra_pivot=ra_pivot,
        dec_pivot=dec_pivot,
        telescope_base2=telescope_base2,
        telescope_ra=telescope_ra,
        telescope_dec=telescope_dec,
        telecope_optical_axis=telecope_optical_axis,
        domewall=domewall,
        dome_pivot=dome_pivot,
        shutter_pivot=shutter_pivot,
        shutter=shutter,
        flap_pivot=flap_pivot,
        flap=flap,
        bottom=bottom,
        light_box=light_box,
        light=light,
        tr_mask=tr_mask,
        sky=sky,
    )
