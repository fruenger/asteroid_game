"""One-shot stage transitions triggered from game_state tick events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ursina import Entity, Mesh, Vec3, camera, color, invoke

if TYPE_CHECKING:
    from game_scene import SceneEntities


def make_stageup_event(scene: "SceneEntities", object_dir_cartesian) -> Callable[[int], None]:
    def stageup_event(stage: int) -> None:
        if stage == 2:
            Entity(
                model=Mesh(vertices=[Vec3(0, 0, 0), Vec3(0, 1000, 0)], mode="line", thickness=2),
                color=color.azure,
                parent=scene.telecope_optical_axis,
            )
            Entity(
                model="sphere",
                color=color.red,
                position=0.6 * camera.clip_plane_far * object_dir_cartesian,
                scale=[50, 50, 50],
            )

        if stage == 4:
            scene.tr_mask.fade_in(2.5)
            invoke(scene.tr_mask.fade_out, 2.5, delay=2.5)
            import game_globals as gg

            gg.cheat_through = False

    return stageup_event
