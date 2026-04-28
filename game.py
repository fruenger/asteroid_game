"""Asteroid game entry: early boot, then Ursina main module hooks (update / input)."""

import game_boot  # noqa: F401 — faulthandler + shadow PRC before Panda3D

import game_app

game_app.bootstrap()


def update():
    game_app.frame_update()


def input(key):
    game_app.handle_input(key)


game_app.run_forever()
