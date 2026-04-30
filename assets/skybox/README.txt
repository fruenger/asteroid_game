Cubemap sky for asteroid_game_touch (`ASTEROID_SKYBOX=/path/to/this/dir`)

Place six square images (same pixel size, e.g. 1024x1024):

  px.png / px.jpg   +X  (or right.*)
  nx.png            -X  (or left.*)
  py.png            +Y  sky above  (or top.*)
  ny.png            -Y  ground below  (or bottom.*)
  pz.png            +Z  (or front.*)
  nz.png            -Z  (or back.*)

OpenGL / world: +Y is up. If a face looks mirrored or on the wrong side,
rename files to match your pack (some use different "front" conventions).

CC0 sources (examples):
  https://polyhaven.com/hdris  (export as separate PNG faces if needed)
  https://ambientcg.com/     (some material packs include cubemap faces)

Run:
  ASTEROID_SKYBOX=/path/to/asteroid_game/assets/skybox ./build/asteroid_game_touch ...

If ASTEROID_SKYBOX loads, the procedural gradient sky is skipped unless you
unset ASTEROID_SKYBOX. ASTEROID_SKY_GRADIENT=0 still disables the fallback gradient
when no cubemap is loaded.
