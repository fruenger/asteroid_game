3D assets for Ursina (game_app) and optional native GLES observatory rig.

Expected GLB names (same as game_scene.py), relative to this folder:

  telescope_mount_base.glb
  telescope_mount_ra.glb
  telescope_mount_dec.glb
  telescope_tube.glb
  observatory_domewall.glb
  observatory_dome.glb
  observatory_shutter.glb
  observatory_flap.glb

With game_state enabled (default in asteroid_game_touch; GAME_USE_GAME_STATE=0 to disable), asteroid_game_touch tries to load these unless overridden by
ASTEROID_MOUNT_BASE, ASTEROID_MOUNT_RA, ASTEROID_MOUNT_DEC, ASTEROID_TUBE, ASTEROID_DOME_WALL, ASTEROID_DOME,
ASTEROID_SHUTTER, ASTEROID_FLAP (absolute paths).

If none load, the single ASTEROID_PLACEHOLDER_GLTF mesh (default assets/touch_placeholder.gltf) is used.

World floor in asteroid_game_touch is y=0 (large quad). Default ASTEROID_RIG_TY is -0.6; tune TX/TY/TZ if the dome floats.
Camera: GAME_CAM_PIVOT_Y (default 1.4), GAME_CAM_MAX_DIST (default 14), GAME_CAM_INIT_DIST (default 7.0), GAME_CAM_INIT_HEIGHT (default 2.0 world Y of camera eye unless explicit pitch is set).
After the dome finishes opening (step 2+), asteroid_game_touch draws a blue laser from the rig optical axis (same kinematics as the GLB mount) and a red target cube (Python tick sends target_guides ox,oy,oz only); GAME_TARGET_GUIDES=0 disables; ASTEROID_TARGET_LASER_LEN, ASTEROID_TARGET_MARKER_DIST, ASTEROID_TARGET_MARKER_SCALE tune sizes.

**[I]** / Schritt-3-Hints: Kuppel-Azimut folgt dem **horizontalen Azimut der optischen Achse** (Blickrichtung). Im **asteroid_game_touch** setzt **[I]** den Azimut aus derselben ``optical_axis_world``-Richtung wie der gezeichnete Laser (Python ``set_dome_az_deg``); ohne Observatorium fällt die Bridge auf ``align_dome_to_telescope`` (Kinematik) zurück.

Native asteroid_game_touch applies dome yaw as **negative** game ``dome_az_deg`` so GLES matches Ursina (``rotation_y`` → Panda heading ``-y``). Optional **GAME_DOME_AZ_OFFSET_DEG** adds a constant yaw if the dome GLB slit rest direction differs from +X. **[I]** slew: **GAME_DOME_ALIGN_ANIM_SEC** (0 = sofort).

Spielstart-Zeit: **GAME_START_BEFORE_WINDOW** (1/0), **GAME_START_HOURS_BEFORE_WINDOW** (Std. vor ``min_time``). Zielhöhe: **GAME_MIN_TARGET_ALTITUDE_DEG** (Standard 10) … **GAME_MAX_TARGET_ALTITUDE_DEG** (Standard 75), gleichverteilt in Grad. Schritt 2→3 (Ursina): Ausrichtung per gerendertem Laser (**telecope_optical_axis**), nicht nur ``rig_kinematics``. Kuppel manuell **[** / **]** in Schritt **2** und **3** (native host und Desktop).

asteroid_game_touch tuning: ASTEROID_SHUTTER_OPEN_DEG (default +67.5), ASTEROID_FLAP_OPEN_DEG (default +80),
ASTEROID_FLAP_AXIS=y|z|x (default y; z matches Ursina hinge, opening sign adjusted for GLES).

GAME_RA_OFFSET_DEG: **touch_game_bridge** setzt **180** falls unset (GLES); Desktop-Ursina typisch **0** (siehe README).
