3D assets for Ursina (game_app) and optional B1 native observatory rig.

Expected GLB names (same as game_scene.py), relative to this folder:

  telescope_mount_base.glb
  telescope_mount_ra.glb
  telescope_mount_dec.glb
  telescope_tube.glb
  observatory_domewall.glb
  observatory_dome.glb
  observatory_shutter.glb
  observatory_flap.glb

With game_state enabled (default in b1_host; B1_USE_GAME_STATE=0 to disable), b1_host tries to load these unless overridden by
B1_MOUNT_BASE, B1_MOUNT_RA, B1_MOUNT_DEC, B1_TUBE, B1_DOME_WALL, B1_DOME,
B1_SHUTTER, B1_FLAP (absolute paths).

If none load, the single B1_GLTF mesh (default assets/b1_placeholder.gltf) is used.

World floor in b1_host is y=0 (large quad). Default B1_RIG_TY is -0.6; tune TX/TY/TZ if the dome floats.
Camera: B1_CAM_PIVOT_Y (default 1.4), B1_CAM_MAX_DIST (default 14), B1_CAM_INIT_DIST (default 5.0, start zoom).
After the dome finishes opening (step 2+), b1_host draws a blue laser from the rig optical axis (same kinematics as the GLB mount) and a red target cube (Python tick sends target_guides ox,oy,oz only); B1_TARGET_GUIDES=0 disables; B1_LASER_LEN, B1_TARGET_MARKER_DIST, B1_TARGET_MARKER_SCALE tune sizes.

**[I]** / Schritt-3-Hints: Kuppel-Azimut folgt dem **horizontalen Azimut der optischen Achse** (Blickrichtung). Im **b1_host** setzt **[I]** den Azimut aus derselben ``optical_axis_world``-Richtung wie der gezeichnete Laser (Python ``set_dome_az_deg``); ohne Observatorium fällt die Bridge auf ``align_dome_to_telescope`` (Kinematik) zurück.

Native b1_host applies dome yaw as **negative** game ``dome_az_deg`` so GLES matches Ursina (``rotation_y`` → Panda heading ``-y``). Optional **B1_DOME_AZ_OFFSET_DEG** adds a constant yaw if the dome GLB slit rest direction differs from +X. **[I]** slew: **B1_DOME_ALIGN_ANIM_SEC** (0 = sofort).

Spielstart-Zeit: **GAME_START_BEFORE_WINDOW** (1/0), **GAME_START_HOURS_BEFORE_WINDOW** (Std. vor ``min_time``). Zielhöhe: **GAME_MIN_TARGET_ALTITUDE_DEG** (Standard 10) … **GAME_MAX_TARGET_ALTITUDE_DEG** (Standard 75), gleichverteilt in Grad. Schritt 2→3 (Ursina): Ausrichtung per gerendertem Laser (**telecope_optical_axis**), nicht nur ``b1_kinematics``. Kuppel manuell **[** / **]** in Schritt **2** und **3** (B1 und Desktop).

b1_host tuning: B1_SHUTTER_OPEN_DEG (default +67.5), B1_FLAP_OPEN_DEG (default +80),
B1_FLAP_AXIS=y|z|x (default y; z matches Ursina hinge, opening sign adjusted for GLES).
B1_RA_OFFSET_DEG: **b1_game_bridge** setzt **180** falls unset (GLES); Desktop-Ursina typisch **0** (siehe README).
