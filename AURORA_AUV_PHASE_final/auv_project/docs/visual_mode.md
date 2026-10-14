# 3D visual mode (real PyBullet window)

The dashboard's 2D maps are not a replacement for seeing the vehicle in the actual simulation. Two ways to open
the real 3D PyBullet window (needs PyBullet and a display on the same machine):

1. `python run_visual_demo.py [--scenario mixed_terrain_obstacles] [--seed 3] [--obstacles 53] [--dynamics 6dof] [--speed 1.0]`
   (`--list` shows scenarios; `--mission default` uses the short configured mission; `--no-follow` frees the camera; `--camera top` starts in top view;
   `--no-underwater` keeps PyBullet's default look)
   This launcher is **not a second scene**: it calls the same `src/simulation.run_simulation.run()` the
   dashboard uses, with a config override. Same world generator, same obstacles, same sensors/EKF/mapper/navigator,
   same collision + sonar geometry. `--obstacles N` asks for exactly N real PyBullet obstacle bodies.
2. Dashboard -> *Next start* -> **3D PyBullet window: on** -> *Save for next start* -> **Start**
   (sets `dashboard.headless: false`; the child process then opens the GUI).

## What you see (all real)
* **underwater look** (`visualization.underwater_theme`, `src/simulation/underwater_look.py`): a deep-water
  background colour, one light from above with shadow rendering, and the water surface sheet at z = 0.
  All three are *rendering of the real scene*: the surface is a VISUAL-ONLY body (no collision shape), so sonar
  ray casts, DVL, collision checks, mapping and planning are byte-for-byte unaffected. `--no-underwater` turns it off.
* the AUV (true simulated state) moving along its mission, paced at `visualization.realtime_factor` (1.0 = real time)
* the uneven seabed mesh and every generated rock / boulder / pillar / wall / reef tower / wreck: the actual PyBullet
  collision bodies (also drawn, no separate "visual-only" copy)
* **sonar rays**: red = measured hit at the measured range, faint blue = valid no-return to max range; dropped rays
  are not drawn at all
* **small yellow dot** (15 cm): the EKF *estimated* position (compare with the AUV body = true position)
* **red points**: OCCUPIED cells of the sonar-built occupancy grid (the same grid the dashboard shows), at the estimated depth
* the path travelled (cyan), mission markers (start/waypoints/goal), and a label with the scenario, seed and body count
* a **third-person follow camera** (see below): behind and above the AUV, turning with its heading, never hidden by terrain
Closing the window stops the simulation.

## Camera (`src/simulation/gui_camera.py`, camera only)
* **FOLLOW AUV** (default): 3.8 m behind and ~1.3 m above the AUV (`distance_m`, `pitch_deg`), aimed 0.8 m ahead so the AUV
  stays near the centre of the picture and the way forward is visible. Yaw follows the AUV's heading, smoothed.
* **TOP VIEW**: straight down above the AUV (`top_height_m`), +Y up like the dashboard maps.
* **RESET VIEW**: drops mouse adjustments and returns to the default view.
* Switch with the buttons in the PyBullet side panel or the keys:
  **FOLLOW AUV / TOP VIEW / RESET VIEW** (always) plus **ORBIT LEFT / ORBIT RIGHT / ZOOM IN / ZOOM OUT**
  (`camera.orbit_zoom_buttons`, on for `run_visual_demo.py`),
  keys **1** follow, **2** top, **3** reset, **4 / 5** orbit ±`orbit_step_deg`, **6 / 7** zoom by `zoom_step`.
  Mouse rotate/zoom still works while following and is remembered as an offset relative to the auto-camera;
  RESET VIEW clears every offset. The same key list is printed in the window itself.
* **Geometry safety**: PyBullet's debug camera has no collision, so the rig ray-tests the *real* collision world (seabed and
  obstacles) from the AUV to the camera and from the camera to the seabed. If the preferred pose is blocked it tries higher
  or side poses, then moves in toward the AUV; it closes in fast and backs off slowly so the picture does not pop.
* **Cost**: in `p.GUI` every PyBullet call is a GUI round trip (~12 ms with software OpenGL). The camera uses 3 calls per
  update (read, one batched ray test, set), updates every `update_every_n_steps` (4) steps and polls buttons/keys only every
  `poll_every_n_updates` (6) updates.
* `camera_follow: false` / `--no-follow` leaves the camera to you (whole-mission overview at start).
* It reads the true pose and the collision world only to place the camera; it creates no bodies and changes nothing else.

### Verifying the camera
`tools/verify_gui_camera.py` measures AUV visibility with PyBullet's own ray tests and a segmentation render from the camera
PyBullet actually holds (line of sight to the AUV, where its centre lands in the frame, AUV pixel count):
`--mode live` runs `run_visual_demo.main()` itself (slow without a GPU: use `--max-wall`), `--mode record` runs the real mission
headless and saves the true poses, `--mode replay` plays those poses through the same real world in the GUI with the real camera
code. Under a headless machine prefix with `xvfb-run -a -s "-screen 0 1280x800x24 +extension GLX"`.

## Honest scope
Overlays are simulation-side debug drawing (they may use the true pose to place themselves) and never feed mapping,
the EKF or the controller. Config: `visualization:` in `configs/config.yaml`. The overlay logic is unit-tested against a
recording stand-in for the PyBullet drawing API (`tests/test_gui_visuals.py`, `tests/test_gui_camera.py`); pixels are checked
by `tools/verify_gui_camera.py` and by looking.## Verifying the environment in real PyBullet
`python tools/verify_pybullet_environment.py [--mission]` prints, per scenario: bodies built, collision + visual geometry
present, seabed is a collision mesh, determinism, ray-by-ray agreement between real PyBullet and the generator's own
geometry, and real sonar hits. `python tools/verify_visual_scene.py` renders frames of the real demo scene in the GUI and
checks objectively that the backdrop is deep water, the seabed/obstacles/AUV are visible (segmentation mask of the real
body ids) and the water surface was created without any collision shape; it writes the PNGs to `results/gui_scene/`
(`--no-underwater` renders the default look for comparison). `python -m pytest -v` runs the same checks (`tests/test_pybullet_world.py`) and ends with an
"AURORA PyBullet execution report" showing whether PyBullet tests ran or were skipped. Set `AURORA_REQUIRE_PYBULLET=1`
to make a missing PyBullet a hard error instead of a skip.


## Presentation pass (rendering only)

* The AUV hull is drawn orange with an "AURORA AUV" tag attached to the body (`visualization.highlight_auv`, `auv_label`).
  Its shape, mass, inertia and collision geometry are untouched; it is NOT scaled (a bigger drawing would no longer match
  what the sonar and the collision detection see). The camera distance (3.8 m) is what makes it read larger.
* Sonar: measured hits are red and thin (`hit_ray_width`); the no-return fan is dim blue, every 6th ray, width 1
  (`free_ray_stride`, `free_ray_width`, `show_free_rays: false` or `--hit-rays-only` for hits only).
* The EKF estimate marker is a 15 cm dot (`estimate_marker_radius_m`) so it no longer hides the AUV when the estimate is good.
* Seabed and obstacle colours are muted (blue-green silt, rock/concrete tones); colours only, the meshes are the same.
* The window opens at 1600x900 (`visualization.window`, `--window WxH`) so the side panel has room for the full button
  labels. If a label is still clipped on your display, use keys 1-7 (always available) or the dashboard camera buttons.

* Side-panel buttons are labelled with the digit of their key (1 follow, 2 top, 3 reset, 4/5 orbit, 6/7 zoom): this PyBullet
  panel clips a label to about two characters, so words like "FOLLOW AUV" were unreadable. PyBullet's default Camera RGB /
  Depth preview panels are switched off (presentation only).

## Changing the goal and the obstacle count while you watch

* `python run_visual_demo.py` now starts the real EKF-only navigator (use `--no-navigation` for the old fixed route) and
  `--goal X Y Z` sets the first goal. In the 3-D window: **arrow keys** move a pending goal (green marker, Shift = 5 m),
  **Page Up/Down** change its depth, **G** or **Enter** applies it (the red GOAL marker moves, A* replans, the follower
  steers to it), **H** cancels. A goal the map rejects (occupied/out of bounds) is reported on the marker and stays pending.
* Obstacle count: `--obstacles N` at launch (procedural scenarios), or in the dashboard set the count and click
  **Apply & restart now**: the run restarts and builds exactly N real obstacle bodies (shown as requested vs actual).
* Panel buttons unreadable on your display? `--no-panel-buttons` hides them; keys 1-7 and the dashboard camera buttons remain.
