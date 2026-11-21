# Third-party versions

Sources only: nothing here is built by git. `revolve build` builds into `ThirdParty/<module>/install/`
(git-ignored) inside the component's dependency image. Never edit these sources; local changes
go in `patches/<module>/*.patch`.

| Module | Path | URL | Pin | Licence | Fetched by |
|---|---|---|---|---|---|
| AutoMoMa (fi6 fork) | automoma | https://github.com/fi6/automoma.git | ec423552bf3ac86ca240062fc0a80bdbbbbd4a63 | MIT | git submodule (shallow; its own third_party/curobo submodule is NOT fetched, cuRobo below is the same pin) |
| cuRobo (fi6 fork) | curobo | https://github.com/fi6/curobo.git | 2015f966fcc4ffacba73d61800770b5e5b373ffe | NVIDIA License (non-commercial) | git submodule (shallow) |
| Rosetta | rosetta | https://github.com/iblnkn/rosetta.git | d552e34cd391490473f7fe0bf1580f56d28aa8f9 | Apache-2.0 | git submodule (shallow) |
| franka_description | franka_description | https://github.com/frankaemika/franka_description.git | 7aeeddc449edf8d62b594f9e36a81da53e7796f9 (tag 2.9.0) | Apache-2.0 | git submodule (shallow) |
| SO-ARM100 (TheRobotStudio) | so_arm100 | https://github.com/TheRobotStudio/SO-ARM100.git | a758567c3978dfeefe282ede0500085a48fe8f78 | Apache-2.0 | git submodule (shallow; the simulator reads Simulation/SO101/so101_new_calib.urdf and its assets/*.stl in place) |
| Gazebo Kura (15 repos: gz-cmake, gz-common, gz-fuel-tools, gz-gui, gz-math, gz-msgs, gz-physics, gz-plugin, gz-rendering, gz-sensors, gz-sim, gz-tools, gz-transport, gz-utils, sdformat) | gz_kura/src (git-ignored) | https://github.com/gazebosim/<repo> | per repo in gz_kura/kura.repos (pinned 2026-11-01) | Apache-2.0 | `gz_kura/fetch.sh` (vcstool `vcs import`, then patches) |
| MuJoCo 3.11.0 | inside gz-physics | - | vendored by gz-physics | Apache-2.0 | with gz-physics |
| ros_gz | ros_gz | https://github.com/gazebosim/ros_gz.git | e740decc40479c1a959c30915075278791f15fdf (branch lyrical) | Apache-2.0 | git submodule (shallow); built against Kura by `revolve build simulator` into ros_gz_ws/install |
| gz_ros2_control | gz_ros2_control | https://github.com/ros-controls/gz_ros2_control.git | e5cab21d9c45fc7bfa3473410b1992a74e0a3155 (branch lyrical, matches the Lyrical ros2_control binaries) | Apache-2.0 | git submodule (shallow); built as ros_gz |
| gz_cmake_vendor | gz_vendors/gz_cmake_vendor | https://github.com/gazebo-release/gz_cmake_vendor.git | d5604eb6038850f2f22e26687a0ec3ee4d72b08d (branch rolling) | Apache-2.0 | git submodule (shallow) |
| gz_utils_vendor | gz_vendors/gz_utils_vendor | https://github.com/gazebo-release/gz_utils_vendor.git | 4f2a41db8022252abeb0c492f1353ddbb698ebac (branch rolling) | Apache-2.0 | git submodule (shallow) |
| gz_math_vendor | gz_vendors/gz_math_vendor | https://github.com/gazebo-release/gz_math_vendor.git | 18f620200ae671940ed2c290390d42e1fee70ac3 (branch rolling) | Apache-2.0 | git submodule (shallow) |
| gz_tools_vendor | gz_vendors/gz_tools_vendor | https://github.com/gazebo-release/gz_tools_vendor.git | 46c088a5c0692db9ee67c806b09bb55f9e5e754a (branch rolling) | Apache-2.0 | git submodule (shallow) |
| sdformat_vendor | gz_vendors/sdformat_vendor | https://github.com/gazebo-release/sdformat_vendor.git | 3b5888f537c75fbb7d1ce462ae3f954f9671bc4e (branch rolling) | Apache-2.0 | git submodule (shallow) |
| gz_plugin_vendor | gz_vendors/gz_plugin_vendor | https://github.com/gazebo-release/gz_plugin_vendor.git | 4013d88cd48285e461ac8b506c21a310c3029a25 (branch rolling) | Apache-2.0 | git submodule (shallow) |
| gz_common_vendor | gz_vendors/gz_common_vendor | https://github.com/gazebo-release/gz_common_vendor.git | 1537f20aec6c2319dcd5b5eba11d1530118dcff1 (branch rolling) | Apache-2.0 | git submodule (shallow) |
| gz_msgs_vendor | gz_vendors/gz_msgs_vendor | https://github.com/gazebo-release/gz_msgs_vendor.git | 4dc447372c001a24736b3939cfc96adb52187e52 (branch rolling) | Apache-2.0 | git submodule (shallow) |
| gz_transport_vendor | gz_vendors/gz_transport_vendor | https://github.com/gazebo-release/gz_transport_vendor.git | 20b5621462bc374e9614eeaedac16cbdb8cb8dcb (branch rolling) | Apache-2.0 | git submodule (shallow) |
| gz_fuel_tools_vendor | gz_vendors/gz_fuel_tools_vendor | https://github.com/gazebo-release/gz_fuel_tools_vendor.git | 28a46ad77fb41a72810676abc0b7b6f8a2f328ba (branch rolling) | Apache-2.0 | git submodule (shallow) |
| gz_rendering_vendor | gz_vendors/gz_rendering_vendor | https://github.com/gazebo-release/gz_rendering_vendor.git | 5745f6b3bd2925a49728365d7aac4330be07c4ae (branch rolling) | Apache-2.0 | git submodule (shallow) |
| gz_physics_vendor | gz_vendors/gz_physics_vendor | https://github.com/gazebo-release/gz_physics_vendor.git | fde7d0a152296a49dcd84a0750578ebc405fce7d (branch rolling) | Apache-2.0 | git submodule (shallow) |
| gz_sensors_vendor | gz_vendors/gz_sensors_vendor | https://github.com/gazebo-release/gz_sensors_vendor.git | 43a64eacc2dbd87e46ee3cadab34b894400a0dfd (branch rolling) | Apache-2.0 | git submodule (shallow) |
| gz_gui_vendor | gz_vendors/gz_gui_vendor | https://github.com/gazebo-release/gz_gui_vendor.git | 7280f054f3fe7b6d5ae8a94f49fc09d33ce802e8 (branch rolling) | Apache-2.0 | git submodule (shallow) |
| gz_sim_vendor | gz_vendors/gz_sim_vendor | https://github.com/gazebo-release/gz_sim_vendor.git | 1c356cec119f1b2f2386dbcff46243e41d93edf6 (branch rolling) | Apache-2.0 | git submodule (shallow) |
| gz_dartsim_vendor | gz_vendors/gz_dartsim_vendor | https://github.com/gazebo-release/gz_dartsim_vendor.git | 160ee5d1d106a2e5f31d1f213b4061d78389958d (branch rolling) | Apache-2.0 | git submodule (shallow) |
| gz_ogre_next_vendor | gz_vendors/gz_ogre_next_vendor | https://github.com/gazebo-release/gz_ogre_next_vendor.git | 9da1382ae79068402994181a9135ed69dbc10a0d (branch rolling) | Apache-2.0 | git submodule (shallow) |

Patches:
- `patches/gz-physics/gz-physics-mujoco-solver-options.patch`: MuJoCo solver options (cone, noslip iterations, ...)
  from environment variables, as SDF does not expose them yet (gazebosim/gz-physics#1119).
- `patches/gz-sim/gz-sim-video-recorder-skip-dropped-frames.patch`

The gz_*_vendor packages (gazebo-release, `rolling` = the Kura majors: gz-sim 11, gz-physics 10, sdformat 17, ...)
only wrap Kura: each one finds the Kura install and vendors nothing (gz_dartsim_vendor / gz_ogre_next_vendor find
the apt DART 6.16 / OGRE-Next 2.3; the build sets GZ_RELAX_VERSION_MATCH, else DART's EXACT 6.16 check fails). They shadow the Jetty
vendors that the Lyrical binaries pull into /opt/ros/lyrical, so ros_gz / gz_ros2_control link Kura.

Fetch: `git submodule update --init` (shallow per .gitmodules; `revolve build simulator` inits the ones it needs) and `ThirdParty/gz_kura/fetch.sh`
(`--check` only checks that every Kura pin is fetchable). Python packages (LeRobot 0.6.1, torch, ...)
are pinned in the component images / lock files, not here.
