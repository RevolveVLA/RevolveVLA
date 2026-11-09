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
| Gazebo Kura (15 repos: gz-cmake, gz-common, gz-fuel-tools, gz-gui, gz-math, gz-msgs, gz-physics, gz-plugin, gz-rendering, gz-sensors, gz-sim, gz-tools, gz-transport, gz-utils, sdformat) | gz_kura/src (git-ignored) | https://github.com/gazebosim/<repo> | per repo in gz_kura/kura.repos (pinned 2026-11-01) | Apache-2.0 | `gz_kura/fetch.sh` (vcstool `vcs import`, then patches) |
| MuJoCo 3.11.0 | inside gz-physics | - | vendored by gz-physics | Apache-2.0 | with gz-physics |

Patches:
- `patches/gz-physics/gz-physics-mujoco-solver-options.patch`: MuJoCo solver options (cone, noslip iterations, ...)
  from environment variables, as SDF does not expose them yet (gazebosim/gz-physics#1119).
- `patches/gz-sim/gz-sim-video-recorder-skip-dropped-frames.patch`

Fetch: `git submodule update --init` (shallow per .gitmodules) and `ThirdParty/gz_kura/fetch.sh`
(`--check` only checks that every Kura pin is fetchable). Python packages (LeRobot 0.6.1, torch, ...)
are pinned in the component images / lock files, not here.
