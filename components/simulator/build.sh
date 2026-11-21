#!/usr/bin/env bash
# ./revolve build simulator: the dependency image, then with colcon INSIDE that image
#   1. Gazebo Kura (+ the MuJoCo plugin) from ThirdParty/gz_kura (fetch.sh: kura.repos pins + ThirdParty/patches)
#      into ThirdParty/gz_kura/install, ThirdParty/gz_kura mounted at /opt/gz_ws;
#   2. ros_gz, gz_ros2_control (+ demos) and the gz_*_vendor packages (all satisfied by the Kura install, nothing
#      vendored) from the ThirdParty submodules into ThirdParty/ros_gz_ws/install, mounted at /opt/ros_gz_ws.
# Both install dirs are git-ignored and mounted at the same paths at runtime (absolute install paths). build/ and
# the ccache dirs persist, so a rebuild only compiles what changed.
#   build.sh [--clean] [colcon args...]     --clean: remove both build/ install/ log/ first (sources kept);
#                                           colcon args go to the Kura build
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${REVOLVE_SIM_IMAGE:-revolve-simulator:latest}
kura=$repo/ThirdParty/gz_kura
ws=$repo/ThirdParty/ros_gz_ws
if [[ ${1:-} == --clean ]]; then
  shift; rm -rf "$kura/build" "$kura/install" "$kura/log" "$ws/build" "$ws/install" "$ws/log"
fi
t0=$(date +%s)
# Init the submodules this build needs if missing; checked-out ones are left as they are.
for p in ThirdParty/ros_gz ThirdParty/gz_ros2_control \
         $(git -C "$repo" config -f .gitmodules --get-regexp '^submodule\..*_vendor\.path$' | cut -d' ' -f2); do
  [[ -e $repo/$p/.git ]] || git -C "$repo" submodule update --init -- "$p"
done
docker build -t "$IMAGE" "$repo/components/simulator"
echo "image $IMAGE: $(docker image inspect "$IMAGE" --format '{{.Size}}') bytes, $(( $(date +%s) - t0 )) s"
mkdir -p "$kura/.ccache" "$ws/.ccache"
colcon_args=(--merge-install --parallel-workers "$(nproc)"
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  --event-handlers console_direct- console_cohesion- status-)
run() { docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e CCACHE_MAXSIZE=10G -e MAKEFLAGS "$@"; }

# 1. Kura, without ROS in the environment (BASH_ENV off), so it cannot pick up the Jetty vendors.
run -e BASH_ENV= -e CCACHE_DIR=/opt/gz_ws/.ccache \
  -v "$kura":/opt/gz_ws -v "$repo/ThirdParty/patches":/opt/patches:ro \
  -w /opt/gz_ws "$IMAGE" bash -c '
set -euo pipefail
git config --global --add safe.directory "*"
bash /opt/gz_ws/fetch.sh
[[ -f kura_versions.txt ]] || vcs log -l1 src > kura_versions.txt
colcon build --base-paths src "$@"
ccache -s | grep -E "Hits|Misses" || true
' build "${colcon_args[@]}" "$@"
test -f "$kura/install/setup.bash"
echo "kura OK: $kura/install ($(du -sh "$kura/install" | cut -f1)), $(( $(date +%s) - t0 )) s"

# 2. ros_gz + gz_ros2_control on /opt/ros/lyrical, Kura first on every path (the image's BASH_ENV).
# GZ_RELAX_VERSION_MATCH (upstream vendor switch): gz_dartsim_vendor asks for DART EXACT 6.16, which the apt 6.16.6
# never matches, so it would build DART from source; the Kura majors match either way.
run -e CCACHE_DIR=/opt/ros_gz_ws/.ccache -e GZ_RELAX_VERSION_MATCH=1 \
  -v "$kura/install":/opt/gz_ws/install:ro -v "$ws":/opt/ros_gz_ws -v "$repo/ThirdParty":/opt/third_party:ro \
  -w /opt/ros_gz_ws "$IMAGE" bash -c '
set -euo pipefail
colcon build --base-paths /opt/third_party/ros_gz /opt/third_party/gz_ros2_control /opt/third_party/gz_vendors \
  --packages-up-to ros_gz_bridge ros_gz_sim ros_gz_image ros_gz_interfaces gz_ros2_control gz_ros2_control_demos "$@"
ccache -s | grep -E "Hits|Misses" || true
' build "${colcon_args[@]}"
test -f "$ws/install/setup.bash"
echo "BUILD OK simulator: $kura/install, $ws/install ($(du -sh "$ws/install" | cut -f1)), $(( $(date +%s) - t0 )) s total"
