#!/usr/bin/env bash
# ./revolve build simulator: the dependency image, then Gazebo Kura (+ the MuJoCo plugin) built with
# colcon INSIDE that image from ThirdParty/gz_kura (fetch.sh: kura.repos pins + ThirdParty/patches)
# into ThirdParty/gz_kura/install (git-ignored). As in the earlier pipeline's Kura build: ThirdParty/gz_kura is
# mounted at /opt/gz_ws (gz embeds absolute install paths, so runs mount install/ at the same place),
# build/ and .ccache/ persist there, so a rebuild only compiles what changed.
#   build.sh [--clean] [colcon args...]     --clean: remove build/ install/ log/ first (sources kept)
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${REVOLVE_SIM_IMAGE:-revolve-simulator:latest}
ws=$repo/ThirdParty/gz_kura
if [[ ${1:-} == --clean ]]; then shift; rm -rf "$ws/build" "$ws/install" "$ws/log"; fi
t0=$(date +%s)
docker build -t "$IMAGE" "$repo/components/simulator"
echo "image $IMAGE: $(docker image inspect "$IMAGE" --format '{{.Size}}') bytes, $(( $(date +%s) - t0 )) s"
mkdir -p "$ws/.ccache"
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -e CCACHE_DIR=/opt/gz_ws/.ccache -e CCACHE_MAXSIZE=10G -e MAKEFLAGS \
  -v "$ws":/opt/gz_ws -v "$repo/ThirdParty/patches":/opt/patches:ro \
  -w /opt/gz_ws "$IMAGE" bash -c '
set -euo pipefail
git config --global --add safe.directory "*"
bash /opt/gz_ws/fetch.sh
[[ -f kura_versions.txt ]] || vcs log -l1 src > kura_versions.txt
colcon build --merge-install --base-paths src --parallel-workers "$(nproc)" \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  --event-handlers console_direct- console_cohesion- status- \
  "$@"
ccache -s | grep -E "Hits|Misses" || true
' build "$@"
test -f "$ws/install/setup.bash"
echo "BUILD OK simulator: $ws/install ($(du -sh "$ws/install" | cut -f1)), $(( $(date +%s) - t0 )) s total"
