#!/usr/bin/env bash
# ./revolve build dataset: the dependency image (Dockerfile), then Rosetta colcon-built INSIDE that image against ROS 2
# Lyrical from ThirdParty/rosetta into ThirdParty/rosetta/install (git-ignored), mounted read-only at
# /opt/rosetta/install at runtime (the install base it is built at). The build works on a copy of the sources in the
# container's /tmp, so the submodule's working tree stays clean; local fixes only as ThirdParty/patches/rosetta/*.patch,
# applied to that copy (none today). Packages: up to lerobot_rosetta (rosetta, rosetta_interfaces, rosetta_ros =
# rosetta_port, lerobot_robot_rosetta, lerobot_rosetta).
# install/BUILD_OK holds the submodule sha, the ROS distro and the patches' digest: an install that differs in any of
# them (e.g. an older Jazzy build of the same sha) is rebuilt.
#   build.sh [--clean]      --clean: remove ThirdParty/rosetta/install first
#   env REVOLVE_DATASET_IMAGE (default revolve-dataset:latest)
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${REVOLVE_DATASET_IMAGE:-revolve-dataset:latest}
src=$repo/ThirdParty/rosetta
inst=$src/install
patches=$repo/ThirdParty/patches/rosetta
if [[ ${1:-} == --clean ]]; then shift; rm -rf "$inst"; fi
t0=$(date +%s)
docker build --build-context rosetta="$src" -t "$IMAGE" "$repo/components/dataset"
echo "image $IMAGE: $(docker image inspect "$IMAGE" --format '{{.Size}}') bytes, $(( $(date +%s) - t0 )) s"
distro=$(docker run --rm --network none "$IMAGE" bash -c 'echo $ROS_DISTRO')
mkdir -p "$patches"
patch_digest=$(find "$patches" -maxdepth 1 -name '*.patch' | sort | xargs -r cat | sha256sum | cut -c1-16)
stamp="$(git -C "$src" rev-parse HEAD) $distro patches:$patch_digest"
if [[ -f $inst/BUILD_OK && $(cat "$inst/BUILD_OK") == "$stamp" ]]; then
  echo "Rosetta already built in $inst ($stamp; --clean to rebuild)"
else
  rm -rf "$inst"
  mkdir -p "$inst" "$patches"
  docker run --rm --network none -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$src":/src/rosetta:ro -v "$patches":/src/patches:ro -v "$inst":/opt/rosetta/install \
    "$IMAGE" bash -c '
set -eo pipefail
mkdir /tmp/rosetta && tar -C /src/rosetta --exclude=./.git --exclude=./install -cf - . | tar -C /tmp/rosetta -xf -
for p in /src/patches/*.patch; do [[ -f $p ]] && git -C /tmp/rosetta apply "$p" && echo "applied $p"; done
source /opt/ros/$ROS_DISTRO/setup.bash
cd /tmp
colcon --log-base /tmp/log build --base-paths /tmp/rosetta/rosetta /tmp/rosetta/robots/ros /tmp/rosetta/policies/lerobot \
  --packages-up-to lerobot_rosetta --build-base /tmp/build --install-base /opt/rosetta/install \
  --event-handlers console_direct-
source /opt/rosetta/install/setup.bash
python3 -c "
from rosetta.policies import available_dataset_writers
from rosetta_ros.offline import port
import rosetta_interfaces.msg, lerobot, torch
assert \"lerobot\" in available_dataset_writers(), available_dataset_writers()
print(\"rosetta_port\", port.__file__, \"writers\", available_dataset_writers(), \"lerobot\", lerobot.__version__,
      \"torch\", torch.__version__)"
'
  echo "$stamp" > "$inst/BUILD_OK"
fi
echo "BUILD OK dataset: $inst ($(du -sh "$inst" | cut -f1), Rosetta $stamp), $(( $(date +%s) - t0 )) s total"
