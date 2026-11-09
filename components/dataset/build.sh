#!/usr/bin/env bash
# ./revolve build dataset: the dependency image, then Rosetta colcon-built INSIDE that image from
# ThirdParty/rosetta into ThirdParty/rosetta/install (git-ignored), mounted read-only at
# /opt/rosetta/install at runtime (the install base it is built at). The build works on a copy of the
# sources in the container's /tmp, so the submodule's working tree stays clean. Packages: up to
# lerobot_rosetta (rosetta, rosetta_interfaces, rosetta_ros = rosetta_port, lerobot_robot_rosetta), as
# the legacy automoma-lerobot Dockerfile. Patches in ThirdParty/patches/rosetta/*.patch are applied to
# the copy (none today: the legacy image used Rosetta unpatched).
#   build.sh [--clean]      --clean: remove ThirdParty/rosetta/install first
#   env REVOLVE_DATASET_IMAGE (default revolve-dataset:latest)
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${REVOLVE_DATASET_IMAGE:-revolve-dataset:latest}
src=$repo/ThirdParty/rosetta
inst=$src/install
sha=$(git -C "$src" rev-parse HEAD)
if [[ ${1:-} == --clean ]]; then shift; rm -rf "$inst"; fi
t0=$(date +%s)
docker build --build-context rosetta="$src" -t "$IMAGE" "$repo/components/dataset"
echo "image $IMAGE: $(docker image inspect "$IMAGE" --format '{{.Size}}') bytes, $(( $(date +%s) - t0 )) s"
if [[ -f $inst/BUILD_OK && $(cat "$inst/BUILD_OK") == "$sha" ]]; then
  echo "Rosetta $sha already built in $inst (--clean to rebuild)"
else
  rm -rf "$inst"
  mkdir -p "$inst"
  docker run --rm --network none -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$src":/src/rosetta:ro -v "$repo/ThirdParty/patches":/src/patches:ro -v "$inst":/opt/rosetta/install \
    "$IMAGE" bash -c '
set -eo pipefail
mkdir /tmp/rosetta && tar -C /src/rosetta --exclude=./.git --exclude=./install -cf - . | tar -C /tmp/rosetta -xf -
for p in /src/patches/rosetta/*.patch; do [[ -f $p ]] && git -C /tmp/rosetta apply "$p" && echo "applied $p"; done
source /opt/ros/$ROS_DISTRO/setup.bash
cd /tmp
colcon --log-base /tmp/log build --base-paths /tmp/rosetta/rosetta /tmp/rosetta/robots/ros /tmp/rosetta/policies/lerobot \
  --packages-up-to lerobot_rosetta --build-base /tmp/build --install-base /opt/rosetta/install \
  --event-handlers console_direct-
source /opt/rosetta/install/setup.bash
python3 -c "
from rosetta.policies import available_dataset_writers
from rosetta_ros.offline import port
import lerobot
assert \"lerobot\" in available_dataset_writers(), available_dataset_writers()
print(\"rosetta_port\", port.__file__, \"writers\", available_dataset_writers(), \"lerobot\", lerobot.__version__)"
'
  echo "$sha" > "$inst/BUILD_OK"
fi
echo "BUILD OK dataset: $inst ($(du -sh "$inst" | cut -f1), Rosetta $sha), $(( $(date +%s) - t0 )) s total"
