#!/usr/bin/env bash
# ./revolve build evaluation: evaluation has no image of its own (it is part of the training
# env). A rollout uses revolve-training (serve), revolve-dataset + ThirdParty/rosetta/install (Rosetta's runner)
# and revolve-simulator-ros, the thin deps-only ROS 2 layer FROM revolve-simulator
# (components/simulator/ros/Dockerfile) that this script builds. The other three are built by their own
# components' build scripts first if missing.
#   build.sh [--clean]      --clean: rebuild revolve-simulator-ros without the layer cache
#   env REVOLVE_SIM_ROS_IMAGE (default revolve-simulator-ros:latest), REVOLVE_SIM_IMAGE (its base)
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${REVOLVE_SIM_ROS_IMAGE:-revolve-simulator-ros:latest}
BASE=${REVOLVE_SIM_IMAGE:-revolve-simulator:latest}
nocache=(); [[ ${1:-} == --clean ]] && nocache=(--no-cache)
t0=$(date +%s)
have() { docker image inspect "$1" >/dev/null 2>&1; }
{ have "$BASE" && [[ -f $repo/ThirdParty/gz_kura/install/setup.bash ]]; } || bash "$repo/components/simulator/build.sh"
have "${REVOLVE_TRAINING_IMAGE:-revolve-training:latest}" || bash "$repo/components/training/build.sh"
{ have "${REVOLVE_DATASET_IMAGE:-revolve-dataset:latest}" && [[ -f $repo/ThirdParty/rosetta/install/BUILD_OK ]]; } \
  || bash "$repo/components/dataset/build.sh"
docker build "${nocache[@]}" --build-arg BASE="$BASE" -t "$IMAGE" "$repo/components/simulator/ros"
echo "BUILD OK evaluation: $IMAGE ($(docker image inspect "$IMAGE" --format '{{.Size}}') bytes), $(( $(date +%s) - t0 )) s"
