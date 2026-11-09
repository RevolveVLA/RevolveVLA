#!/usr/bin/env bash
# ./revolve build training: the dependency image only. Nothing is built from source: lerobot-train comes
# from the hashed lock, and serve runs Rosetta's lerobot_rosetta (pure Python) from the ThirdParty/rosetta
# submodule mounted read-only at runtime.
#   build.sh [--clean]      --clean: build without the layer cache
#   env REVOLVE_TRAINING_IMAGE (default revolve-training:latest)
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${REVOLVE_TRAINING_IMAGE:-revolve-training:latest}
args=()
if [[ ${1:-} == --clean ]]; then args=(--no-cache); fi
t0=$(date +%s)
docker build "${args[@]}" -t "$IMAGE" "$repo/components/training"
test -f "$repo/ThirdParty/rosetta/policies/lerobot/lerobot_rosetta/lerobot_rosetta/policy_server.py" \
  || { echo "ThirdParty/rosetta not checked out (git submodule update --init ThirdParty/rosetta)"; exit 1; }
echo "BUILD OK training: image $IMAGE $(docker image inspect "$IMAGE" --format '{{.Size}}') bytes, $(( $(date +%s) - t0 )) s"
