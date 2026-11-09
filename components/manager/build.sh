#!/usr/bin/env bash
# ./revolve build manager: the slim dependency image (python:3.12-slim + requirements.txt). Nothing to build from source.
#   build.sh [--clean]      --clean: build without the layer cache
#   env REVOLVE_MANAGER_IMAGE (default revolve-manager:latest)
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${REVOLVE_MANAGER_IMAGE:-revolve-manager:latest}
args=()
if [[ ${1:-} == --clean ]]; then args=(--no-cache); fi
t0=$(date +%s)
docker build "${args[@]}" -t "$IMAGE" "$repo/components/manager"
echo "BUILD OK manager: image $IMAGE $(docker image inspect "$IMAGE" --format '{{.Size}}') bytes, $(( $(date +%s) - t0 )) s"
