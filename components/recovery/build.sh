#!/usr/bin/env bash
# ./revolve build recovery: the datagen build first when its image or cuRobo install is missing
# (recovery runs on both), then revolve-recovery FROM the datagen image (nothing added).
#   build.sh [--clean]      --clean: passed to the datagen build (rebuilds cuRobo)
#   env REVOLVE_DATAGEN_IMAGE, REVOLVE_RECOVERY_IMAGE
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
DG=${REVOLVE_DATAGEN_IMAGE:-revolve-datagen:latest}
IMAGE=${REVOLVE_RECOVERY_IMAGE:-revolve-recovery:latest}
t0=$(date +%s)
if [[ ${1:-} == --clean ]] || ! docker image inspect "$DG" >/dev/null 2>&1 \
    || [[ ! -f $repo/ThirdParty/curobo/install/BUILD_OK ]]; then
  bash "$repo/components/datagen/build.sh" "$@"
fi
docker build --build-arg DATAGEN_IMAGE="$DG" -t "$IMAGE" "$repo/components/recovery"
echo "BUILD OK recovery: image $IMAGE FROM $DG ($(docker image inspect "$IMAGE" --format '{{.Size}}') bytes), $(( $(date +%s) - t0 )) s"
