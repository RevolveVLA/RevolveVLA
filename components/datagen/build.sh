#!/usr/bin/env bash
# ./revolve build datagen: the dependency image, then cuRobo (its CUDA extensions, for the image's
# TORCH_CUDA_ARCH_LIST) built INSIDE that image from ThirdParty/curobo into ThirdParty/curobo/install
# (git-ignored), mounted read-only at /opt/curobo/install at runtime. The build works on a copy of the
# sources in the container's /tmp, so the submodule's working tree stays clean. AutoMoMa is pure Python
# and is not built: jobs mount ThirdParty/automoma at /opt/automoma (its configs and the paths it
# resolves from its own source) and put it on PYTHONPATH; here it is only import-checked.
# Patches in ThirdParty/patches/curobo/*.patch are applied to the copy (none today; AutoMoMa is read in place).
#   build.sh [--clean]      --clean: remove ThirdParty/curobo/install first
#   env REVOLVE_GPU_ARCH (default 12.0 = sm_120; 8.9 for Modal), MAX_JOBS (nvcc jobs, default 4)
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${REVOLVE_DATAGEN_IMAGE:-revolve-datagen:latest}
ARCH=${REVOLVE_GPU_ARCH:-12.0}
inst=$repo/ThirdParty/curobo/install
if [[ ${1:-} == --clean ]]; then shift; rm -rf "$inst"; fi
t0=$(date +%s)
docker build --build-arg TORCH_CUDA_ARCH_LIST="$ARCH" -t "$IMAGE" "$repo/components/datagen"
echo "image $IMAGE: $(docker image inspect "$IMAGE" --format '{{.Size}}') bytes, $(( $(date +%s) - t0 )) s"
if [[ -f $inst/BUILD_OK && $(cat "$inst/BUILD_OK") == "$ARCH" ]]; then
  echo "cuRobo already built for $ARCH in $inst (--clean to rebuild)"
else
  rm -rf "$inst"
  mkdir -p "$inst"
  docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e MAX_JOBS="${MAX_JOBS:-4}" \
    -v "$repo/ThirdParty/curobo":/src/curobo:ro -v "$repo/ThirdParty/automoma":/opt/automoma:ro \
    -v "$repo/ThirdParty/patches":/src/patches:ro -v "$inst":/opt/curobo/install \
    "$IMAGE" bash -c '
set -euo pipefail
mkdir /tmp/curobo && tar -C /src/curobo --exclude=./.git --exclude=./install -cf - . | tar -C /tmp/curobo -xf -
for p in /src/patches/curobo/*.patch; do [[ -f $p ]] && git -C /tmp/curobo apply "$p" && echo "applied $p"; done
cd /tmp/curobo
# no tags in the shallow submodule: pretend the version (as the legacy Dockerfile)
SETUPTOOLS_SCM_PRETEND_VERSION=0.7.4.post1.dev6 UV_CACHE_DIR=/tmp/uv \
  uv pip install --no-build-isolation --no-deps --target /opt/curobo/install .
cd /
PYTHONPATH=/opt/curobo/install:/opt/automoma python -c "
import torch, curobo, automoma
from curobo.curobolib import kinematics_fused_cu, geom_cu, lbfgs_step_cu, line_search_cu, tensor_step_cu
from automoma.planning.pipeline import PlanningPipeline
print(\"curobo\", curobo.__file__, \"automoma\", automoma.__file__)"
'
  echo "$ARCH" > "$inst/BUILD_OK"
fi
echo "BUILD OK datagen: $inst ($(du -sh "$inst" | cut -f1), arch $ARCH), $(( $(date +%s) - t0 )) s total"
