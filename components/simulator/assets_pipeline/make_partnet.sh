#!/usr/bin/env bash
# Build the PartNet-Mobility kitchen models (fridge 11211, cabinet 46452 as
# kitchen_counter) from assets/partnet_mobility/dataset/ into assets/generated_partnet/
# and load-test each one alone under the MuJoCo plugin. table_01 and cup stay the
# ReplicaCAD/primitive ones in assets/generated/ (specs/scenes/kitchen_partnet/scene.yaml lists both dirs).
# Re-runs itself in the revolve-simulator container (./revolve build simulator).
#   components/simulator/assets_pipeline/make_partnet.sh              build + load test
#   components/simulator/assets_pipeline/make_partnet.sh --snapshot   ... + one PNG per model (GPU)
#   components/simulator/assets_pipeline/make_partnet.sh --no-test    build only
set -euo pipefail
cd "$(dirname "$0")/../../.."                  # the repo root: assets/ is git-ignored there
if [[ ! -f /.dockerenv ]]; then
  [[ -f ThirdParty/gz_kura/install/setup.bash ]] || { echo "Build Gazebo first: ./revolve build simulator"; exit 1; }
  exec docker run --rm --gpus all -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -e __NV_PRIME_RENDER_OFFLOAD=1 -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    -v "$PWD/ThirdParty/gz_kura/install":/opt/gz_ws/install:ro -v "$PWD":/repo -w /repo \
    revolve-simulator:latest bash components/simulator/assets_pipeline/make_partnet.sh "$@"
fi
set +u; source /opt/gz_ws/install/setup.bash; set -u

D=assets/partnet_mobility/dataset
OUT=assets/generated_partnet
A=components/simulator/assets_pipeline
C=$A/configs
for id in 11211 46452; do
  [[ -f $D/$id/mobility.urdf ]] || { echo "PartNet-Mobility object $id missing under $D"; exit 1; }
done
# The download is untrusted data: -I never imports from it, and the pipeline only
# parses its URDF/OBJ/MTL text and copies mesh/texture files.
P="python3 -I $A/urdf_to_gz.py"

$P $D/11211/mobility.urdf $OUT --config $C/partnet_fridge.yaml
$P $D/46452/mobility.urdf $OUT --config $C/partnet_cabinet.yaml

[[ "${1:-}" == --no-test ]] && exit 0
python3 $A/load_test.py --models-dir $OUT "$@" fridge kitchen_counter
