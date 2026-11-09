#!/usr/bin/env bash
# Build the kitchen's table_01 and cup from assets/replicacad/ into assets/generated/
# and load-test each one alone under the MuJoCo plugin (the fridge and cabinet are
# PartNet-Mobility: make_partnet.sh; the room: make_room.py). Re-runs itself in the
# revolve-simulator container (./revolve build simulator).
#   components/simulator/assets_pipeline/make_replicacad.sh              build + load test
#   components/simulator/assets_pipeline/make_replicacad.sh --snapshot   ... + one PNG per model (GPU)
#   components/simulator/assets_pipeline/make_replicacad.sh --no-test    build only
set -euo pipefail
cd "$(dirname "$0")/../../.."                  # the repo root: assets/ is git-ignored there
if [[ ! -f /.dockerenv ]]; then
  [[ -f ThirdParty/gz_kura/install/setup.bash ]] || { echo "Build Gazebo first: ./revolve build simulator"; exit 1; }
  exec docker run --rm --gpus all -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -e __NV_PRIME_RENDER_OFFLOAD=1 -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    -v "$PWD/ThirdParty/gz_kura/install":/opt/gz_ws/install:ro -v "$PWD":/repo -w /repo \
    revolve-simulator:latest bash components/simulator/assets_pipeline/make_replicacad.sh "$@"
fi
set +u; source /opt/gz_ws/install/setup.bash; set -u

R=assets/replicacad
OUT=assets/generated
A=components/simulator/assets_pipeline
C=$A/configs
[[ -f $R/configs/objects/frl_apartment_table_01.object_config.json ]] || { echo "ReplicaCAD files missing under $R"; exit 1; }
# -I: never import anything from the (untrusted) asset directories
P="python3 -I $A/urdf_to_gz.py"

$P $R/configs/objects/frl_apartment_table_01.object_config.json $OUT --config $C/replicacad_table_01.yaml
$P $R/configs/objects/frl_apartment_cup_01.object_config.json $OUT --config $C/replicacad_cup_01.yaml

# Cup for the scene: the ReplicaCAD cup if it fits the Panda gripper, else a
# primitive cylinder cup. Width = smaller horizontal extent of its collisions.
WIDTH=$(python3 -I -c 'import json,sys; a=json.load(open(sys.argv[1]))["aabb_model_frame"]; print(min(a[1][0]-a[0][0], a[1][1]-a[0][1]))' \
  $OUT/replicacad_cup_01/manifest.json)
if python3 -I -c "import sys; sys.exit(0 if $WIDTH <= 0.07 else 1)"; then
  echo "cup: ReplicaCAD cup_01 is $WIDTH m wide (<= 0.07): using it"
  $P $R/configs/objects/frl_apartment_cup_01.object_config.json $OUT --config $C/replicacad_cup_01.yaml --name cup
else
  echo "cup: ReplicaCAD cup_01 is $WIDTH m wide (> 0.07): using the primitive cylinder cup"
  $P $A/primitives/cup_cylinder.urdf $OUT --config $C/cup_cylinder.yaml
fi

[[ "${1:-}" == --no-test ]] && exit 0
python3 $A/load_test.py --models-dir $OUT "$@" \
  table_01 cup replicacad_cup_01
