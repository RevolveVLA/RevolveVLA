#!/usr/bin/env bash
# Build the kitchen's table_01 and cup from assets/replicacad/ into assets/generated/
# (the fridge and cabinet are PartNet-Mobility: make_partnet.sh; the room: make_room.py).
# Re-runs itself in the revolve-assets container (assets_pipeline/Dockerfile, built here).
set -euo pipefail
cd "$(dirname "$0")/.."                        # the repo root: assets/ is git-ignored there
if [[ ! -f /.dockerenv ]]; then
  docker build -q -t revolve-assets:latest assets_pipeline >/dev/null
  exec docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/repo -w /repo \
    revolve-assets:latest bash assets_pipeline/make_replicacad.sh "$@"
fi

R=assets/replicacad
OUT=assets/generated
A=assets_pipeline
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
