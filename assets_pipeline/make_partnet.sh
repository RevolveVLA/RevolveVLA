#!/usr/bin/env bash
# Build the PartNet-Mobility kitchen models (fridge 11211, cabinet 46452 as
# kitchen_counter) from assets/partnet_mobility/dataset/ into assets/generated_partnet/.
# table_01 and cup stay the ReplicaCAD/primitive ones in assets/generated/
# (specs/scenes/kitchen_partnet/scene.yaml lists both dirs).
# Re-runs itself in the revolve-assets container (assets_pipeline/Dockerfile, built here).
set -euo pipefail
cd "$(dirname "$0")/.."                        # the repo root: assets/ is git-ignored there
if [[ ! -f /.dockerenv ]]; then
  docker build -q -t revolve-assets:latest assets_pipeline >/dev/null
  exec docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/repo -w /repo \
    revolve-assets:latest bash assets_pipeline/make_partnet.sh "$@"
fi

D=assets/partnet_mobility/dataset
OUT=assets/generated_partnet
A=assets_pipeline
C=$A/configs
for id in 11211 46452; do
  [[ -f $D/$id/mobility.urdf ]] || { echo "PartNet-Mobility object $id missing under $D"; exit 1; }
done
# The download is untrusted data: -I never imports from it, and the pipeline only
# parses its URDF/OBJ/MTL text and copies mesh/texture files.
P="python3 -I $A/urdf_to_gz.py"

$P $D/11211/mobility.urdf $OUT --config $C/partnet_fridge.yaml
$P $D/46452/mobility.urdf $OUT --config $C/partnet_cabinet.yaml
