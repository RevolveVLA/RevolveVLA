#!/usr/bin/env bash
# One-time asset step (./revolve assets): fetch the kitchen's inputs into assets/ where their licences allow
# it, then convert them to Gazebo SDF models (urdf_to_gz.py, make_room.py):
#   OUT/generated_partnet/  fridge (PartNet-Mobility 11211), kitchen_counter (46452)
#   OUT/generated/          table_01, replicacad_cup_01, cup (ReplicaCAD or a primitive), kitchen_room_partnet
# OUT defaults to assets/ (the dirs specs/scenes/*/scene.yaml lists). Re-runs itself in the revolve-assets
# container (assets_pipeline/Dockerfile, built here; no GPU, no Gazebo).
#   assets_pipeline/make_assets.sh [--out DIR] [--offline]    --offline: no downloads, room in plain colours
set -euo pipefail
cd "$(dirname "$0")/.."                        # the repo root: assets/ is git-ignored there
OUT=assets OFFLINE=
while (($#)); do
  case $1 in
    --out) OUT=$2; shift 2 ;;
    --offline) OFFLINE=--offline; shift ;;
    *) echo "usage: $0 [--out DIR] [--offline]"; exit 2 ;;
  esac
done
if [[ ! -f /.dockerenv ]]; then
  mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
  docker build -q -t revolve-assets:latest assets_pipeline >/dev/null
  exec docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/repo -v "$OUT":/out -w /repo \
    revolve-assets:latest bash assets_pipeline/make_assets.sh --out /out $OFFLINE
fi

A=assets_pipeline
C=$A/configs
D=assets/partnet_mobility/dataset
R=assets/replicacad
# The downloads are untrusted data: -I never imports from them, and the pipeline only parses their
# URDF/OBJ/MTL/JSON text and copies mesh/texture files.
P="python3 -I $A/urdf_to_gz.py"

# --- Inputs (licence-bound, non-commercial: never commit them or put them in an image) ---
# ReplicaCAD table + cup (8 files, 1.2 MB; CC BY-NC 4.0, credit Meta Habitat 2.0 if the table is in a video).
HF=https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset/resolve/main
for f in LICENSE.txt README.md \
  objects/{frl_apartment_table_01.glb,frl_apartment_cup_01.glb} \
  objects/convex/{frl_apartment_table_01_cv_decomp.glb,frl_apartment_cup_01_cv_decomp.glb} \
  configs/objects/{frl_apartment_table_01,frl_apartment_cup_01}.object_config.json; do
  [[ -f $R/$f ]] && continue
  [[ -n $OFFLINE ]] && { echo "ReplicaCAD file $R/$f missing and --offline"; exit 1; }
  echo "download ReplicaCAD $f"
  curl -fsSL --create-dirs -o "$R/$f.part" "$HF/$f" || { rm -f "$R/$f.part"; echo "download of $HF/$f failed"; exit 1; }
  mv "$R/$f.part" "$R/$f"
done
# PartNet-Mobility 11211 + 46452: gated (SAPIEN terms), no automatic download. Unpacked from
# assets/partnet_mobility/zips/<id>.zip if only the zip is there.
missing=
for id in 11211 46452; do
  [[ -f $D/$id/mobility.urdf ]] && continue
  if [[ -f assets/partnet_mobility/zips/$id.zip ]]; then
    echo "unpack PartNet-Mobility $id.zip"
    python3 -I -c '
import os, sys, zipfile
z, dst, top = zipfile.ZipFile(sys.argv[1]), sys.argv[2], sys.argv[3] + "/"
for n in z.namelist():  # untrusted archive: only plain relative paths under <id>/
    if not n.startswith(top) or os.path.isabs(n) or ".." in n.split("/"):
        sys.exit(f"{sys.argv[1]}: unexpected entry {n!r}")
z.extractall(dst)' "assets/partnet_mobility/zips/$id.zip" "$D" "$id"
  else
    missing+=" $id"
  fi
done
if [[ -n $missing ]]; then
  cat <<MSG
PartNet-Mobility objects missing:$missing. They cannot be downloaded automatically: request access to
PartNet-Mobility (SAPIEN, https://sapien.ucsd.edu; non-commercial research use only), download the objects
and put each as assets/partnet_mobility/zips/<id>.zip or unpacked as $D/<id>/mobility.urdf, then re-run.
MSG
  exit 1
fi

# --- Conversion ---
$P $D/11211/mobility.urdf "$OUT/generated_partnet" --config $C/partnet_fridge.yaml
$P $D/46452/mobility.urdf "$OUT/generated_partnet" --config $C/partnet_cabinet.yaml

G=$OUT/generated
$P $R/configs/objects/frl_apartment_table_01.object_config.json "$G" --config $C/replicacad_table_01.yaml
$P $R/configs/objects/frl_apartment_cup_01.object_config.json "$G" --config $C/replicacad_cup_01.yaml
# Cup for the scene: the ReplicaCAD cup if it fits the Panda gripper, else a
# primitive cylinder cup. Width = smaller horizontal extent of its collisions.
WIDTH=$(python3 -I -c 'import json,sys; a=json.load(open(sys.argv[1]))["aabb_model_frame"]; print(min(a[1][0]-a[0][0], a[1][1]-a[0][1]))' \
  "$G/replicacad_cup_01/manifest.json")
if python3 -I -c "import sys; sys.exit(0 if $WIDTH <= 0.07 else 1)"; then
  echo "cup: ReplicaCAD cup_01 is $WIDTH m wide (<= 0.07): using it"
  $P $R/configs/objects/frl_apartment_cup_01.object_config.json "$G" --config $C/replicacad_cup_01.yaml --name cup
else
  echo "cup: ReplicaCAD cup_01 is $WIDTH m wide (> 0.07): using the primitive cylinder cup"
  $P $A/primitives/cup_cylinder.urdf "$G" --config $C/cup_cylinder.yaml
fi

# The room shell; its CC0 ambientCG textures are downloaded once into assets/textures/.
python3 -I $A/make_room.py --out "$G" $OFFLINE
echo "assets: models in $OUT/generated_partnet and $G"
