#!/usr/bin/env bash
# Fetch Gazebo Kura sources at the pins in kura.repos into src/ (git-ignored) and apply
# ThirdParty/patches/<repo>/*.patch once each.
#   ./fetch.sh          import (skipped if src/gz-physics exists) + patches
#   ./fetch.sh --check  only check that every pinned sha can be fetched (depth 1, no blobs; nothing kept).
#                       `vcs validate` cannot verify a bare sha without cloning (vcstool 0.3.0 also crashes on it).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
patches=$here/../patches
if [[ "${1:-}" == "--check" ]]; then
  mkdir -p "$here/src"; tmp=$(mktemp -d -p "$here/src" .check.XXXXXX); trap 'rm -rf "$tmp"' EXIT; bad=0
  while read -r name url sha; do
    git init -q "$tmp/$name"
    if git -C "$tmp/$name" fetch -q --depth 1 --filter=blob:none "$url" "$sha" 2>/dev/null; then
      echo "ok      $name $sha"
    else
      echo "MISSING $name $sha ($url)"; bad=1
    fi
  done < <(python3 -c 'import sys, yaml
for n, r in yaml.safe_load(open(sys.argv[1]))["repositories"].items(): print(n, r["url"], r["version"])' "$here/kura.repos")
  exit $bad
fi
mkdir -p "$here/src"
if [[ ! -d "$here/src/gz-physics" ]]; then
  vcs import "$here/src" < "$here/kura.repos"
fi
for dir in "$patches"/*/; do
  repo=$(basename "$dir")
  [[ -d "$here/src/$repo" ]] || continue
  for p in "$dir"*.patch; do
    [[ -e "$p" ]] || continue
    if git -C "$here/src/$repo" apply --reverse --check "$p" 2>/dev/null; then
      echo "patch already applied: $repo/$(basename "$p")"
    else
      git -C "$here/src/$repo" apply "$p" && echo "applied patch: $repo/$(basename "$p")"
    fi
  done
done
