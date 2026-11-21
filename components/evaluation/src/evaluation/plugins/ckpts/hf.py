"""Checkpoint on the Hugging Face Hub: `ckpt: {source: hf, repo: org/name, revision: main, subfolder: a/b}`.

Downloaded once into `runs/ckpt_cache/hf/<org>__<name>/<commit sha>/` (only `subfolder/*` when given) and reused without
network when the commit is pinned (`revision` = a full sha). The token (private repos) is only what huggingface_hub reads
itself ($HF_TOKEN / its login store): it is never passed on, logged or written here. huggingface_hub is imported lazily."""
from __future__ import annotations

import re
from pathlib import Path

from ...core import sets
from ...core.errors import EvalError
from ...core.interfaces import CkptSource, ResolvedCkpt
from ...core.registry import REGISTRY
from .local import origin_of

REPO = re.compile(r"[\w.-]+/[\w.-]+")
REVISION = re.compile(r"\w[\w./-]*")
SHA = re.compile(r"[0-9a-f]{40}")


@REGISTRY.register("ckpt", "hf")
class HfCkpt(CkptSource):
    def __init__(self, spec, params):
        if set(params) - {"repo", "revision", "subfolder"} or "repo" not in params:
            raise EvalError(f"ckpt hf: needs `repo`, optional `revision`, `subfolder` (got {sorted(params)})")
        self.spec, self.repo = spec, str(params["repo"])
        self.revision, self.subfolder = str(params.get("revision", "main")), str(params.get("subfolder", "")).strip("/")
        if not REPO.fullmatch(self.repo):
            raise EvalError(f"ckpt hf: repo {self.repo!r} is not org/name")
        if not REVISION.fullmatch(self.revision) or ".." in self.revision:
            raise EvalError(f"ckpt hf: invalid revision {self.revision!r}")
        if ".." in Path(self.subfolder).parts:
            raise EvalError(f"ckpt hf: subfolder {self.subfolder!r} must stay inside the repo")

    def resolve(self, out):
        root = sets.repo_root(self.spec, out)
        commit = self.revision if SHA.fullmatch(self.revision) else self._commit()
        cache = root / "runs" / "ckpt_cache" / "hf" / self.repo.replace("/", "__") / commit
        done = cache / f".complete_{self.subfolder.replace('/', '__') or 'all'}"
        if not done.is_file():
            self._download(cache, commit)
            done.touch()
        ckpt = cache / self.subfolder
        if not (ckpt / "config.json").is_file():
            raise EvalError(f"ckpt hf: {self.repo}@{commit} has no config.json in {self.subfolder or 'its root'!r}")
        return ResolvedCkpt(str(ckpt), origin_of(ckpt, root, source="hf", repo=self.repo, revision=self.revision,
                                                 commit=commit, subfolder=self.subfolder or None))

    def _commit(self) -> str:
        from huggingface_hub import HfApi
        return HfApi().model_info(self.repo, revision=self.revision).sha

    def _download(self, cache: Path, commit: str):
        from huggingface_hub import snapshot_download
        cache.mkdir(parents=True, exist_ok=True)
        snapshot_download(self.repo, revision=commit, local_dir=cache,
                          allow_patterns=[f"{self.subfolder}/*"] if self.subfolder else None)
