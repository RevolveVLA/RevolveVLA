# revolve-training: dependencies only for SmolVLA training (LeRobot 0.6.1's lerobot-train) and serving
# (Rosetta's lerobot_rosetta.policy_server over LeRobot 0.6.1's async server). One image for both: the serve
# lock (requirements.lock here, copied from the earlier pipeline's serve lock) is the training
# lock plus the async extra at the same versions and hashes.
# Ported from the legacy policy setup: the earlier pipeline's venv recipe (Machine.ensure_venv) /
# evaluate.py ensure_serve_venv (uv-managed Python 3.12.12, `uv pip sync --require-hashes <lock>`) and the
# its Modal image (debian slim + ffmpeg for torchcodec + uv 0.9.15). PyPI torch
# 2.11.0 is the CUDA 13.0 build: the CUDA runtime comes from its nvidia-* wheels (host driver >= 580).
# Plus: the revolve_common deps (job wrapper), the Modal client (training backend modal), openssh-client and
# rsync (training backend gpu_machine), all resolved with the lock as a constraint (no locked pin changes).
# Rosetta is not here: serve mounts ThirdParty/rosetta's lerobot_rosetta sources (pure Python) read-only.
#   docker build -t revolve-training components/training
FROM debian:bookworm-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg openssh-client rsync ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.15 /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_PYTHON_PREFERENCE=only-managed UV_NO_CONFIG=1 UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
COPY requirements.lock /opt/requirements.lock
RUN uv venv --python 3.12.12 /opt/venv \
    && uv pip sync --no-cache --require-hashes /opt/requirements.lock \
    && chmod -R a+rX /opt/uv-python /opt/venv

RUN grep -oE '^[A-Za-z0-9_.-]+==[^ ]+' /opt/requirements.lock > /tmp/constraints.txt \
    && uv pip install --no-cache -c /tmp/constraints.txt pydantic==2.14.3 fastapi==0.162.2 uvicorn==0.54.0 modal==1.6.3 \
    && rm /tmp/constraints.txt && uv pip freeze > /opt/pip-freeze.txt \
    && python -c "import lerobot, torch, pydantic, fastapi, modal; print('lerobot', lerobot.__version__, 'torch', torch.__version__, torch.version.cuda, 'modal', modal.__version__)"

ENV PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /repo
CMD ["bash"]
