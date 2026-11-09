# revolve-datagen: dependencies only for AutoMoMa's plan step (cuRobo motion generation), no Isaac Sim.
# CUDA 12.8 devel (nvcc, for building cuRobo's extensions) + uv-managed Python 3.11 (AutoMoMa needs
# exactly 3.11) + PyTorch 2.7.0 cu128 + the plan-step pip deps + the revolve_common deps.
# Nothing is built or copied in: `./revolve build datagen` builds cuRobo (CUDA extensions, for
# TORCH_CUDA_ARCH_LIST) INSIDE this image from ThirdParty/curobo into ThirdParty/curobo/install
# (git-ignored), mounted at /opt/curobo/install at runtime; AutoMoMa (pure Python) is mounted from
# ThirdParty/automoma at /opt/automoma and put on PYTHONPATH (what its editable install did).
# Ported from the earlier pipeline's planner Dockerfile: every layer up to the plan-step deps is unchanged
# (so they come from the build cache of automoma-planner); the cuRobo/AutoMoMa fetch + compile layers
# are gone.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
# Python outside /root so the container also runs as the host user.
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_PYTHON_PREFERENCE=only-managed UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN uv venv --python 3.11 /opt/venv

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Plan-step deps, pinned as in AutoMoMa's docker/requirements-automoma.txt, plus the
# build tools cuRobo needs without build isolation (setuptools < 81 per AutoMoMa).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install numpy==1.26.0 scipy==1.15.3 PyYAML==6.0.2 h5py==3.16.0 omegaconf==2.3.0 \
      yourdfpy==0.0.60 scikit-learn==1.8.0 scikit-image==0.26.0 numpy-quaternion==2024.0.13 \
      usd-core==26.5 warp-lang==1.12.1 tqdm \
      "setuptools>=71,<81" "setuptools-scm>=9.2.0" wheel ninja tomli

# revolve_common (job wrapper, spec loader): the host .venv's pins.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install pydantic==2.14.3 fastapi==0.162.2 uvicorn==0.54.0 \
    && python -c "import pydantic, fastapi, uvicorn, torch, warp, pxr; print('pydantic', pydantic.VERSION)"

# GPU arch the cuRobo build (inside this image) compiles for: 12.0 = RTX 5060 (sm_120, this laptop),
# 8.9 = Modal L4/RTX 4050 (sm_89). No GPU during docker build or the cuRobo build, so it is fixed here.
ARG TORCH_CUDA_ARCH_LIST=12.0
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /repo
CMD ["bash"]
