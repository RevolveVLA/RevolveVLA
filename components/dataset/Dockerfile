# revolve-dataset: dependencies only for Rosetta's offline porter (MCAP bags -> LeRobot v3.0 dataset) and
# LeRobot 0.6.1's dataset tools (load check, aggregate), CPU only.
# Ported from the earlier pipeline's dataset Dockerfile (automoma-lerobot; the recipe of Rosetta's lerobot CI):
# ros:jazzy-ros-base, rosdep over rosetta/, robots/ros/ and policies/lerobot/, pip `lerobot[dataset]==0.6.1`
# + CPU torch with --break-system-packages --ignore-installed, protobuf/grpcio upgraded. Pins unchanged.
# What changed: Rosetta is NOT fetched or built here. rosdep reads its package.xml files from the
# ThirdParty/rosetta submodule through a build-time bind mount (nothing copied into the image);
# `./revolve build dataset` colcon-builds Rosetta INSIDE this image into ThirdParty/rosetta/install
# (git-ignored), mounted at /opt/rosetta/install at runtime (the path it is built at). Plus the
# revolve_common deps (job wrapper, spec loader). LeRobot 0.6.2 drops the async inference Rosetta's
# policy runner uses (evaluation runs that runner on this image).
#   docker build --build-context rosetta=ThirdParty/rosetta -t revolve-dataset components/dataset
FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
# pip for LeRobot; ffmpeg's shared libraries for torchcodec (LeRobot's video decoder)
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,from=rosetta,target=/tmp/rosetta \
    apt-get update && rosdep update --rosdistro "$ROS_DISTRO" \
    && rosdep install -y --rosdistro "$ROS_DISTRO" --ignore-src \
       --from-paths /tmp/rosetta/rosetta /tmp/rosetta/robots/ros /tmp/rosetta/policies/lerobot \
    && rm -rf /var/lib/apt/lists/*

# lerobot and torch have no rosdep keys. Pinned inside lerobot 0.6.1's ranges (torch < 2.12,
# torchvision < 0.27, torchcodec < 0.12); torchcodec 0.11 is the build for torch 2.11.
RUN pip install --no-cache-dir --break-system-packages --ignore-installed \
      'lerobot[dataset]==0.6.1' torch==2.11.0 torchvision==0.26.0 torchcodec==0.11.1 \
      --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    && pip install --no-cache-dir --break-system-packages --ignore-installed --upgrade protobuf grpcio \
    && mkdir -p /opt/rosetta && pip freeze > /opt/rosetta/pip-freeze.txt

# revolve_common (job wrapper, spec loader): the host .venv's pins.
RUN pip install --no-cache-dir --break-system-packages pydantic==2.14.3 fastapi==0.162.2 uvicorn==0.54.0 \
    && python3 -c "import pydantic, fastapi, uvicorn, lerobot, torch; print('pydantic', pydantic.VERSION, 'lerobot', lerobot.__version__)"

ENV PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
WORKDIR /repo
CMD ["bash"]
