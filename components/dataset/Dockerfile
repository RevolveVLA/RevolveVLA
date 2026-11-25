# revolve-dataset: dependencies only for the dataset component (components/dataset/DATASET-v2.md section 4): ROS 2
# Lyrical on Ubuntu 26.04 (Python 3.14), like the simulator image; rosbag2 with MCAP; the rosdep keys of Rosetta's
# packages; LeRobot 0.6.1 + CPU torch from pip. CPU only.
# Nothing of Rosetta is copied in: rosdep reads its package.xml files through a build-time bind mount, and
# `./revolve build dataset` (build.sh) colcon-builds Rosetta INSIDE this image into ThirdParty/rosetta/install
# (git-ignored), mounted read-only at /opt/rosetta/install at runtime (the path it is built at).
#   docker build --build-context rosetta=ThirdParty/rosetta -t revolve-dataset components/dataset
FROM ubuntu:26.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC LANG=C.UTF-8 ROS_DISTRO=lyrical

# --- the ROS 2 apt source (ros-apt-source, as the simulator image) ---
ARG ROS_APT_SOURCE_VERSION=1.3.0
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSL -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.resolute_all.deb" \
    && dpkg -i /tmp/ros2-apt-source.deb && rm /tmp/ros2-apt-source.deb \
    && rm -rf /var/lib/apt/lists/*

# --- ROS 2 Lyrical: ros-base, rosbag2 with the MCAP storage plugin (the bags are MCAP, zstd chunks), colcon and
# rosdep for Rosetta's build; pip for LeRobot; ffmpeg's shared libraries for torchcodec (LeRobot's video decoder) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-lyrical-ros-base ros-lyrical-rosbag2-py ros-lyrical-rosbag2-storage-mcap \
      python3-colcon-common-extensions python3-rosdep python3-pip build-essential cmake git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- the rosdep keys of Rosetta's packages up to lerobot_rosetta (rosetta_interfaces needs rosidl) ---
RUN --mount=type=bind,from=rosetta,target=/tmp/rosetta \
    apt-get update && rosdep init && rosdep update --rosdistro "$ROS_DISTRO" \
    && rosdep install -y --rosdistro "$ROS_DISTRO" --ignore-src \
       --from-paths /tmp/rosetta/rosetta /tmp/rosetta/robots/ros /tmp/rosetta/policies/lerobot \
    && rm -rf /var/lib/apt/lists/* /root/.ros

# --- LeRobot 0.6.1 + CPU torch 2.11 and revolve_common's packages: requirements.txt, generated from
# requirements.in + overrides.txt (numpy 2.3.5: lerobot 0.6.1 pins numpy < 2.3, which has no cp314 wheel).
# --no-deps: the lock is complete; --ignore-installed: never uninstall apt's Python packages. ---
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages --ignore-installed --no-deps -r /tmp/requirements.txt \
      --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    && rm /tmp/requirements.txt && mkdir -p /opt/rosetta && pip freeze > /opt/rosetta/pip-freeze.txt \
    && python3 -c "import cv2, lerobot, numpy, torch, torchcodec, pydantic, fastapi, uvicorn; print('lerobot', lerobot.__version__, 'torch', torch.__version__, 'numpy', numpy.__version__, 'cv2', cv2.__version__)"

ENV PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
WORKDIR /repo
CMD ["bash"]
