# revolve-simulator: dependencies only, Ubuntu 26.04 (resolute) + ROS 2 Lyrical binaries + the apt build deps of
# Gazebo "Kura" (gz-sim 11 / gz-physics 10 main, with the MuJoCo physics plugin).
# Nothing is built or copied in: `./revolve build simulator` builds, with colcon INSIDE this image,
#   1. Kura from ThirdParty/gz_kura into ThirdParty/gz_kura/install, mounted at /opt/gz_ws/install,
#   2. ros_gz + gz_ros2_control (+ the gz_*_vendor packages, satisfied by Kura) from ThirdParty into
#      ThirdParty/ros_gz_ws/install, mounted at /opt/ros_gz_ws/install
# (both git-ignored, mounted at the paths they are built at: gz and colcon embed absolute install paths).
# The Lyrical binaries ros-lyrical-ros-gz* / gz-ros2-control* are built against Gazebo Jetty: never install them.
# The binary set below still pulls the Jetty sdformat / gz-math / gz-utils / gz-tools vendors into /opt/ros/lyrical,
# so Kura goes first on every path (ENV below and build.sh).
FROM ubuntu:26.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC LANG=C.UTF-8

# --- Base tools + apt sources: OSRF (DART, OGRE-Next, Zenoh for Kura) and ROS 2 (ros-apt-source, as the install
# guide does) ---
ARG ROS_APT_SOURCE_VERSION=1.3.0
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg lsb-release git \
    && curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
         -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable resolute main" \
         > /etc/apt/sources.list.d/gazebo-stable.list \
    && curl -fsSL -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.resolute_all.deb" \
    && dpkg -i /tmp/ros2-apt-source.deb && rm /tmp/ros2-apt-source.deb \
    && rm -rf /var/lib/apt/lists/*

# --- Kura build dependencies: union of packages[-noble].apt across the Kura repos (same names on resolute) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      binutils-dev build-essential cmake cppzmq-dev freeglut3-dev libassimp-dev \
      libavcodec-dev libavdevice-dev libavformat-dev libavutil-dev libbenchmark-dev \
      libcli11-dev libcurl4-openssl-dev libdart6.16-collision-bullet-dev \
      libdart6.16-collision-ode-dev libdart6.16-dev libdart6.16-external-ikfast-dev \
      libdart6.16-external-odelcpsolver-dev libdart6.16-utils-urdf-dev libdwarf-dev \
      libdw-dev libeigen3-dev libfreeimage-dev libgdal-dev libgflags-dev libgl1-mesa-dev \
      libglew-dev libglu1-mesa-dev libjsoncpp-dev libogre-1.9-dev libogre-next-2.3-dev \
      libprotobuf-dev libprotoc-dev libpython3-dev libqt6svg6 libspdlog-dev \
      libsqlite3-dev libswscale-dev libtinyxml2-dev liburdfdom-dev libvulkan-dev \
      libwebsockets-dev libxi-dev libxml2-utils libxmu-dev libyaml-dev libzenohc-dev \
      libzenohcpp-dev libzip-dev libzmq3-dev mesa-utils pkg-config protobuf-compiler \
      python3-dev python3-numpy python3-protobuf python3-psutil python3-pybind11 \
      python3-pytest python3-yaml qml6-module-qt5compat-graphicaleffects qml6-module-qtcharts \
      qml6-module-qtcore qml6-module-qt-labs-folderlistmodel qml6-module-qt-labs-platform \
      qml6-module-qt-labs-settings qml6-module-qtpositioning qml6-module-qtqml \
      qml6-module-qtqml-models qml6-module-qtqml-workerscript qml6-module-qtquick \
      qml6-module-qtquick-controls qml6-module-qtquick-dialogs qml6-module-qtquick-layouts \
      qml6-module-qtquick-templates qml6-module-qtquick-window qt6-5compat-dev \
      qt6-base-dev qt6-base-private-dev qt6-declarative-dev qt6-declarative-private-dev \
      ruby ruby-dev swig uuid-dev x11-utils xvfb \
      libegl1 libgl1 libglvnd0 libglx0 ffmpeg \
      ccache python3-colcon-common-extensions python3-vcstool python3-pip \
    && rm -rf /var/lib/apt/lists/*

# --- ROS 2 Lyrical: ros2_control + controllers, MoveIt 2 (+ moveit_py, the Panda resources), xacro,
# robot_state_publisher, rosbag2 (MCAP), launch_testing; plus the rosdep keys of ros_gz / gz_ros2_control
# (+ demos) and ament_cmake_vendor_package for the gz_*_vendor packages ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-lyrical-ros-base ros-lyrical-ros2-control ros-lyrical-ros2-controllers \
      ros-lyrical-xacro ros-lyrical-robot-state-publisher \
      ros-lyrical-moveit ros-lyrical-moveit-py \
      ros-lyrical-moveit-resources-panda-description ros-lyrical-moveit-resources-panda-moveit-config \
      ros-lyrical-rosbag2 ros-lyrical-rosbag2-storage-mcap \
      ros-lyrical-launch-testing ros-lyrical-launch-testing-ament-cmake ros-lyrical-launch-pytest \
      ros-lyrical-ament-cmake ros-lyrical-ament-cmake-python ros-lyrical-ament-cmake-vendor-package \
      ros-lyrical-actuator-msgs ros-lyrical-gps-msgs ros-lyrical-vision-msgs ros-lyrical-marine-acoustic-msgs \
      ros-lyrical-simulation-interfaces ros-lyrical-image-transport ros-lyrical-image-transport-plugins \
      ros-lyrical-control-toolbox ros-lyrical-ros2-control-cmake ros-lyrical-ros2controlcli \
      ros-lyrical-spdlog-vendor ros-lyrical-yaml-cpp-vendor \
    && rm -rf /var/lib/apt/lists/* \
    && bash -c "source /opt/ros/lyrical/setup.bash && python3 -c 'import rclpy, moveit.planning'"

# --- GPU rendering: the NVIDIA EGL vendor (glvnd falls back to Mesa when the driver is not mounted) ---
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}\n' \
         > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Kura first on every path; /opt/ros/lyrical and the ros_gz workspace come after it via /etc/bash.bashrc.
ENV GZ_WS=/opt/gz_ws/install ROS_GZ_WS=/opt/ros_gz_ws/install ROS_DISTRO=lyrical
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    QT_QPA_PLATFORM=xcb \
    PYTHONDONTWRITEBYTECODE=1 \
    BASH_ENV=/opt/revolve_env.sh

# Sourced by every bash (interactive via bash.bashrc, `bash -c` via BASH_ENV): ROS, then the ros_gz workspace,
# then Kura prepended again so it wins over the Jetty vendors in /opt/ros/lyrical.
RUN printf '%s\n' \
      '[ -n "${REVOLVE_ENV_DONE:-}" ] && return 0; export REVOLVE_ENV_DONE=1' \
      'source /opt/ros/lyrical/setup.bash' \
      '[ -f $ROS_GZ_WS/setup.bash ] && source $ROS_GZ_WS/setup.bash' \
      'export PATH=$GZ_WS/bin:$PATH LD_LIBRARY_PATH=$GZ_WS/lib:$LD_LIBRARY_PATH' \
      'export PYTHONPATH=$GZ_WS/lib/python:$PYTHONPATH CMAKE_PREFIX_PATH=$GZ_WS:$CMAKE_PREFIX_PATH' \
      'export GZ_CONFIG_PATH=$GZ_WS/share/gz${GZ_CONFIG_PATH:+:$GZ_CONFIG_PATH}' \
      > /opt/revolve_env.sh \
    && echo 'source /opt/revolve_env.sh' >> /etc/bash.bashrc
WORKDIR /repo
CMD ["bash"]
