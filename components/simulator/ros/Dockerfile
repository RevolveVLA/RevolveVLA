# revolve-simulator-ros: revolve-simulator plus ROS 2 Jazzy ros-base (rclpy, sensor_msgs, rosgraph_msgs) for the
# policy rollout's ROS 2 <-> gz-transport relay (src/sim/ros_relay.py) beside gz. Dependencies only, nothing
# built or copied in. Ported from the earlier pipeline's ROS Dockerfile (same ROS apt source and
# packages; python3-pil dropped: the base image has pillow). ros-base pulls no gz package, so Gazebo stays the
# Kura build mounted at /opt/gz_ws/install. ROS is sourced only for the relay process, never for gz.
# Built by `./revolve build evaluation` (components/evaluation/build.sh).
ARG BASE=revolve-simulator:latest
FROM ${BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The ROS apt source package (ros-infrastructure/ros-apt-source), as the Jazzy install guide does
ARG ROS_APT_SOURCE_VERSION=1.3.0
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.noble_all.deb" \
    && dpkg -i /tmp/ros2-apt-source.deb && rm /tmp/ros2-apt-source.deb \
    && apt-get update && apt-get install -y --no-install-recommends ros-jazzy-ros-base \
    && rm -rf /var/lib/apt/lists/* \
    && bash -c "source /opt/ros/jazzy/setup.bash && python3 -c 'import rclpy, sensor_msgs.msg, rosgraph_msgs.msg, PIL'"
