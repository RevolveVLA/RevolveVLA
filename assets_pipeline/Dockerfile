# revolve-assets: dependencies of the one-time asset conversion (make_assets.sh), nothing copied in.
# Same Python stack as the image the existing assets were made in (Ubuntu 24.04 apt numpy / scipy / PyYAML, pip
# coacd + trimesh + pillow at the same pins), so regenerated models match them. No Gazebo: the simulator loads
# the models later.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1
# scipy comes from apt so pip leaves the system numpy alone.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl python3 python3-pip python3-numpy python3-scipy python3-yaml \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages coacd==1.0.14 trimesh==4.11.5 pillow==11.3.0 \
    && python3 -c "import coacd, trimesh, numpy, yaml, PIL; print('numpy', numpy.__version__)"
WORKDIR /repo
CMD ["bash"]
