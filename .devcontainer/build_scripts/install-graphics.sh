#!/usr/bin/bash
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -q -y --no-install-recommends \
    libegl1 \
    libgl1 \
    libgl1-mesa-glx \
    libgles2 \
    libglew2.2 \
    libglfw3 \
    libglvnd0 \
    libglx0 \
    libosmesa6

apt-get clean
rm -rf /var/lib/apt/lists/*

MY_DIR=$(dirname "$0")
cp $MY_DIR/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
chmod 777 /usr/share/glvnd/egl_vendor.d/10_nvidia.json
