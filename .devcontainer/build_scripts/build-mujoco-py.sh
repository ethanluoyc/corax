#!/usr/bin/bash
# Build mujoco-py from source
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

MY_DIR=$(dirname "$0")

mkdir -p /opt/mujoco-py
curl -o mujoco-py.tar.gz -L https://github.com/openai/mujoco-py/archive/refs/tags/v2.1.2.14.tar.gz
tar --strip-components 1 -xzf mujoco-py.tar.gz -C /opt/mujoco-py
rm mujoco-py.tar.gz

patch -d /opt/mujoco-py -p1 < $MY_DIR/mujoco-py.patch

apt-get update
apt install -y --no-install-recommends libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

cd /opt/mujoco-py
pip install build
python3 -m build -w .
