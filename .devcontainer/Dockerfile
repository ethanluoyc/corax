# syntax=docker/dockerfile:1
ARG base_image=python:3.10-bookworm
FROM $base_image as base

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -q -y --no-install-recommends \
    curl git ffmpeg \
    libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

## Install EGL & OSMesa runtime libraries
RUN mkdir -p /build_scripts
ADD ./build_scripts/install-graphics.sh /build_scripts
ADD ./build_scripts/10_nvidia.json /build_scripts
# Enable graphics capabilities
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
RUN /build_scripts/install-graphics.sh

## Install MuJoCo
ENV MUJOCO_DIR=/opt/mujoco
ENV MUJOCO_VERSION=mujoco210
ENV MUJOCO_PY_MUJOCO_PATH=${MUJOCO_DIR}/${MUJOCO_VERSION}
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MUJOCO_PY_MUJOCO_PATH}/bin
ADD ./build_scripts/install-mujoco.sh /build_scripts
RUN /build_scripts/install-mujoco.sh --mujoco-dir ${MUJOCO_DIR} --mujoco-version ${MUJOCO_VERSION}


# FROM runtime as mujoco-py-builder
# Build mujoco-py from source
# ADD ./build_scripts/build-mujoco-py.sh /build_scripts
# ADD ./build_scripts/mujoco-py.patch /build_scripts
# RUN /build_scripts/build-mujoco-py.sh

# FROM runtime
# COPY --from=mujoco-py-builder /opt/mujoco-py/dist /opt/mujoco-py/dist
# # Install Python dependencies
# COPY ./test-requirements.txt /build_scripts/
# RUN python3 -m pip install --no-cache-dir \
#     /opt/mujoco-py/dist/*.whl \
#     -r /build_scripts/test-requirements.txt && \
#     rm -rf /opt/mujoco-py/dist
