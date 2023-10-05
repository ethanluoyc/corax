FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as builder

ENV PYTHON_VERSION=python3.9
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    patchelf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -q https://gitlab.com/nvidia/container-images/opengl/-/raw/ubuntu20.04/glvnd/runtime/10_nvidia.json \
    -o /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install MuJoCo
ENV MUJOCO_DIR=/opt/mujoco
ENV MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MUJOCO_PY_MUJOCO_PATH}/bin
RUN mkdir -p ${MUJOCO_DIR} && \
    curl -O -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
    tar -C ${MUJOCO_DIR} -xvzf mujoco210-linux-x86_64.tar.gz && \
    rm mujoco210-linux-x86_64.tar.gz

COPY cluster/setup.python.sh /setup.python.sh
RUN /setup.python.sh ${PYTHON_VERSION}

COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -U pip wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base
ENV PYTHON_VERSION=python3.9
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
    cuda-nvcc-11-8 \
    curl \
    git \
    ffmpeg \
    libegl1 \
    libgl1 \
    libgl1-mesa-glx \
    libgles2 \
    libglew2.2 \
    libglfw3 \
    libglvnd0 \
    libglx0 \
    libosmesa6 \
    # Extra deps for downstream
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup Python
COPY --from=builder --chown=655 /usr/share/glvnd/egl_vendor.d/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY --from=builder /opt/mujoco /opt/mujoco
COPY cluster/setup.python.sh /setup.python.sh
RUN /setup.python.sh ${PYTHON_VERSION}

COPY --from=builder /opt/venv /opt/venv
# Needed by for GPU rendering, if used
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV MUJOCO_DIR=/opt/mujoco
ENV MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MUJOCO_PY_MUJOCO_PATH}/bin
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
