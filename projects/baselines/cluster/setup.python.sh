#!/usr/bin/bash
set -xue
# Install Python packages for this container's version
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [PYTHON_VERSION]"
    exit 1
fi

PYTHON_VERSION=$1
export DEBIAN_FRONTEND=noninteractive;
apt-get update
apt-get install -y gnupg ca-certificates software-properties-common
# Deadsnakes: https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa
# apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776
# # Set up custom sources
# echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" >> /etc/apt/sources.list.d/custom.list
# echo "deb-src http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" >> /etc/apt/sources.list.d/custom.list
add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt-get install -y --no-install-recommends \
    curl \
    $PYTHON_VERSION \
    $PYTHON_VERSION-dev \
    $PYTHON_VERSION-venv \
    $PYTHON_VERSION-distutils

ln -sf /usr/bin/$PYTHON_VERSION /usr/bin/python3
ln -sf /usr/bin/$PYTHON_VERSION /usr/bin/python
rm -rf /var/lib/apt/lists/*

# Install pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py; \
python3 get-pip.py
rm get-pip.py
