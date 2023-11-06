#!/bin/bash
# Install mujoco library

set -e

## Parse command-line arguments

usage() {
    echo "Install Mujoco"
    echo "  Usage: $0 [OPTIONS]"
    echo ""
    echo "    OPTIONS                                  DESCRIPTION"
    echo "    --mujoco-version    MUJOCO_VERSION       Version of mujoco to install. Default: mjpro150"
    echo "    --mujoco-dir        MUJOCO_DIR           Directory to install."
    exit $1
}

# Set defaults
CPU_ARCH="$(dpkg --print-architecture)"
DRY=0
MUJOCO_VERSION="mujoco210"
MUJOCO_DIR=""

args=$(getopt -o h --long mujoco-version:,mujoco-dir: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "$args"
while [ : ]; do
    case "$1" in
        --mujoco-dir)
            MUJOCO_DIR=$2
            shift 2;
            ;;
        --mujoco-version)
            MUJOCO_VERSION=$2
            shift 2;
            ;;
        --)
            shift;
            break
            ;;
        *)
            echo "UNKNOWN OPTION $1"
            usage 1
    esac
done

if [[ -z "$MUJOCO_DIR" ]]; then
    echo "ERROR: Must specify --mujoco-dir"
    usage
fi

mkdir -p ${MUJOCO_DIR}

function install_mjpro150() {
    echo "Installing mujoco $MUJOCO_VERSION"
    MJPRO_150_URL=https://www.roboti.us/download/mjpro150_linux.zip
    MJPRO_150_FILENAME=$(basename $MJPRO_150_URL)
    curl -O -L $MJPRO_150_URL
    unzip $MJPRO_150_FILENAME -d ${MUJOCO_DIR}
    rm $MJPRO_150_FILENAME

    # Free license key after DeepMind acquisition
    echo "Installing free license key"
    MUJOCO_LICENSE_KEY_URL=https://www.roboti.us/file/mjkey.txt
    curl -L $MUJOCO_LICENSE_KEY_URL -o ${MUJOCO_DIR}/mjpro150/bin/mjkey.txt
}

function install_mujoco210() {
    echo "Installing mujoco $MUJOCO_VERSION"
    MUJOCO_210_URL=https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
    curl -O -L $MUJOCO_210_URL
    curl -O -L $MUJOCO_210_URL.sha256
    echo "Verifying SHA256 checksum"
    sha256sum -c $(basename $MUJOCO_210_URL).sha256
    tar -C ${MUJOCO_DIR} -xzf $(basename $MUJOCO_210_URL)
    rm $(basename $MUJOCO_210_URL)*
}

case $MUJOCO_VERSION in
    mjpro150)
        install_mjpro150
        ;;
    mujoco210)
        install_mujoco210
        ;;
    *)
        echo "ERROR Unknown mujoco version $MUJOCO_VERSION"
        exit 1
        ;;

esac
