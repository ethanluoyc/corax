#!/usr/bin/env bash
set -eux

uv pip compile requirements/test.in -o requirements/test.txt
uv pip compile requirements/dev.in -o requirements/dev.txt
uv pip compile --all-extras \
    pyproject.toml \
    projects/baselines/requirements.in \
    --emit-find-links \
    -o projects/baselines/requirements.txt
