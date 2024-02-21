#!/usr/bin/env bash
set -eux

uv pip compile requirements/base.in -o requirements/base.txt
uv pip compile requirements/test.in -o requirements/test.txt
uv pip compile requirements/dev.in -o requirements/dev.txt
uv pip compile requirements/baselines.in --emit-find-links -o requirements/baselines.txt
