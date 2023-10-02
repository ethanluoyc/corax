N_CPUS ?= $(shell grep -c ^processor /proc/cpuinfo)
SRCS ?= corax

lint:
	ruff .
	black --check --diff .

fmt:
	ruff --fix .
	black .

test:
	CUDA_VISIBLE_DEVICES='' JAX_DISBLAE_MOST_OPTIMIZATIONS=1 \
	pytest -n $(N_CPUS) -rf --durations=10 $(SRCS)

.PHONY: lint fmt
