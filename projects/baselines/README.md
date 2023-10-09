# Corax baselines

This package contains examples and baseline agents developed with Corax.
It provides a starting point for developing new agents and running baseline experiments.

The code is organized such that each agent resides in a separate directory with a python
binary entrypoint (e.g. `main.py`) and configuration files (via ml_collections).

## Running with local virtual environment
You should install the dependencies needed for the baselines in addition to the corax
by running
```bash
pip install -r requirements.txt -e .
```

## Running with Singularity
For production experiments on a cluster, we recommend using Singularity containers.
You can create a Singularity container that installs all of the runtime dependencies by
running the following command from this directory:

```bash
DOCKER_BUILDKIT=1 docker build --squash --tag corax:latest -f cluster/Dockerfile .
singularity build cluster/corax-latest.sif docker-daemon://corax:latest
```

Then you can spawn a shell and run the examples in the package using the singularity container with
```bash
# Create a shell and add corax and baselines to PYTHONPATH
singularity shell --nv --env PYTHONPATH=$PWD:$PWD/../../ cluster/corax-latest.sif
```

## Launching with LXM3
[LXM3](https://github.com/ethanluoyc/lxm3) users, please refer to cluster/launcher.py
for an example of how to launch experiments with LXM3. You may need to modify the
launcher to suit your needs.

## Recommendation for downstream users.
For downstream users, we recommend performing the following steps to
bootstrap your project. These guidelines are not only useful for corax but also
generally good practices for reproducible research.

1. Fork this package into your own repository.
2. Update the requirements.txt to reflect your project's dependencies. In particular,
consider adding `corax[jax,tf]` to the requirements.txt.
3. To prevent supply-chain attacks and ensure reproducibility, we recommend that you
both pin the version of Corax as well as other dependencies. You may want to consider
using [pip-tools](https://pip-tools.readthedocs.io/en/latest/) for creating a pinned
requirements including all of the transitive dependencies. Note that if your project
depends on a git repository, pip-tools will not be hermetic if you specify the dependency
as use a branch. In this case, you should consider either pinning the git repo with the
exact commit hash or the source code archive. For example, to pin Corax to the commit
[86bd13](https://github.com/ethanluoyc/corax/commit/86bd1373d7818cda7a48183c73d5cc16014e87a7)
use either
```
corax @ https://github.com/ethanluoyc/corax/archive/86bd1373d7818cda7a48183c73d5cc16014e87a7.zip
```
or
```
corax @ git+https://github.com/ethanluoyc/corax@86bd1373d7818cda7a48183c73d5cc16014e87a7
```
in your `requirements.in`.
4. If you use containers like Singularity/Docker, you should use your lock file for installing
the exact verisons.
