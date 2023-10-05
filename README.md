# Corax: Core RL in JAX
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![test](https://github.com/ethanluoyc/corax/actions/workflows/test.yml/badge.svg)](https://github.com/ethanluoyc/corax/actions/workflows/test.yml)

**[Installation](#installation)** |
**[Examples](#examples)** |
**[Agents](#agents)** |
**[Datasets](#datasets)**

Corax is a library for reinforcement learning algorithms in JAX. It aims at providing
modular, pure and functional components for RL algorithms that can be easily used in
different training loops and accelerator configurations. Currently, we are exploring the
design of a LearnerCore and ActorCore design that allows easy composition and scaling of
RL algorithms. At the same time, Corax aims to provide strong baseline agents that can
be forked and customized for future RL research.

Corax starts as a fork of the
[dm-acme](https://github.com/google-deepmind/acme/tree/master) library while aiming to
provide a better experience for researchers working on Online/offline RL in JAX. Future
development of Corax may diverge from the design in Acme.

## Installation
You can install Corax with
```
pip install 'git+https://github.com/ethanluoyc/corax#egg=corax[tf,jax]'
```

Note that to run Corax with GPU, you need to install JAX with GPU support. Follow
the instructions [here](https://jax.readthedocs.io/en/latest/installation.html) for how
to install JAX with GPU support.

## Examples
Examples can be found in [projects](projects/).

## Development

```bash
git clone https://github.com/ethanluoyc/jax
cd corax
# Create a virtual environment with the method of your choice.
python3 -m venv .venv
source .venv/bin/activate
# Then run
pip install -e '.[dev]'
# Install pre-commit hooks if you intend to create PRs.
pre-commit install
# Install the baselines by running
pip install -r projects/baselines/requirements.txt -e projects/baselines
```

## Agents
Corax includes high-quality implementation of many popular RL agents. These agents are
meant to be forked and customized for future RL research.

The implementation has been used in numerous research projects and we intend to provide
benchmark results for these agents in the future.

Corax currently implements the following agents JAX:

| Agent                | Paper                    | Code                                                           |
|----------------------|--------------------------|----------------------------------------------------------------|
| CalQL                | [Nakamoto et al., 2023]  | [calql](corax/agents/jax/calql/)                               |
| CQL                  | [Kumar et al., 2020]     | [calql](corax/agents/jax/calql/)                               |
| IQL                  | [Kostrikov et al., 2021] | [iql](corax/agents/jax/iql/)                                   |
| RLPD                 | [Ball et al., 2023]      | [redq](corax/agents/jax/redq/)                                 |
| Decision Transformer | [Chen et al., 2021a]     | [decision_transformer](corax/agents/jax/decision_transformer/) |
| DrQ-v2(-BC)          | [Yarats et al., 2021]    | [drq_v2](corax/agents/jax/drq_v2/)                             |
| ORIL                 | [Zolna et al., 2020]     | [oril](corax/agents/jax/oril/)                                 |
| OTR                  | [Luo et al., 2023]       | [otr](corax/agents/jax/otr/)                                   |
| REDQ                 | [Chen et al., 2021b]     | [redq](corax/agents/jax/redq/)                                 |
| TD3                  | [Fujimoto et al., 2018]  | [td3](corax/agents/jax/td3/)                                   |
| TD3-BC               | [Fujimoto et al., 2021]  | [td3](corax/agents/jax/td3/)                                   |
| TD-MPC               | [Hansen et al., 2021]    | [tdmpc](corax/agents/jax/tdmpc/)                               |

More agents, including those implemented in
[Magi](https://github.com/ethanluoyc/magi/tree/main/magi) may be added in the future.
Contributions to include new agents are welcome!

## Datasets
For online RL, Corax uses [Reverb](https://github.com/google-deepmind/reverb/) for online RL agents.

When working with offline RL, existing datasets provided by the community may come in
different formats. It can be time-consuming to integrate existing algorithms with
different datasets.

Therefore, for offline RL, Corax provides additional
[TFDS](https://github.com/tensorflow/datasets/tree/master) dataset builders that can
build datasets stored in [RLDS](https://github.com/google-research/rlds) format. This
allows easily running the same offline RL algorithm on offline RL datasets in a
consistent manner. You may want to check out the
[list](https://github.com/google-research/rlds/tree/main#available-datasets) of datasets
officially supported by the TFDS/RLDS team.

In addition to the official RLDS datasets, the following datasets can be built with
Corax:

| Dataset         | Paper                    | Code                                                |
|-----------------|--------------------------|-----------------------------------------------------|
| V-D4RL          | [Lu et al., 2023]        | [vd4rl](corax/datasets/tfds/vd4rl/)                 |
| Watch and Match | [Haldar et al., 2022]    | [rot](corax/datasets/tfds/rot/)                     |
| ExoRL           | [Yarats et al., 2022]    | [exorl](corax/datasets/tfds/exorl/)                 |
| GWIL            | [Fickinger et al., 2022] | [gwil](corax/datasets/tfds/gwil/)                   |
| Adroit Binary   | [Nair et al., 2022]      | [adroit_binary](corax/datasets/tfds/adroit_binary/) |

NOTE: Some of these datasets do not yet cover all splits provided by the original
dataset. They will be added as the need arises.

## Acknowledgements
We would like to thank the [Acme](https://github.com/google-deepmind/acme) authors who
have provided a great starting point for Corax. Without them, Corax would not exist as a
significant portion of the current code is forked from them. You should check out Acme
if you are looking for more RL agent implementations.

We would like to thank the authors of the original papers for open-sourcing their code
which has been a great help in our re-implementation.

<!-- Agents -->
[Nakamoto et al., 2023]: https://arxiv.org/abs/2303.05479
[Chen et al., 2021a]: https://arxiv.org/abs/2106.01345
[Yarats et al., 2021]: https://arxiv.org/abs/2107.09645
[Zolna et al., 2020]: https://arxiv.org/pdf/2011.13885.pdf
[Kostrikov et al., 2021]: https://openreview.net/forum?id=68n2s9ZJWF8
[Chen et al., 2021b]: https://arxiv.org/abs/2101.05982
[Ball et al., 2023]: https://arxiv.org/abs/2302.02948
[Fujimoto et al., 2018]: https://arxiv.org/abs/1802.09477
[Fujimoto et al., 2021]: https://arxiv.org/abs/2106.06860.pdf
[Hansen et al., 2021]: https://arxiv.org/abs/2203.04955
[Kumar et al., 2020]: https://arxiv.org/abs/2006.04779
[Luo et al., 2023]: https://arxiv.org/abs/2303.13971

<!-- Papers -->
[Lu et al., 2023]: https://arxiv.org/abs/1806.06920
[Haldar et al., 2022]: https://openreview.net/forum?id=ZUtgUA0Fuwd
[Yarats et al., 2022]: https://arxiv.org/abs/2201.13425
[Fickinger et al., 2022]: https://arxiv.org/abs/2110.03684
[Nair et al., 2022]: https://arxiv.org/abs/2006.09359
