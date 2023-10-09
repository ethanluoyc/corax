# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX experiment utils."""

from baselines.experiments.config import CheckpointingConfig
from baselines.experiments.config import DeprecatedPolicyFactory
from baselines.experiments.config import EvaluatorFactory
from baselines.experiments.config import ExperimentConfig
from baselines.experiments.config import MakeActorFn
from baselines.experiments.config import NetworkFactory
from baselines.experiments.config import OfflineExperimentConfig
from baselines.experiments.config import PolicyFactory
from baselines.experiments.config import default_evaluator_factory
from baselines.experiments.config import make_policy
from baselines.experiments.imitation_experiment import ImitationExperimentConfig
from baselines.experiments.imitation_experiment import run_imitation_experiment
from baselines.experiments.imitation_loop import EpisodeRewarder
from baselines.experiments.imitation_loop import ImitationEnvironmentLoop
from baselines.experiments.run_experiment import run_experiment
from baselines.experiments.run_offline_experiment import run_offline_experiment
