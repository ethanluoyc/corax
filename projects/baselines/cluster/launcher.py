import functools
import importlib
import os
import subprocess
import sys

from absl import app
from absl import flags
from absl import logging
from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.contrib import ucl
from ml_collections import config_flags

_LAUNCH_ON_CLUSTER = flags.DEFINE_boolean(
    "launch_on_cluster", False, "Launch on cluster"
)
_SINGULARITY_CONTAINER = flags.DEFINE_string(
    "container", "corax-latest.sif", "Path to singularity container"
)
_EXP_NAME = flags.DEFINE_string("exp_name", None, "Name of experiment")
_ENTRYPOINT = flags.DEFINE_string("entrypoint", None, "Module of the entrypoint")
_SWEEP = flags.DEFINE_string("sweep", None, "Name of the sweep")
_TIMEOUT = flags.DEFINE_integer("timeout", 2, "Timeout in hours")
_WANDB_GROUP = flags.DEFINE_string("wandb_group", "{xid}_{name}", "wandb group")
_WANDB_PROJECT = flags.DEFINE_string("wandb_project", "corax", "wandb project")
_WANDB_ENTITY = flags.DEFINE_string("wandb_entity", "ethanluoyc", "wandb entity")
_WANDB_MODE = flags.DEFINE_string("wandb_mode", "online", "wandb mode")
_MAX_NUM_JOBS = flags.DEFINE_integer("max_num_jobs", None, "Maximum number of jobs")
config_flags.DEFINE_config_file("config", None, "Path to config")
flags.mark_flags_as_required(["config", "entrypoint"])

FLAGS = flags.FLAGS


@functools.lru_cache()
def _get_vcs_info():
    vcs = None
    try:
        import vcsinfo

        vcs_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
        vcs = vcsinfo.detect_vcs(vcs_root)
    except subprocess.SubprocessError:
        logging.warn("Failed to detect VCS info")
    return vcs


def _get_wandb_env_vars(work_unit: xm.WorkUnit, experiment_name: str):
    xid = work_unit.experiment_id
    wid = work_unit.work_unit_id
    env_vars = {
        "WANDB_PROJECT": _WANDB_PROJECT.value,
        "WANDB_ENTITY": _WANDB_ENTITY.value,
        "WANDB_NAME": f"{experiment_name}_{xid}_{wid}",
        "WANDB_MODE": _WANDB_MODE.value,
        "WANDB_RUN_GROUP": _WANDB_GROUP.value.format(name=experiment_name, xid=xid),
    }
    vcs = _get_vcs_info()
    if vcs is not None:
        env_vars["WANDB_GIT_REMOTE_URL"] = vcs.upstream_repo
        env_vars["WANDB_GIT_COMMIT"] = vcs.id

    return env_vars


def _get_hyper():
    if _SWEEP.value is not None:
        sweep_file = config_flags.get_config_filename(FLAGS["config"])
        sys.path.insert(0, os.path.abspath(os.path.dirname(sweep_file)))
        sweep_module, _ = os.path.splitext(os.path.basename(sweep_file))
        m = importlib.import_module(sweep_module)
        sys.path.pop(0)
        sweep_fn_name = f"sweep_{_SWEEP.value}"
        logging.info(f"Running sweep {sweep_fn_name}")
        sweep_fn = getattr(m, sweep_fn_name, None)
        if sweep_fn is None:
            raise ValueError(f"Sweep {sweep_fn_name} does not exist in {sweep_file}")
        else:
            return sweep_fn()
    else:
        return [{}]


def main(_):
    exp_name = _EXP_NAME.value
    if exp_name is None:
        exp_name = _ENTRYPOINT.value.replace(".", "_")
    with xm_cluster.create_experiment(experiment_title=exp_name) as experiment:
        job_requirements = xm_cluster.JobRequirements(gpu=1, ram=8 * xm.GB)
        env_vars = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        if _LAUNCH_ON_CLUSTER.value:
            # TODO: Make this configurable for non-UCL clusters
            tfds_data_dir = "/cluster/project0/offline_rl/tensorflow_datasets"
            d4rl_dataset_dir = "/cluster/project0/offline_rl/d4rl"
            executor = ucl.UclGridEngine(
                job_requirements,
                walltime=_TIMEOUT.value * xm.Hr,
                singularity_options=xm_cluster.SingularityOptions(
                    bind={
                        tfds_data_dir: tfds_data_dir,
                        d4rl_dataset_dir: d4rl_dataset_dir,
                    }
                ),
            )
            env_vars["TFDS_DATA_DIR"] = tfds_data_dir
            env_vars["D4RL_DATASET_DIR"] = d4rl_dataset_dir
        else:
            executor = xm_cluster.Local(job_requirements)

        config_resource = xm_cluster.Fileset(
            files={config_flags.get_config_filename(FLAGS["config"]): "config.py"}
        )

        spec = xm_cluster.PythonPackage(
            path="..",
            entrypoint=xm_cluster.ModuleName(_ENTRYPOINT.value),
            extra_packages=[os.path.join(os.path.dirname(__file__), "../../../")],
            resources=[config_resource],
        )

        singularity_container = _SINGULARITY_CONTAINER.value

        if singularity_container:
            spec = xm_cluster.SingularityContainer(
                spec, image_path=singularity_container
            )

        args = {"config": config_resource.get_path("config.py", executor.Spec())}  # type: ignore
        overrides = config_flags.get_override_values(FLAGS["config"])
        overrides = {f"config.{k}": v for k, v in overrides.items()}
        logging.info("Overrides: %r", overrides)
        args.update(overrides)

        sweep = list(_get_hyper())
        if _MAX_NUM_JOBS.value is not None:
            sweep = sweep[: _MAX_NUM_JOBS.value]

        logging.info("Will launch %d jobs", len(sweep))

        [executable] = experiment.package(
            [
                xm.Packageable(
                    spec, executor_spec=executor.Spec(), args=args, env_vars=env_vars
                )
            ]
        )

        async def make_job(work_unit: xm.WorkUnit, **args):
            job = xm.Job(
                executable,
                executor,
                args=args,
                env_vars={**_get_wandb_env_vars(work_unit, exp_name)},
            )

            work_unit.add(job)

        with experiment.batch():
            for parameters in sweep:
                experiment.add(make_job, parameters)


if __name__ == "__main__":
    app.run(main)
