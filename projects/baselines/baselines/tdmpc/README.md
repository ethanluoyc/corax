# TD-MPC Experiments

This folder contains experiment launchers for running TD-MPC on dm_control.

To run the experiments on dm_control, run the following.

```shell
python -m cdrl.examples.tdmpc.main \
  --config cdrl/examples/tdmpc/configs/walker.py \
  --config.task=walker-walk
```

See [configs/](./configs/) for configurations for other environments.
