# Randomized Ensembled Double Q-Learning: Learning Fast Without a Model

This folder contains an implementation of Randomized Ensembled Double Q-Learning (REDQ)
introduced in ([Chen et al., 2021]). This implementation can be set up to
also learn from an additional offline dataset as in the RLPD paper ([Ball et al., 2023]).

Notes:
- In the REDQ paper, the author uses the terminology of Update-To-Data (UTD) ratio.
In this implemenation, we use the num_sgd_steps_per_step parameter to control the UTD ratio.
The samples-per-insert (SPI) parameter used by Reverb needs to be increased accordingly
to reflect doing more SGD steps per insert.

[Chen et al., 2021]: https://arxiv.org/abs/2101.05982
[Ball et al., 2023]: https://arxiv.org/abs/2302.02948
