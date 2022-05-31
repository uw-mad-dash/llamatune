## Supplementary Material for "LlamaTune: Sample-Efficient DBMS Configuration Tuning"

Core functionality is implemented in the following files:

`adapters/bias_sampling.py`: hybrid knobs definition

`adapters/condigspace/low_embeddings.py`: random linear projections (can also be combined with bucketization and biased sampling)

`adapters/configspace/quantization.py`: knob values bucketization

`space.py`: constructs the optimizer's input space

`optimizer.py`: optimizer initialization

`run-smac.py`: main script used for experiments for BO-based optimizers

`run-ddpg.py`: main script used for experiments for RL-based optimizer
