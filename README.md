## Supplementary Material for "LlamaTune: Sample-Efficient DBMS Configuration Tuning"

You can find our code at this url: https://github.com/uw-mad-dash/llamatune/tree/march2022

Core functionality is implemented in the following files:

`adapters/bias_sampling.py`: hybrid knobs definition

`adapters/condigspace/low_embeddings.py`: random linear projections (can also be combined with bucketization and biased sampling)

`adapters/configspace/quantization.py`: knob values bucketization

`space.py`: constructs the optimizer's input space

`optimizer.py`: optimizer initialization

`run-smac.py`: main script used for experiemnts (feedback loop)

