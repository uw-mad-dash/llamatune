# LlamaTune: Sample-Efficient DBMS Configuration Tuning

This repository contains the source code for the paper "[LlamaTune: Sample-Efficient DBMS Configuration Tuning](https://arxiv.org/abs/2203.05128)" (to appear in [VLDB'22](https://vldb.org/2022)). For more information regarding this project, please refer to the paper.

## Source Code Structure

LlamaTune uses [ConfigSpace](https://automl.github.io/ConfigSpace/) to define the DBMS configuration space, and employs custom *adapters* to perform the search space transformations described in the paper.

- `adapters/`
  - `configspace/`
    - `low_embeddings.py`: random linear projections (**Sec. 3**).
    - `quantization.py`: bucketization of knob values ranges (**Sec. 4.2**).
  - `bias_sampling.py`: definition and handling of hybrid knobs (**Sec. 4.1**).
- `configs/`
  - `benchbase/*`: experiment config files for BenchBase workloads
  - `ycsb/*`: experiment config files for YCSB workloads
- `ddpg/*`: definition of the actor-critic DDPG neural network architecture used in CDBTune.
- `executors/`: auxiliary code to run workload in the DBMS throught an in-house executor framework.
- `spaces/`:
  - `definitions/*`: PostgreSQL knob definition for v9.6 and v13.6.
- `config.py`: config file parsing & workload options definition
- `optimizer.py`: initialization of the BO-based or RL-based optimizer.
- `run-ddpg.py`: main script used for experiments for **BO-based** optimizers
- `run-smac.py`: main script used for experiments for **RL-based** optimizer
- `space.py`: definition of the input space that is fed to the underlying optimizer.
- `storage.py`: auxiliary code to store the result of the evaluated configurations.

**NOTE**: The complete LlamaTune pipeline, which includes low-dimensional tuning, hybrid knob handling, and large knob values bucketization is included in `adapters/configspace/low_embeddings.py`.

## Environment

We ran our experiments on [Cloudlab](https://cloudlab.us), on nodes of type`c220g5`. We used PostgreSQL v9.6 and v13.6 as our target DBMS.

Source code requires Python 3.7+ for running. To install the required packages run the following command:

```bash
pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Run Experiments

Main scripts used for running experiments are `run-smac.py` (for BO methods) and `run-ddpg.py` (for RL method). These can be invoked using the following syntax:

```bash
python3 run-smac.py <config_file> <seed>
```

where `<config_file>` is an experiment configuration file found in `config/*` directory, and `<seed>` is the random seed used by the optimizer. In our experiments, we used the same five different seeds (i.e., `1-5`).

Configuration files used for producing the same figure are typically groupped together in the same directory. For instance, this holds true for the sensitivity analysis performed for the low-dimensional projections, biased sampling, and bucketization techniques.

### Usage Examples

To run the "vanilla" SMAC on all workloads (for a single seed), one should run the following commands:

```bash
python3 run-smac.py configs/ycsb/ycsbA/ycsbA.all.ini 1
python3 run-smac.py configs/ycsb/ycsbB/ycsbB.all.ini 1
python3 run-smac.py configs/benchbase/tpcc/tpcc.all.ini 1
python3 run-smac.py configs/benchbase/seats/seats.all.ini 1
python3 run-smac.py configs/benchbase/twitter/twitter.all.ini 1
python3 run-smac.py configs/benchbase/resourcestresser/resourcestresser.all.ini 1
```

Similarly, to employ LlamaTune's search space transformations, while using SMAC as the undrlying optimizer, the following configuration files should be used:

```bash
python3 run-smac.py configs/ycsb/ycsbA/ycsbA.all.llama.ini 1
python3 run-smac.py configs/ycsb/ycsbB/ycsbB.all.llama.ini 1
python3 run-smac.py configs/benchbase/tpcc/tpcc.all.llama.ini 1
python3 run-smac.py configs/benchbase/seats/seats.all.llama.ini 1
python3 run-smac.py configs/benchbase/twitter/twitter.all.llama.ini 1
python3 run-smac.py configs/benchbase/resourcestresser/resourcestresser.all.llama.ini 1
```

One can also use the RL-based optimizer (i.e., DDPG) by replacing `run-smac.py` with `run-ddpg.py`.

Please note that this repository does not include the scripts used to evaluate the configurations suggested by the optimizer on the DBMS.
