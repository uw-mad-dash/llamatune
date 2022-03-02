import json
import logging
import os
import random
import subprocess
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cityblock

import grpc
from google.protobuf.struct_pb2 import Struct
from google.protobuf.json_format import MessageToDict

from sklearn.preprocessing import StandardScaler

# pylint: disable=import-error
from executors.grpc.nautilus_rpc_pb2 import ExecuteRequest, EmptyMessage
from executors.grpc.nautilus_rpc_pb2_grpc import ExecutionServiceStub

from mlos.Logger import create_logger
logger = create_logger(__name__, logging_level=logging.INFO)

def run_command(cmd, **kwargs):
    logger.debug(f'Running command: `{cmd}`...')
    logger.debug(50 * '=')

    cp = None
    try:
        cp = subprocess.run(cmd, shell=True, **kwargs)
        if cp.returncode != 0:
            logger.warn(f'Non-zero code [{cp.returncode}] for command `{cmd}`')

    except Exception as err:
        logger.error(err)
        logger.error(f'Error while running command `{cmd}`')

    return cp

def trim_disks():
    logger.info('Executing TRIM on all mount points')
    try:
        run_command('sudo fstrim -av', check=True)
    except Exception as err:
        logger.warn(f'Error while TRIMing: {repr(err)}')

def get_measured_performance(perf_stats, benchmark):
    """ Return throughput & 95-th latency percentile """
    if benchmark == 'ycsb':
        overall_stats = perf_stats['ycsb']['groups']['overall']['statistics']
        throughput, runtime = (
            overall_stats['Throughput(ops/sec)'],
            overall_stats['RunTime(ms)'] / 1000.0)
        # Manually compute latency (weighted by ops)
        groups = [
            g for name, g in perf_stats['ycsb']['groups'].items()
            if name != 'overall'
        ]
        latency_info = [ # latencies are in micro-seconds
            (float(g['statistics']['p95']), int(g['statistics']['Return=OK']))
            for g in groups
        ]
        latencies, weights = tuple(zip(*latency_info))
        latency = np.average(latencies, weights=weights) / 1000.0

    elif benchmark == 'oltpbench':
        summary_stats = perf_stats['oltpbench_summary']
        throughput, latency, runtime = (
            summary_stats['throughput(req/sec)'],
            summary_stats['95th_lat(ms)'],
            summary_stats['time(sec)'])
    elif benchmark == 'benchbase':
        summary_stats = perf_stats['benchbase_summary']
        throughput, latency, runtime = (
            summary_stats['throughput(req/sec)'],
            summary_stats['95th_lat(ms)'],
            summary_stats['time(sec)'])
    else:
        raise NotImplementedError(f'Benchmark `{benchmark}` is not supported')

    return {
        'throughput': throughput,
        'latency': latency,
        'runtime': runtime,
    }

def is_result_valid(results, benchmark):
    # Check results
    run_info, perf_stats = results['run_info'], results['performance_stats']

    if benchmark == 'ycsb':
        check_fields = [
            run_info['warm_up']['result'],
            run_info['benchmark']['result'],
            perf_stats['ycsb_result'],
            perf_stats['ycsb_raw_result'],
        ]
    elif benchmark == 'oltpbench':
        check_fields = [
            run_info['benchmark']['result'],
            perf_stats['oltpbench_summary_result'],
        ]
    elif benchmark == 'benchbase':
        check_fields = [
            run_info['benchmark']['result'],
            perf_stats['benchbase_summary_result'],
        ]
    else:
        raise NotImplementedError(f'Benchmark `{benchmark}` is not supported')

    return all(v == 'ok' for v in check_fields)

class ExecutorInterface(ABC):
    def __init__(self, spaces, storage, **kwargs):
        self.spaces = spaces
        self.storage = storage

    @abstractmethod
    def evaluate_configuration(self, dbms_info, benchmark_info):
        raise NotImplementedError

class DummyExecutor(ExecutorInterface):
    def evaluate_configuration(self, dbms_info, benchmark_info):
        return {
            'throughput': float(random.randint(1000, 10000)),
            'latency': float(random.randint(1000, 10000)),
            'runtime': 0,
        }

class NautilusExecutor(ExecutorInterface):
    GRPC_MAX_MESSAGE_LENGTH = 32 * (2 ** 20) # 32MB
    # NOTE: Nautilus already has a soft time limit (default is *1.5 hour*)
    EXECUTE_TIMEOUT_SECS = 4 * 60 * 60 # 4 hours

    def __init__(self, spaces, storage, host=None, port=None, n_retries=10):
        super().__init__(spaces, storage)

        self.host, self.port = host, port
        self.iter = 0

        delay = 2
        for idx in range(1, n_retries + 1):
            logger.debug(f'Trying connecting to Nautilus [#={idx}]...')
            try:
                self._try_connect(timeout=5)
            except Exception as err:
                logger.debug(f'Failed to connect: {repr(err)}')
                logger.debug(f'Trying again in {delay} seconds')
                time.sleep(delay)
                delay *= 2
            else:
                logger.info('Connected to Nautilus!')
                return

        raise RuntimeError(f'Cannot connect to Nautilus @ {host}:{port}')

    def evaluate_configuration(self, dbms_info, benchmark_info):
        """ Call Nautilus executor RPC """
        # trim disks before sending request
        trim_disks()

        # NOTE: protobuf explicitely converts ints to floats; this is a workaround
        # https://stackoverflow.com/questions/51818125/how-to-use-ints-in-a-protobuf-struct
        config = { }
        for k, v in dbms_info['config'].items():
            if isinstance(v, int):
                config[k] = str(v)
            else: config[k] = v
        dbms_info['config'] = config

        # Construct request
        config = Struct()
        config.update(dbms_info['config']) # pylint: disable=no-member
        dbms_info = ExecuteRequest.DBMSInfo(
            name=dbms_info['name'], config=config)

        if benchmark_info['name'] == 'ycsb':
            request = ExecuteRequest(
                dbms_info=dbms_info, ycsb_info=benchmark_info)
        elif benchmark_info['name'] == 'oltpbench':
            request = ExecuteRequest(
                dbms_info=dbms_info, oltpbench_info=benchmark_info)
        elif benchmark_info['name'] == 'benchbase':
            request = ExecuteRequest(
                dbms_info=dbms_info, benchbase_info=benchmark_info)
        else:
            raise ValueError(f"Benchmark `{benchmark_info['name']}' not found")

        # Do RPC
        logger.debug(f'Calling Nautilus RPC with request:\n{request}')
        response = self.stub.Execute(request, timeout=self.EXECUTE_TIMEOUT_SECS)

        logger.info(f'Received response JSON [len={len(response.results)}]')
        results = MessageToDict(response)['results']

        # Save results
        self.storage.store_executor_result(self.iter, results)
        self.iter += 1

        # Check results
        try:
            is_valid = is_result_valid(results, benchmark_info['name'])
        except Exception as err:
            logger.error(f'Exception while trying to check result: {str(err)}')
            is_valid = False
        finally:
            if not is_valid:
                logger.error('Nautilus experienced an error.. check logs :(')
                return None

        # Retrieve throughput & latency stats
        return get_measured_performance(
                results['performance_stats'], benchmark_info['name'])

    def close(self):
        """ Close connection to Nautilus """
        self.channel.close()

    def _try_connect(self, **kwargs):
        """ Attempt to connect to host:port address """
        self.channel = grpc.insecure_channel(
            f'{self.host}:{self.port}',
            options=[ # send/recv up to 32 MB of messages (4MB default)
                ('grpc.max_send_message_length', self.GRPC_MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', self.GRPC_MAX_MESSAGE_LENGTH),
            ])
        self.stub = ExecutionServiceStub(self.channel)

        response = self.stub.Heartbeat(EmptyMessage(), **kwargs)
        logger.info(f'{10*"="} Nautilus Info {10*"="}')
        logger.info(f'Alive since  : {response.alive_since.ToDatetime()}')
        logger.info(f'Current time : {response.time_now.ToDatetime()}')
        logger.info(f'Jobs finished: {response.jobs_finished}')
        logger.info(f'{35 * "="}')

class QueryFromDatasetExecutor(ExecutorInterface):
    def __init__(self, spaces, storage, dataset=None):
        super().__init__(spaces, storage)
        assert dataset != None, 'Please provide dataset filepath'

        filepath = Path('./datasets') / dataset
        assert filepath.exists(), f'Dataset filepath [@{filepath}] does not exist'

        configs, perfs = self._parse(filepath)

        # Find which knobs are numeric, and which categorical
        self.numerical_knobs = np.array([ col for col in configs.columns
            if pd.api.types.is_numeric_dtype(configs[col]) ])
        self.categorical_knobs = np.array([ col for col in configs.columns
            if pd.api.types.is_string_dtype(configs[col]) ])
        assert (len(self.numerical_knobs) + len(self.categorical_knobs) == len(configs.columns))

        logger.info(f'Numerical knobs: {self.numerical_knobs}')
        logger.info(f'Categorical knobs: {self.categorical_knobs}')

        # save samples
        self._samples = pd.concat([configs, perfs], axis=1)
        self.samples_used = np.zeros(len(self.samples), dtype=np.bool)

        # pre-compute normalization of numerical samples for easier closest-point search
        self._store_normalize_numerical_knobs_of_samples(configs, self.numerical_knobs)

        # default values to be used for pruned knobs
        self.default_config = spaces.get_default_config_point()

    def _parse(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        logger.info(f'Read {len(lines)} samples from dataset [@ {filepath}]')

        self.benchmark_info = None
        configs, perfs = [ ], [ ]

        confs_seen = set()

        # Retrieve config, performance tuples
        for result in map(json.loads, lines):
            dbms_config = result['task_args']['dbms']['config']
            if dbms_config == None: # default configuration
                continue

            benchmark_info = result['task_args']['benchmark']
            self.benchmark_info = self.benchmark_info or benchmark_info
            if self.benchmark_info != benchmark_info: # TODO: deepdiff
                logger.error('Benchmark info differs across samples -- skip sample')
                continue

            # Check result
            if not is_result_valid(result, benchmark_info['name']):
                logger.warning('Skipping non-valid result... :(')
                continue

            # Check if we have already seen this conf
            conf_fs = frozenset(dbms_config.items())
            if conf_fs in confs_seen:
                logger.warning('Skipping already seen configuration...')
                continue
            confs_seen |= { conf_fs }

            dbms_config = self.spaces.unfinalize_conf(dbms_config)
            configs.append(dbms_config)

            # Retrieve throughput
            throughput, _ = get_measured_throughput(
                    result['performance_stats'], benchmark_info['name'])
            perfs.append(throughput)

        configs, perfs = pd.DataFrame(configs), pd.DataFrame(perfs, columns=['throughput'])
        logger.info(f'Found {len(configs)} valid samples!')

        return configs, perfs

    def _store_normalize_numerical_knobs_of_samples(self, samples, knobs):
        logger.info(f'(Re)normalizing {len(samples)} samples...')
        logger.info(f'Keeping following numerical knobs: {knobs}')

        # normalize samples (used for distance computation)
        scaler = StandardScaler()
        scaled_configs = scaler.fit_transform(samples[knobs])
        self.scaled_samples = pd.DataFrame(data=scaled_configs, columns=knobs)

        # store scaler & number of num knobs
        self.scaler = scaler
        self.n_numerical_knobs = len(knobs)

    @property
    def samples(self):
        return self._samples

    def evaluate_configuration(self, dbms_info, benchmark_info):
        #if self.benchmark_info != benchmark_info:
        #    logger.error('Benchmark info from dataset differs from the provided\n'
        #        f'Dataset:\t{self.benchmark_info}\nExperiment:\t{benchmark_info}')
        start = datetime.now()

        # remove suffixes, string to numeric, etc
        config = self.spaces.unfinalize_conf(dbms_info['config'])

        # find which knobs are tuned (may be less than all due to pruning)
        categorical_knobs = [ col for col in self.categorical_knobs if col in config ]
        numerical_knobs = [ col for col in self.numerical_knobs if col in config ]

        # NOTE: old way of handling pruned knobs -- i.e. re-train scaler
        #if len(numerical_knobs) < self.n_numerical_knobs:
        #    # pruning occured -- need to update scaler & re-normalize samples
        #    self._store_normalize_numerical_knobs_of_samples(
        #        self.samples[numerical_knobs], numerical_knobs)

        # fill pruned/missing knob values with default ones
        missing_numeric_knobs = set(self.numerical_knobs) - set(numerical_knobs)
        for knob in missing_numeric_knobs:
            config[knob] = self.default_config[knob]
        config = pd.Series(config)

        if len(missing_numeric_knobs) > 0:
            logger.info(f'Filled config with default values:\n{config}')
            numerical_knobs = self.numerical_knobs

        # normize given config values
        scaled_values = self.scaler.transform([ config[numerical_knobs] ])[0]
        scaled_config = pd.Series(data=scaled_values, index=numerical_knobs)

        # consider samples where all categorical knobs match to given config
        samples = self.samples
        for col in categorical_knobs:
            samples = samples.loc[samples[col] == config[col]]
        logger.info(f'Found {len(samples)} candidate points after fixing categorical knobs\' values')

        # find "closest" previously evaluated sample
        closest_dist, closest_idx = None, None
        for idx, _ in samples.iterrows():
            if self.samples_used[idx]:
                # skip already used point
                continue

            # compute distance
            total_dist = euclidean(
                self.scaled_samples.iloc[idx][numerical_knobs].values,
                scaled_config[numerical_knobs].values
            )
            # update closer point
            if (closest_dist == None) or (total_dist < closest_dist):
                closest_dist, closest_idx, = total_dist, idx

        # Find and mark out used point
        closest_sample = self.samples.iloc[closest_idx]
        self.samples_used[closest_idx] = True

        logger.info(f'Closest sample is:\n{closest_sample}')
        logger.info(f'L2-norm Distance is: {closest_dist: .3f}')

        return {
            'sample': self.spaces.point_from_dict({ k: v
                for k, v in zip(closest_sample.index.values, closest_sample.values)
                if k != 'throughput'}),
            'throughput': closest_sample['throughput'],
            'runtime': (datetime.now() - start).total_seconds(),
        }


class ExecutorFactory:
    concrete_classes = {
        'NautilusExecutor': NautilusExecutor,
        'QueryFromDatasetExecutor': QueryFromDatasetExecutor,
        'DummyExecutor': DummyExecutor,
    }

    @staticmethod
    def from_config(config, spaces, storage, **extra_kwargs):
        executor_config = deepcopy(config['executor'])

        classname = executor_config.pop('classname', None)
        assert classname != None, 'Please specify the *executor* class name'

        try:
            class_ = ExecutorFactory.concrete_classes[classname]
        except KeyError:
            raise ValueError(f'Executor class "{classname}" not found. '
            f'Options are [{", ".join(ExecutorFactory.concrete_classes.keys())}]')

        # Override with local
        executor_config.update(**extra_kwargs)

        return class_(spaces, storage, **executor_config)
