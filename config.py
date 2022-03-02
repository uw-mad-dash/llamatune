import logging
from configparser import SafeConfigParser

# pylint: disable=import-error
from mlos.Logger import create_logger
logger = create_logger(__name__, logging_level=logging.INFO)

# YCSB
ycsb_defaut_workload_properties = dict(
    recordcount=18000000,
    operationcount=2000000000,
    threadcount=40
)

# OLTP-Bench
oltpbench_default_workload_properties = {
    # TODO: crashes for PSQL
    'auctionmark': dict(
        scalefactor=40,
        terminals=40,
        work_rate='unlimited'
    ),
    'seats': dict(
        scalefactor=50, # ~20 GB
        terminals=40,
        work_rate='unlimited'
    ),
    'tpcc': dict(
        scalefactor=100, # ~20-22 GB
        terminals=40,
        work_rate='unlimited'
    ),
    'twitter': dict(
        scalefactor=1500, # ~20 GB
        terminals=40,
        work_rate='unlimited'
    ),
    'wikipedia': dict(
        scalefactor=65, # ~22 GB
        terminals=40,
        work_rate='unlimited'
    ),
    'ycsb': dict(
        # NOTE: if crashes try lower scalefactor values (5K worked for PSQL)
        scalefactor=9000,
        terminals=40,
        work_rate='unlimited'
    )
}

# Benchbase
benchbase_default_workload_properties = {
    # CRASHES
    'auctionmark': dict(
        scalefactor=50, # ~20 GB
        terminals=40,
        work_rate='unlimited'
    ),
    # CRASHES
    'epinions': dict(
        scalefactor=100, # ~5.5 GB
        terminals=40,
        work_rate='unlimited'
    ),
    'resourcestresser': dict(
        scalefactor=20000,  # ~20 GB
        terminals=40,
        work_rate='unlimited',
    ),
    'seats': dict(
        scalefactor=50, # ~20 GB
        terminals=40,
        work_rate='unlimited'
    ),
    # CRASHES
    'smallbank': dict(
        scalefactor=25, # ~22 GB
        terminals=40,
        work_rate='unlimited'
    ),
    # CRASHES
    'tatp': dict(
        scalefactor=100, # ~20 GB
        terminals=40,
        work_rate='unlimited'
    ),
    'tpcc': dict(
        scalefactor=100, # ~20 GB
        terminals=40,
        work_rate='unlimited'
    ),
    'twitter': dict(
        scalefactor=1500, # ~20 GB
        terminals=40,
        work_rate='unlimited'
    ),
    # CRASHES
    'wikipedia': dict(
        scalefactor=65, # ~22 GB
        terminals=40,
        work_rate='unlimited'
    ),
    'ycsb': dict(
        scalefactor=9000, # ~19 GB
        terminals=40,
        work_rate='unlimited'
    ),

    # Rate-limited
    'tpcc-lat': dict(
        scalefactor=100, # ~20 GB
        terminals=40,
        work_rate="2000", # ~50% of max
    ),
    'seats-lat': dict(
        scalefactor=50, # ~20 GB
        terminals=40,
        work_rate="8000", # ~50% of max
    ),
    'twitter-lat': dict(
        scalefactor=1500, # ~20 GB
        terminals=40,
        work_rate="60000", # ~50% of max
    ),
    'resourcestresser-lat': dict(
        scalefactor=20000,  # ~20 GB
        terminals=40,
        work_rate="7",
    ),

}

workload_defaults = {
    'WARMUP_DURATION_SECS': 30,
    'BENCHMARK_DURATION_SECS': 300,
}

class Configuration:
    def __init__(self):
        self.parser = SafeConfigParser()
        self.parser.optionxform = str

        self.dict = { }

    def update_from_file(self, filename):
        if not self.parser.read(filename):
            raise FileNotFoundError(f'Config file [@{filename}] not found! :(')

        for name in self.parser.sections():
            self.dict[name] = dict(self.parser.items(name))

        self._postprocess()

    def _postprocess(self):
        # Save dbms_info
        assert 'dbms_info' in self.dict, 'Section `dbms_info` not specified'
        self._dbms_info = self.dict['dbms_info']

        # Save benchmark_info
        assert 'benchmark_info' in self.dict, 'Section `benchmark_info` not specified'
        self._benchmark_info = self.dict['benchmark_info']

        benchmark_name, workload = (
            self._benchmark_info['name'], self._benchmark_info['workload'])
        assert benchmark_name in ['ycsb', 'oltpbench', 'benchbase']

        if benchmark_name == 'ycsb':
            workload_properties = ycsb_defaut_workload_properties
        elif benchmark_name == 'oltpbench':
            if workload.startswith('ycsb'):
                workload_properties = oltpbench_default_workload_properties['ycsb']
            else:
                workload_properties = oltpbench_default_workload_properties[workload]
        else: # benchbase
            if workload.startswith('ycsb'):
                workload_properties = benchbase_default_workload_properties['ycsb']
            else:
                workload_properties = benchbase_default_workload_properties[workload]

        if workload.endswith('-lat'):
            # Provision for latency-focused configs
            workload = workload[:-len('-lat')]

        self._benchmark_info.update(
            workload=workload,
            warmup_duration=workload_defaults['WARMUP_DURATION_SECS'],
            benchmark_duration=workload_defaults['BENCHMARK_DURATION_SECS'],
            workload_properties=workload_properties,
        )
        if benchmark_name != 'ycsb':
            self._benchmark_info.update(
                capture_raw_perf_stats=False,
            )

        # global section
        self._iters = int(self.dict['global']['iters'])

    def __getitem__(self, key):
        return self.dict[key]

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

    @property
    def dbms_info(self):
        return self._dbms_info

    @property
    def benchmark_info(self):
        return self._benchmark_info

    @property
    def iters(self):
        return self._iters

config = Configuration()
